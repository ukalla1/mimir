"""End-to-end orchestration for the general nutrition assistant.

Wires together: parser -> DB search -> gap analysis -> DB nutrient query
-> GAT expansion -> preference re-ranking -> recommendation generation.
"""

from __future__ import annotations

import duckdb

from nutri_rag.config import DB_PATH, FOOD_EMBEDDINGS_PATH, USER_DB_PATH
from nutri_rag.parse import parse_meal
from nutri_rag.search import search_food, get_nutrients
from nutri_rag.llm import chat_completion
from nutri_rag.assistant.gap_analyzer import analyze_gap
from nutri_rag.assistant.food_recommender import FoodRecommender
from nutri_rag.assistant.preference_db import PreferenceDB
from nutri_rag.assistant.prompt import format_recommendation_prompt


class NutriAssistant:
    """Interactive nutrition assistant with GAT-based recommendations."""

    def __init__(
        self,
        db_path: str = DB_PATH,
        embeddings_path: str = FOOD_EMBEDDINGS_PATH,
        user_db_path: str = USER_DB_PATH,
    ):
        self._db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None
        self._recommender = FoodRecommender(db_path, embeddings_path)
        self._pref_db = PreferenceDB(user_db_path)

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(self._db_path, read_only=True)
        return self._con

    def _lookup_eaten_foods(
        self,
        meal_description: str,
        meal_type: str = "breakfast",
    ) -> list[dict]:
        """Parse and look up nutrients for foods the user has eaten."""
        items = parse_meal(meal_description)
        eaten = []

        for item in items:
            df = search_food(self.con, item.food_term, k=1)
            if len(df) == 0:
                continue

            fdc_id = int(df.iloc[0]["fdc_id"])
            description = df.iloc[0]["description"]
            nutrients = get_nutrients(self.con, fdc_id, key_only=True)

            eaten.append({
                "fdc_id": fdc_id,
                "description": description,
                "nutrients": nutrients,
                "quantity": item.quantity,
                "meal_type": meal_type,
            })

        return eaten

    def recommend(
        self,
        user_message: str,
        meal_history: list[dict] | None = None,
        eaten_description: str | None = None,
        eaten_meal_type: str = "breakfast",
        next_meal: str = "lunch",
        user_id: str = "default",
    ) -> str:
        """Generate a personalized meal recommendation.

        Args:
            user_message: The user's question (e.g., "What should I eat for lunch?")
            meal_history: Pre-built list of eaten food dicts. If None, uses eaten_description.
            eaten_description: Text describing what user ate (e.g., "apple and milk").
            eaten_meal_type: What meal the eaten food was for.
            next_meal: What meal to recommend.
            user_id: User ID for preference tracking.

        Returns:
            Natural language meal recommendation string.
        """
        # Step 1-2: Parse and look up what user ate
        if meal_history is None:
            if eaten_description:
                meal_history = self._lookup_eaten_foods(eaten_description, eaten_meal_type)
            else:
                meal_history = []

        if not meal_history:
            return ("I couldn't identify any foods from your description. "
                    "Could you describe what you ate more specifically?")

        # Step 3: LLM Call 1 — Gap analysis
        gap_result = analyze_gap(meal_history, next_meal=next_meal)
        reasoning = gap_result["reasoning"]
        targets = gap_result["targets"]

        # Step 4-5: DB nutrient query + GAT neighbor expansion
        eaten_fdc_ids = {item["fdc_id"] for item in meal_history}
        options = self._recommender.recommend(
            targets=targets,
            exclude_fdc_ids=eaten_fdc_ids,
        )

        if not options:
            return ("I found your nutritional gaps but couldn't find matching "
                    "foods in the database. Please try a different query.")

        # Step 6: Preference re-ranking
        fdc_ids = [opt.fdc_id for opt in options]
        pref_scores = self._pref_db.get_preference_scores(fdc_ids, user_id)
        for opt in options:
            opt.preference_score = pref_scores.get(opt.fdc_id, 0.5)

        # Sort: seeds first (sorted by preference), then neighbors (sorted by similarity)
        seeds = sorted(
            [o for o in options if o.is_seed],
            key=lambda o: o.preference_score,
            reverse=True,
        )
        neighbors = sorted(
            [o for o in options if not o.is_seed],
            key=lambda o: o.gat_similarity,
            reverse=True,
        )
        ranked_options = seeds + neighbors

        # Step 7: Get preference summary for prompt context
        pref_summary = self._pref_db.get_preference_summary(user_id)

        # Step 8: LLM Call 2 — Generate recommendation
        messages = format_recommendation_prompt(
            gap_reasoning=reasoning,
            targets=targets,
            options=ranked_options,
            next_meal=next_meal,
            preference_summary=pref_summary if pref_summary["favorites"] else None,
        )

        recommendation = chat_completion(messages, max_tokens=1024)

        # Record offered foods in preference DB
        offered = [{"fdc_id": o.fdc_id, "description": o.description}
                   for o in ranked_options[:10]]
        self._pref_db.record_offered_foods(
            meal_type=next_meal,
            options=offered,
            user_id=user_id,
        )

        return recommendation

    def record_user_choice(
        self,
        meal_type: str,
        chosen_fdc_ids: set[int],
        user_id: str = "default",
    ):
        """Record which foods the user actually chose from recommendations."""
        # Update the most recent offered foods as chosen
        for fdc_id in chosen_fdc_ids:
            self._pref_db.record_choice(
                meal_type=meal_type,
                fdc_id=fdc_id,
                description="",  # will be looked up from history
                offered=True,
                chosen=True,
                user_id=user_id,
            )

    def close(self):
        if self._con:
            self._con.close()
        self._pref_db.close()
