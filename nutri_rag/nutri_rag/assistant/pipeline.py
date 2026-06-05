"""End-to-end orchestration for the general nutrition assistant.

Wires together: parser -> DB search -> gap analysis -> DB nutrient query
-> GAT expansion -> preference re-ranking -> recommendation generation.
"""

from __future__ import annotations

import os

import duckdb

from nutri_rag.config import DB_PATH, FOOD_EMBEDDINGS_PATH, USER_DB_PATH
from nutri_rag.parse import parse_meal
from nutri_rag.search import search_food, search_food_v2, get_nutrients
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
        eaten_retrieval_mode: str | None = None,
    ):
        self._db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None
        self._recommender = FoodRecommender(db_path, embeddings_path)
        self._pref_db = PreferenceDB(user_db_path)
        # Lazy: MealRecommender only loads recipe embeddings when activated
        self._meal_recommender = None
        # Eaten-food identification mode:
        #   "hybrid"    — score-fusion via pseudo-anchor (new default, Phase A)
        #   "text_top1" — legacy text top-1 (kept as opt-in fallback)
        # Override via EATEN_RETRIEVAL_MODE env var or constructor arg.
        self._eaten_mode = eaten_retrieval_mode or os.environ.get(
            "EATEN_RETRIEVAL_MODE", "hybrid"
        )
        # nutrition→Food recommendation mode (Phase C):
        #   "v1" — legacy seed selection + neighbor expansion (current default)
        #   "v2" — target-as-query hybrid (Phase C addition)
        # Override via RECOMMEND_MODE env var.
        self._recommend_mode = os.environ.get("RECOMMEND_MODE", "v1")
        # Phase D meal-layer composition:
        #   "off" — pipeline returns Phase C food-level recommendations only
        #   "on"  — additionally retrieve top-K recipes from the recommended
        #           foods + pantry and pass them to the LLM
        # Default off — opt-in, preserves Phase C behavior.
        self._meal_compose_mode = os.environ.get("MEAL_COMPOSE_MODE", "off")

    def _get_meal_recommender(self):
        """Lazy-init MealRecommender (avoids loading recipe embeddings unless used)."""
        if self._meal_recommender is None:
            from nutri_rag.assistant.meal_recommender import MealRecommender
            self._meal_recommender = MealRecommender(db_path=self._db_path)
        return self._meal_recommender

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
        """Parse and look up nutrients for foods the user has eaten.

        Retrieval mode is selected by self._eaten_mode (see __init__).
        """
        items = parse_meal(meal_description)
        eaten = []

        for item in items:
            if self._eaten_mode == "text_top1":
                # Legacy path — kept as opt-in fallback for A/B comparison.
                df = search_food(self.con, item.food_term, k=1)
            else:
                # Default (hybrid via pseudo-anchor) — Phase A new behavior.
                # Same code path as NutriBench v5 (scripts/run_bench.py --mode v5):
                # v5 acc/MAE predicts this stage's quality, no separate robot
                # regression test needed. See plans/vectorized-twirling-valley.md.
                df = search_food_v2(
                    item.food_term, mode="hybrid", k=1, alpha=0.5,
                    db_path=self._db_path,
                )
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
        available_fdc_ids: set[int] | None = None,
    ) -> str:
        """Generate a personalized meal recommendation.

        Args:
            user_message: The user's question (e.g., "What should I eat for lunch?")
            meal_history: Pre-built list of eaten food dicts. If None, uses eaten_description.
            eaten_description: Text describing what user ate (e.g., "apple and milk").
            eaten_meal_type: What meal the eaten food was for.
            next_meal: What meal to recommend.
            user_id: User ID for preference tracking.
            available_fdc_ids: Optional hard availability filter applied to
                both seeds and expanded neighbors. None = no filter (current
                behavior). Phase B addition.

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
        # Dispatch v1 (current) vs v2 (target-as-query hybrid) via env var.
        eaten_fdc_ids = {item["fdc_id"] for item in meal_history}
        if self._recommend_mode == "v2":
            options = self._recommender.recommend_v2(
                targets=targets,
                exclude_fdc_ids=eaten_fdc_ids,
                available_fdc_ids=available_fdc_ids,
            )
        else:
            options = self._recommender.recommend(
                targets=targets,
                exclude_fdc_ids=eaten_fdc_ids,
                available_fdc_ids=available_fdc_ids,
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

        # Phase D: optional meal-layer composition.
        # When MEAL_COMPOSE_MODE=on, retrieve concrete recipes from the
        # recommended foods + pantry and include them in the LLM prompt.
        # Uses the same hybrid_rank_recipes primitive that PFoodReq's
        # --retrieval-style embedding_first mode exercises — so PFoodReq
        # scores are a regression signal for this step.
        meal_candidates = None
        if self._meal_compose_mode == "on" and ranked_options:
            try:
                # Negative signal = user dislikes / allergies (NOT eaten foods —
                # users can repeat what they ate). PreferenceDB exposes these
                # as USDA-style descriptions; MealRecommender matches them by
                # substring against recipe ingredient strings.
                disliked = pref_summary.get("disliked", []) if pref_summary else None
                meal_candidates = self._get_meal_recommender().recommend_meal(
                    recommended_foods=ranked_options[:10],
                    targets=targets,
                    available_fdc_ids=available_fdc_ids,
                    next_meal=next_meal,
                    disliked_names=disliked,
                )
            except Exception as e:
                # Graceful degradation: log and continue with food-only output
                print(f"[pipeline] meal composition failed ({e}); falling back to food-level")
                meal_candidates = None

        # Step 8: LLM Call 2 — Generate recommendation
        messages = format_recommendation_prompt(
            gap_reasoning=reasoning,
            targets=targets,
            options=ranked_options,
            next_meal=next_meal,
            preference_summary=pref_summary if pref_summary["favorites"] else None,
            meal_candidates=meal_candidates,
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
