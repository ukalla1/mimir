"""User preference database for tracking food choice history.

Records which foods were offered and which the user chose, enabling
personalized re-ranking of future recommendations.
"""

from __future__ import annotations

import duckdb

from nutri_rag.config import USER_DB_PATH


class PreferenceDB:
    """DuckDB-backed user food preference tracker."""

    def __init__(self, db_path: str = USER_DB_PATH):
        self._con = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS user_choices (
                id          INTEGER PRIMARY KEY,
                timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id     VARCHAR DEFAULT 'default',
                meal_type   VARCHAR,
                food_fdc_id BIGINT,
                food_desc   VARCHAR,
                offered     BOOLEAN,
                chosen      BOOLEAN
            )
        """)
        # Auto-increment via sequence
        self._con.execute("""
            CREATE SEQUENCE IF NOT EXISTS user_choices_id_seq START 1
        """)

    def record_choice(
        self,
        meal_type: str,
        fdc_id: int,
        description: str,
        offered: bool = True,
        chosen: bool = False,
        user_id: str = "default",
    ):
        """Log a user's food choice decision."""
        self._con.execute("""
            INSERT INTO user_choices (id, user_id, meal_type, food_fdc_id, food_desc, offered, chosen)
            VALUES (nextval('user_choices_id_seq'), ?, ?, ?, ?, ?, ?)
        """, [user_id, meal_type, fdc_id, description, offered, chosen])

    def record_offered_foods(
        self,
        meal_type: str,
        options: list[dict],
        chosen_fdc_ids: set[int] | None = None,
        user_id: str = "default",
    ):
        """Batch-record offered foods and which were chosen.

        Args:
            meal_type: breakfast/lunch/dinner/snack
            options: List of dicts with fdc_id and description
            chosen_fdc_ids: Set of fdc_ids the user selected
        """
        chosen = chosen_fdc_ids or set()
        for opt in options:
            self.record_choice(
                meal_type=meal_type,
                fdc_id=opt["fdc_id"],
                description=opt["description"],
                offered=True,
                chosen=opt["fdc_id"] in chosen,
                user_id=user_id,
            )

    def get_preference_scores(
        self,
        fdc_ids: list[int],
        user_id: str = "default",
    ) -> dict[int, float]:
        """Get preference scores (chosen/offered ratio) for given foods.

        Returns dict mapping fdc_id -> score (0.0 to 1.0).
        Foods never seen get 0.5 (neutral).
        """
        if not fdc_ids:
            return {}

        placeholders = ",".join(str(int(fid)) for fid in fdc_ids)
        df = self._con.execute(f"""
            SELECT food_fdc_id,
                   SUM(CASE WHEN chosen THEN 1 ELSE 0 END) AS chosen_count,
                   SUM(CASE WHEN offered THEN 1 ELSE 0 END) AS offered_count
            FROM user_choices
            WHERE user_id = ?
              AND food_fdc_id IN ({placeholders})
            GROUP BY food_fdc_id
        """, [user_id]).df()

        scores = {int(fid): 0.5 for fid in fdc_ids}  # default neutral
        for _, row in df.iterrows():
            fid = int(row["food_fdc_id"])
            offered = int(row["offered_count"])
            if offered > 0:
                scores[fid] = float(row["chosen_count"]) / offered

        return scores

    def get_history(
        self,
        user_id: str = "default",
        limit: int = 20,
    ) -> list[dict]:
        """Get recent food choice history."""
        df = self._con.execute("""
            SELECT timestamp, meal_type, food_fdc_id, food_desc, offered, chosen
            FROM user_choices
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, [user_id, limit]).df()

        return df.to_dict("records")

    def get_preference_summary(self, user_id: str = "default") -> dict:
        """Get a summary of user preferences for prompt context.

        Returns dict with 'favorites' (high chosen rate) and
        'disliked' (offered but never chosen).
        """
        df = self._con.execute("""
            SELECT food_desc,
                   SUM(CASE WHEN chosen THEN 1 ELSE 0 END) AS chosen_count,
                   SUM(CASE WHEN offered THEN 1 ELSE 0 END) AS offered_count
            FROM user_choices
            WHERE user_id = ?
            GROUP BY food_desc, food_fdc_id
            HAVING offered_count >= 2
            ORDER BY chosen_count DESC
        """, [user_id]).df()

        favorites = []
        disliked = []
        for _, row in df.iterrows():
            ratio = row["chosen_count"] / row["offered_count"]
            if ratio >= 0.5:
                favorites.append(row["food_desc"])
            elif ratio == 0:
                disliked.append(row["food_desc"])

        return {"favorites": favorites, "disliked": disliked}

    def close(self):
        self._con.close()
