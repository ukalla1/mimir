import duckdb
import pandas as pd


def search_food(con, query, k=10):
    q = query.lower().replace("'", "''")
    return con.execute(f"""
        SELECT fdc_id, description
        FROM nodes_food
        WHERE lower(description) LIKE '%' || '{q}' || '%'
        LIMIT {k}
    """).df()


def get_food_nutrient_profile(con, fdc_id):
    df = con.execute(f"""
        SELECT n.nutrient_name, e.amount
        FROM edges_food_contains_nutrient e
        JOIN nodes_nutrient n USING(nutrient_id)
        WHERE e.fdc_id = {int(fdc_id)}
        ORDER BY e.amount DESC
    """).df()

    return df