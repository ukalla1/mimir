import pandas as pd


def compute_macro_labels(con, vis_fdcs):

    vis_fdcs_sql = ",".join(map(str, vis_fdcs))

    def fetch_exact(name, unit=None):

        cond = f"lower(trim(n.nutrient_name)) = lower(trim('{name}'))"

        if unit is not None:
            cond += f" AND upper(trim(n.unit_name)) = upper(trim('{unit}'))"

        q = f"""
        SELECT e.fdc_id, SUM(e.amount) AS amount
        FROM edges_food_contains_nutrient e
        JOIN nodes_nutrient n ON e.nutrient_id = n.nutrient_id
        WHERE e.fdc_id IN ({vis_fdcs_sql})
          AND {cond}
          AND e.amount IS NOT NULL
        GROUP BY e.fdc_id
        """

        return con.execute(q).df()

    df_prot = fetch_exact("Protein")
    df_fat = fetch_exact("Total lipid (fat)")
    df_carb = fetch_exact("Carbohydrate, by difference")
    df_kcal = fetch_exact("Energy", "KCAL")

    vis_tbl = pd.DataFrame({"fdc_id": vis_fdcs})

    vis_tbl = vis_tbl.merge(df_prot.rename(columns={"amount": "protein_g"}), on="fdc_id", how="left")
    vis_tbl = vis_tbl.merge(df_fat.rename(columns={"amount": "fat_g"}), on="fdc_id", how="left")
    vis_tbl = vis_tbl.merge(df_carb.rename(columns={"amount": "carb_g"}), on="fdc_id", how="left")
    vis_tbl = vis_tbl.merge(df_kcal.rename(columns={"amount": "kcal"}), on="fdc_id", how="left")

    vis_tbl.fillna(0.0, inplace=True)

    def macro_label(r):

        vals = {
            "Protein": r["protein_g"],
            "Fat": r["fat_g"],
            "Carb": r["carb_g"],
        }

        return max(vals, key=vals.get)

    vis_tbl["macro_dom"] = vis_tbl.apply(macro_label, axis=1)

    return vis_tbl