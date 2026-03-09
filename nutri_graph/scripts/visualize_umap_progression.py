import re
from pathlib import Path

import duckdb
import numpy as np
import plotly.express as px
from sklearn.cluster import MiniBatchKMeans

from nutri_graph.visualization.umap import UMAPProjector


# -----------------------------
# Settings (match Colab)
# -----------------------------
SNAPSHOT_DIR = Path("outputs/snapshots")
OUT_DIR = Path("outputs/umap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 12  # same as Colab


# -----------------------------
# Helpers copied from your Colab logic
# -----------------------------
def quantile_range(x, q=0.995, pad=0.06):
    lo, hi = np.quantile(x, [1 - q, q])
    p = pad * (hi - lo + 1e-9)
    return (float(lo - p), float(hi + p))


VIVID_CLUSTER_PALETTE = (
    px.colors.qualitative.Bold
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Vivid
)


def make_discrete_palette(num_classes):
    pal = VIVID_CLUSTER_PALETTE
    if num_classes <= len(pal):
        return pal[:num_classes]
    reps = int(np.ceil(num_classes / len(pal)))
    return (pal * reps)[:num_classes]


def apply_umap_paper_style(fig, title, x_rng, y_rng, legend_title):
    # Matches your Colab "paper-style UMAP" block
    PLOT_WIDTH = 780
    PLOT_HEIGHT = 520

    FONT_FAMILY = "Arial"
    BASE_FONT = 20
    TITLE_FONT = 22
    AXIS_FONT = 20
    LEGEND_FONT = 18

    MARKER_SIZE = 4.2
    MARKER_OPAC = 0.95
    MARKER_LINE_W = 0.3
    MARKER_LINE_C = "rgba(0,0,0,0.35)"

    GRID_COLOR = "rgba(0,0,0,0.18)"
    AXIS_COLOR = "black"

    fig.update_traces(
        marker=dict(
            size=MARKER_SIZE,
            opacity=MARKER_OPAC,
            line=dict(width=MARKER_LINE_W, color=MARKER_LINE_C),
        )
    )

    fig.update_layout(
        template="plotly_white",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(family=FONT_FAMILY, size=TITLE_FONT, color="black"),
        ),
        font=dict(family=FONT_FAMILY, size=BASE_FONT, color="black"),
        legend=dict(
            title=dict(text=f"<b>{legend_title}</b>"),
            font=dict(family=FONT_FAMILY, size=LEGEND_FONT, color="black"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=2,
            itemsizing="constant",
        ),
        margin=dict(l=75, r=25, t=75, b=70),
    )

    fig.update_xaxes(
        title=dict(text="<b>UMAP-1</b>", font=dict(size=AXIS_FONT, color="black")),
        range=list(x_rng),
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        showline=True,
        linewidth=3,
        linecolor=AXIS_COLOR,
        mirror=True,
        ticks="outside",
        tickwidth=3,
        tickcolor=AXIS_COLOR,
        tickfont=dict(size=AXIS_FONT, color="black"),
        zeroline=False,
    )
    fig.update_yaxes(
        title=dict(text="<b>UMAP-2</b>", font=dict(size=AXIS_FONT, color="black")),
        range=list(y_rng),
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        showline=True,
        linewidth=3,
        linecolor=AXIS_COLOR,
        mirror=True,
        ticks="outside",
        tickwidth=3,
        tickcolor=AXIS_COLOR,
        tickfont=dict(size=AXIS_FONT, color="black"),
        zeroline=False,
    )
    return fig


def save_plotly(fig, base_path_no_ext):
    html_path = str(base_path_no_ext) + ".html"
    png_path = str(base_path_no_ext) + ".png"
    fig.write_html(html_path)
    try:
        fig.write_image(png_path, scale=3)  # requires kaleido
    except Exception as e:
        print(f"[warn] PNG export failed: {e}")
    print("Saved:", html_path, "and", png_path)


# -----------------------------
# Load snapshots (match our training pipeline naming)
# -----------------------------
if not (SNAPSHOT_DIR / "vis_idx.npy").exists():
    raise FileNotFoundError(
        "Missing outputs/snapshots/vis_idx.npy. "
        "Re-run training; SnapshotManager should save it."
    )

vis_idx = np.load(SNAPSHOT_DIR / "vis_idx.npy")

snap_files = sorted(SNAPSHOT_DIR.glob("food_emb_epoch_*.npy"))
if len(snap_files) == 0:
    raise FileNotFoundError(
        "No snapshot files found. Expected outputs/snapshots/food_emb_epoch_*.npy"
    )

# Parse epochs
epochs = []
snapshots = {}
for f in snap_files:
    m = re.search(r"food_emb_epoch_(\d+)\.npy$", f.name)
    if m:
        ep = int(m.group(1))
        snapshots[ep] = np.load(f)
        epochs.append(ep)

epochs = sorted(epochs)
REF_EPOCH = max(epochs)
X_ref = snapshots[REF_EPOCH]
print("UMAP REF_EPOCH =", REF_EPOCH, "| snapshots =", epochs)


# -----------------------------
# Fit UMAP ONCE on reference epoch (Colab)
# -----------------------------
umap_proj = UMAPProjector()
umap_proj.fit(X_ref)

# -----------------------------
# Fit MiniBatchKMeans ONCE on reference epoch (Colab)
# -----------------------------
kmeans = MiniBatchKMeans(
    n_clusters=K,
    random_state=0,
    batch_size=2048,
    n_init="auto",
).fit(X_ref)


# -----------------------------
# Build vis_tbl (macro + kcal + description), same as Colab
# -----------------------------
con = duckdb.connect("data/nutri_kb.duckdb", read_only=True)

foods = con.execute("SELECT fdc_id, description, food_category_id FROM nodes_food").df()
foods["fdc_id"] = foods["fdc_id"].astype(int)
foods["food_category_id"] = foods["food_category_id"].fillna(-1).astype(int)

VIS_FDCS = foods["fdc_id"].to_numpy()[vis_idx].astype(int)
vis_fdcs_sql = ",".join(map(str, VIS_FDCS.tolist()))

def fetch_exact(name, unit=None, out_col="amount"):
    where = [f"lower(trim(n.nutrient_name)) = lower(trim('{name}'))"]
    if unit is not None:
        where.append(f"upper(trim(n.unit_name)) = upper(trim('{unit}'))")
    where_sql = " AND ".join(where)

    q = f"""
    SELECT e.fdc_id, SUM(e.amount) AS {out_col}
    FROM edges_food_contains_nutrient e
    JOIN nodes_nutrient n ON e.nutrient_id = n.nutrient_id
    WHERE e.fdc_id IN ({vis_fdcs_sql})
      AND {where_sql}
      AND e.amount IS NOT NULL
    GROUP BY e.fdc_id
    """
    return con.execute(q).df()

df_prot = fetch_exact("Protein", out_col="protein_g")
df_fat  = fetch_exact("Total lipid (fat)", out_col="fat_g")
df_carb = fetch_exact("Carbohydrate, by difference", out_col="carb_g")
df_kcal = fetch_exact("Energy", unit="KCAL", out_col="kcal")

import pandas as pd
vis_tbl = pd.DataFrame({"fdc_id": VIS_FDCS})
vis_tbl = vis_tbl.merge(df_prot, on="fdc_id", how="left")
vis_tbl = vis_tbl.merge(df_fat,  on="fdc_id", how="left")
vis_tbl = vis_tbl.merge(df_carb, on="fdc_id", how="left")
vis_tbl = vis_tbl.merge(df_kcal, on="fdc_id", how="left")

for c in ["protein_g", "fat_g", "carb_g", "kcal"]:
    vis_tbl[c] = vis_tbl[c].fillna(0.0)

def macro_label_row(r):
    vals = {"Protein": r["protein_g"], "Fat": r["fat_g"], "Carb": r["carb_g"]}
    m = max(vals, key=vals.get)
    if vals[m] <= 1e-6:
        return "Other"
    return m

vis_tbl["macro_dom"] = vis_tbl.apply(macro_label_row, axis=1)

bins = [-1, 50, 150, 300, 600, 2000]
labels = ["<=50", "50-150", "150-300", "300-600", ">=600"]
vis_tbl["kcal_bin"] = pd.cut(vis_tbl["kcal"], bins=bins, labels=labels)

vis_tbl["description"] = foods.loc[vis_idx, "description"].to_numpy()

con.close()


# -----------------------------
# Plot per epoch (cluster colors like Colab)
# -----------------------------
for ep in epochs:
    X = snapshots[ep]
    coords = umap_proj.transform(X)

    cluster_labels = kmeans.predict(X).astype(int)
    vis_tbl["cluster_id"] = cluster_labels
    vis_tbl["cluster_id_str"] = vis_tbl["cluster_id"].astype(str)

    x_rng = quantile_range(coords[:, 0])
    y_rng = quantile_range(coords[:, 1])

    palette = make_discrete_palette(int(vis_tbl["cluster_id"].nunique()))

    fig = px.scatter(
        vis_tbl,
        x=coords[:, 0],
        y=coords[:, 1],
        color=vis_tbl["cluster_id_str"],
        hover_data=["fdc_id", "description", "protein_g", "fat_g", "carb_g", "kcal"],
        render_mode="webgl",
        template="plotly_white",
        color_discrete_sequence=palette,
        title=f"Food Embeddings UMAP @ Epoch {ep} (Learned Clusters, K={K})",
    )

    fig = apply_umap_paper_style(
        fig,
        title=f"Food Embeddings UMAP @ Epoch {ep} (Learned Clusters, K={K})",
        x_rng=x_rng,
        y_rng=y_rng,
        legend_title="Cluster ID",
    )

    save_plotly(fig, OUT_DIR / f"umap_clusters_vivid_epoch_{ep}")