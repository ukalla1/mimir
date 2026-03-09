import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd

import plotly.express as px
import numpy as np


def quantile_range(x, q=0.995, pad=0.06):

    lo, hi = np.quantile(x, [1-q, q])
    p = pad * (hi - lo + 1e-9)

    return (float(lo - p), float(hi + p))


def plot_umap_clusters(coords, labels, meta_df, title, output):

    df = meta_df.copy()

    df["umap1"] = coords[:, 0]
    df["umap2"] = coords[:, 1]
    df["cluster_id"] = labels.astype(int)

    x_rng = quantile_range(df["umap1"].to_numpy())
    y_rng = quantile_range(df["umap2"].to_numpy())

    fig = px.scatter(
        df,
        x="umap1",
        y="umap2",
        color="cluster_id",
        hover_data=["fdc_id", "description", "protein_g", "fat_g", "carb_g", "kcal"],
        render_mode="webgl",
        template="simple_white",
        title=title
    )

    fig.update_traces(marker=dict(size=4, opacity=0.95))

    fig.update_layout(
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        xaxis=dict(range=list(x_rng)),
        yaxis=dict(range=list(y_rng)),
    )

    fig.write_html(output + ".html")

    try:
        fig.write_image(output + ".png", scale=3)
    except:
        pass

def plotly_cluster_plot(
    umap_proj,
    labels,
    food_names,
    save_path,
):

    df = pd.DataFrame(
        {
            "x": umap_proj[:, 0],
            "y": umap_proj[:, 1],
            "cluster": labels,
            "food": food_names,
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_name="food",
        title="Food Embedding Clusters",
    )

    fig.write_html(save_path)