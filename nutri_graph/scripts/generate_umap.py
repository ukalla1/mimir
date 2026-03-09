import torch
import umap
import plotly.express as px
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":
    emb = torch.load("outputs/embeddings/node_embeddings.pt")
    emb = emb.numpy()

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.12,
        metric="cosine",
        random_state=0
    )

    coords = reducer.fit_transform(emb)

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        title="Food Embedding UMAP"
    )

    out_dir = Path("outputs/umap")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(out_dir / "umap.html")
    fig.write_image(out_dir / "umap.png", scale=3)

    print("UMAP saved.")

    kmeans = MiniBatchKMeans(n_clusters=12, random_state=0)
    clusters = kmeans.fit_predict(emb)

    torch.save(clusters, "outputs/embeddings/cluster_labels.pt")