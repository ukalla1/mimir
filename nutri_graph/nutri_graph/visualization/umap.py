import umap


class UMAPProjector:

    def __init__(self):

        self.model = None

    def fit(self, embeddings):

        self.model = umap.UMAP(
            n_neighbors=30,
            min_dist=0.12,
            metric="cosine",
            random_state=0
        ).fit(embeddings)

    def transform(self, embeddings):

        if self.model is None:
            raise RuntimeError("UMAP must be fitted first")

        return self.model.transform(embeddings)