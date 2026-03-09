import numpy as np
from sklearn.cluster import MiniBatchKMeans


class ClusterModel:

    def __init__(self, n_clusters=12):
        self.n_clusters = n_clusters
        self.model = None

    def fit(self, embeddings):

        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=0,
            batch_size=2048,
            # n_init="auto",
            n_init=20,
        ).fit(embeddings)

    def predict(self, embeddings):

        if self.model is None:
            raise RuntimeError("ClusterModel must be fitted first")

        return self.model.predict(embeddings)