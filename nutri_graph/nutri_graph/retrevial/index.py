from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch


class FoodRetriever:

    def __init__(self, emb_path):
        emb = torch.load(emb_path).numpy()
        self.index = NearestNeighbors(metric="cosine")
        self.index.fit(emb)
        self.emb = emb

    def nearest(self, idx, k=10):
        dists, neigh = self.index.kneighbors(self.emb[idx:idx+1], n_neighbors=k)
        return neigh[0]