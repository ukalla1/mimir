import numpy as np
import torch


class BipartiteNegativeSampler:

    def __init__(self, num_foods, num_nutrients, food_to_nutrs):
        self.num_foods = num_foods
        self.num_nutrients = num_nutrients
        self.food_to_nutrs = food_to_nutrs
        self.rng = np.random.default_rng(0)

    def sample(self, num_samples):

        neg_src = np.empty(num_samples, dtype=np.int64)
        neg_dst = np.empty(num_samples, dtype=np.int64)

        filled = 0

        while filled < num_samples:

            f = self.rng.integers(0, self.num_foods)
            n = self.rng.integers(0, self.num_nutrients)

            if n not in self.food_to_nutrs[f]:

                neg_src[filled] = f
                neg_dst[filled] = n + self.num_foods

                filled += 1

        return torch.tensor(np.vstack([neg_src, neg_dst]), dtype=torch.long)