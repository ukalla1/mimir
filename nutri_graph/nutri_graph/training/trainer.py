import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


class Trainer:
    """
    Colab-faithful trainer:
      - Train/Val/Test split on positive edges
      - Two-head decoding: existence + amount regression
      - Regression uses standardized log1p(amount)
      - Existence uses BCE with negative sampling
      - Scheduler: ReduceLROnPlateau(mode="min", factor=0.6, patience=4)
      - Best model tracked by val_rmse
      - Snapshot capture at selected epochs (food-only, L2-normalized, indexed by VIS_IDX)
      - Optional contrastive block can be added later (kept off by default)
    """

    def __init__(
        self,
        model,
        data,
        meta: Dict,
        config,
        snapshot_mgr=None,
        snapshot_epochs: Optional[List[int]] = None,
        use_contrastive: bool = False,
        lambda_contrast: float = 0.08,
        contrast_bs: int = 512,
        contrast_neg_k: int = 32,
        contrast_tau: float = 0.2,
    ):
        self.model = model
        self.data = data
        self.meta = meta
        self.config = config

        self.snapshot_mgr = snapshot_mgr
        self.snapshot_epochs = snapshot_epochs or []
        self.use_contrastive = use_contrastive
        self.lambda_contrast = lambda_contrast
        self.contrast_bs = contrast_bs
        self.contrast_neg_k = contrast_neg_k
        self.contrast_tau = contrast_tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.data = self.data.to(self.device)

        # Required meta keys (Colab-aligned dataset.py should provide these)
        self.num_foods = int(meta["NUM_FOODS"])
        self.num_nutrients = int(meta["NUM_NUTRIENTS"])
        self.food_to_nutrs = meta["food_to_nutrs"]

        # Colab uses full node_ids each step
        self.node_ids = torch.arange(self.data.num_nodes, device=self.device, dtype=torch.long)
        self.node_type_t = self.data.node_type.to(self.device)

        self.edge_index_all_t = self.data.edge_index
        self.edge_attr_all_t = self.data.edge_attr

        # Supervised positive edges + regression targets
        self.pos_edge_index = self.data.pos_edge_index.to(self.device)      # [2, num_pos]
        self.edge_attr_pos = self.data.edge_attr_pos.to(self.device)        # [num_pos, 1]
        self.y_all = self.edge_attr_pos.squeeze(-1)                         # [num_pos]

        # Optimizer + scheduler (match Colab)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.LR),
            weight_decay=float(config.WEIGHT_DECAY),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.6, patience=4
        )

        self.bce = nn.BCEWithLogitsLoss()

        # numpy RNG for negative sampling (match Colab seed)
        self.rng = np.random.default_rng(0)

        # Split indices (match Colab ratios)
        num_pos = self.pos_edge_index.size(1)
        perm = torch.randperm(num_pos, device=self.device)

        train_end = int(0.85 * num_pos)
        val_end = int(0.92 * num_pos)

        self.train_idx = perm[:train_end]
        self.val_idx = perm[train_end:val_end]
        self.test_idx = perm[val_end:]

        # Standardization stats from training set only (match Colab)
        y_train = self.y_all[self.train_idx]
        self.y_mean = y_train.mean()
        self.y_std = y_train.std().clamp_min(1e-6)

        # Best model tracking (match Colab)
        self.best_val_rmse = float("inf")
        self.best_state = None

    def standardize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def destandardize(self, y_stdspace: torch.Tensor) -> torch.Tensor:
        return y_stdspace * self.y_std + self.y_mean

    # -------- Colab-faithful negative sampling --------
    def sample_negative_bipartite(self, num_samples: int) -> torch.Tensor:
        """
        Return edge_index [2, num_samples] with global indexing:
          food: [0..NUM_FOODS-1]
          nutrient: [NUM_FOODS..NUM_FOODS+NUM_NUTRIENTS-1]
        Matches your Colab function (batch oversampling + membership check).
        """
        neg_src = np.empty(num_samples, dtype=np.int64)
        neg_dst = np.empty(num_samples, dtype=np.int64)

        filled = 0
        while filled < num_samples:
            batch = (num_samples - filled) * 3
            cand_food = self.rng.integers(0, self.num_foods, size=batch, dtype=np.int64)
            cand_nutr = self.rng.integers(0, self.num_nutrients, size=batch, dtype=np.int64)

            for f, n in zip(cand_food, cand_nutr):
                if n not in self.food_to_nutrs[f]:
                    neg_src[filled] = f
                    neg_dst[filled] = n + self.num_foods
                    filled += 1
                    if filled >= num_samples:
                        break

        return torch.tensor(np.vstack([neg_src, neg_dst]), dtype=torch.long, device=self.device)

    # -------- Evaluation (match Colab) --------
    def eval_split(self, split_idx: torch.Tensor) -> Tuple[float, float, float]:
        self.model.eval()
        with torch.no_grad():
            h = self.model.encode(self.node_ids, self.node_type_t, self.edge_index_all_t, self.edge_attr_all_t)

            pos_ei = self.pos_edge_index[:, split_idx]
            pos_y = self.y_all[split_idx]
            pos_yz = self.standardize(pos_y)

            pos_exist_logits = self.model.decode_exist(h, pos_ei)
            pos_amt_pred_z = self.model.decode_amount(h, pos_ei)

            neg_ei = self.sample_negative_bipartite(pos_ei.size(1))
            neg_exist_logits = self.model.decode_exist(h, neg_ei)

            # Regression metrics in log1p space (destandardize predictions)
            pos_amt_pred = self.destandardize(pos_amt_pred_z)
            y_true_reg = pos_y.detach().cpu().numpy()
            y_pred_reg = pos_amt_pred.detach().cpu().numpy()

            mae = mean_absolute_error(y_true_reg, y_pred_reg)
            rmse = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))

            # AUC for existence
            y_true = torch.cat([
                torch.ones_like(pos_exist_logits),
                torch.zeros_like(neg_exist_logits),
            ]).detach().cpu().numpy()

            y_score = torch.cat([pos_exist_logits, neg_exist_logits]).detach().cpu().numpy()
            auc = roc_auc_score(y_true, y_score)

            return mae, rmse, auc

    def train(self) -> Dict:
        history = {"train_loss": [], "val_mae": [], "val_rmse": [], "val_auc": [], "lr": []}

        max_epochs = int(self.config.MAX_EPOCHS)

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            h = self.model.encode(self.node_ids, self.node_type_t, self.edge_index_all_t, self.edge_attr_all_t)

            pos_ei = self.pos_edge_index[:, self.train_idx]
            pos_y = self.y_all[self.train_idx]
            pos_yz = self.standardize(pos_y)

            pos_exist_logits = self.model.decode_exist(h, pos_ei)
            pos_amt_pred_z = self.model.decode_amount(h, pos_ei)

            neg_ei = self.sample_negative_bipartite(pos_ei.size(1))
            neg_exist_logits = self.model.decode_exist(h, neg_ei)

            loss_exist = self.bce(pos_exist_logits, torch.ones_like(pos_exist_logits)) + \
                         self.bce(neg_exist_logits, torch.zeros_like(neg_exist_logits))

            loss_amt = F.smooth_l1_loss(pos_amt_pred_z, pos_yz)

            # Colab default: loss = loss_amt + 0.4*loss_exist
            loss = loss_amt + 0.4 * loss_exist

            # (Optional) contrastive hook — left disabled by default (matches your notebook)
            # if self.use_contrastive:
            #     loss = loss + self.lambda_contrast * loss_contrast

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()

            # Validation (match Colab)
            val_mae, val_rmse, val_auc = self.eval_split(self.val_idx)
            self.scheduler.step(val_rmse)

            lr = float(self.optimizer.param_groups[0]["lr"])
            history["train_loss"].append(float(loss.item()))
            history["val_mae"].append(float(val_mae))
            history["val_rmse"].append(float(val_rmse))
            history["val_auc"].append(float(val_auc))
            history["lr"].append(lr)

            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} "
                f"| val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f} | val_AUC={val_auc:.4f} | lr={lr:.2e}"
            )

            # ---- Snapshot saving (match Colab) ----
            if self.snapshot_mgr is not None and epoch in set(self.snapshot_epochs):
                self.model.eval()
                with torch.no_grad():
                    h_ep = self.model.encode(
                        self.node_ids, self.node_type_t, self.edge_index_all_t, self.edge_attr_all_t
                    ).detach().cpu().numpy()
                self.snapshot_mgr.maybe_save(epoch, self.snapshot_epochs, h_ep, self.num_foods)
                self.model.train()
                print(f"[snapshot] saved embeddings for epoch {epoch}")

            # ---- Best checkpoint by val_rmse (match Colab) ----
            if val_rmse + 1e-5 < self.best_val_rmse:
                self.best_val_rmse = float(val_rmse)
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        # Load best model (match Colab)
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        test_mae, test_rmse, test_auc = self.eval_split(self.test_idx)
        test_metrics = {"MAE_log1p": float(test_mae), "RMSE_log1p": float(test_rmse), "AUC": float(test_auc)}
        print("\nTEST:", test_metrics)

        return {"history": history, "test": test_metrics, "best_val_rmse": float(self.best_val_rmse)}