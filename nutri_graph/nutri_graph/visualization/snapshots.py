import numpy as np
from pathlib import Path
from typing import Dict, Iterable, Optional


class SnapshotManager:
    """
    Colab-faithful snapshot capture:
      - Accepts VIS_IDX (deterministic) and stores it
      - Captures food embeddings at selected epochs
      - L2-normalizes food embeddings before storing
      - Stores (epoch -> [VIS_N, dim]) in memory AND saves to disk
    """

    def __init__(self, vis_idx: np.ndarray, out_dir: Optional[str] = None):
        self.vis_idx = np.asarray(vis_idx, dtype=np.int64)
        self.snapshots: Dict[int, np.ndarray] = {}
        self.out_dir = Path(out_dir) if out_dir is not None else None

        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.out_dir / "vis_idx.npy", self.vis_idx)

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + eps)

    def maybe_save(
        self,
        epoch: int,
        epochs_to_capture: Iterable[int],
        embeddings_np: np.ndarray,
        num_foods: int,
    ) -> None:
        if epoch not in set(epochs_to_capture):
            return

        food_emb = embeddings_np[:num_foods]
        food_emb = self._l2_normalize(food_emb)
        snap = food_emb[self.vis_idx]

        self.snapshots[int(epoch)] = snap

        if self.out_dir is not None:
            np.save(self.out_dir / f"food_emb_epoch_{int(epoch)}.npy", snap)

    def get_snapshots(self) -> Dict[int, np.ndarray]:
        return self.snapshots