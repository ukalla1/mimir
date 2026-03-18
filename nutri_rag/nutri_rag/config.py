"""Paths, constants, and settings for nutri_rag."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent          # nutri_rag/
_MIMIR_ROOT = _PROJECT_ROOT.parent                              # mimir/
_NUTRI_GRAPH = _MIMIR_ROOT / "nutri_graph"

DB_PATH = str(_NUTRI_GRAPH / "data" / "nutri_kb.duckdb")
FOOD_EMBEDDINGS_PATH = str(_NUTRI_GRAPH / "outputs" / "embeddings" / "food_embeddings.npy")
NODE_EMBEDDINGS_PATH = str(_NUTRI_GRAPH / "outputs" / "embeddings" / "node_embeddings.pt")
USER_DB_PATH = str(_PROJECT_ROOT / "user_preferences.duckdb")

# ── LLM ────────────────────────────────────────────────────────────────
LLM_BASE_URL = "http://localhost:8080/v1/chat/completions"
LLM_MODEL = "qwen3.5-9b"

# ── Retrieval ──────────────────────────────────────────────────────────
TOP_K_FOODS = 3          # max DB matches per parsed food item
GAT_NEIGHBORS_K = 5      # GAT embedding neighbors per seed candidate

# ── Nutrients of interest ──────────────────────────────────────────────
KEY_NUTRIENTS = [
    "Carbohydrate, by difference",
    "Protein",
    "Total lipid (fat)",
    "Energy",
    "Energy (Atwater General Factors)",
    "Energy (Atwater Specific Factors)",
]
