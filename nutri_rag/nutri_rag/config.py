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

# ── Text Embeddings (V1 RAG) ─────────────────────────────────────────
TEXT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TEXT_EMBEDDING_DIM = 1024
TEXT_EMBEDDINGS_DIR = str(_PROJECT_ROOT / "data" / "embeddings")
TEXT_EMBEDDINGS_PATH = str(_PROJECT_ROOT / "data" / "embeddings" / "food_text_embeddings.npy")
TEXT_FDC_IDS_PATH = str(_PROJECT_ROOT / "data" / "embeddings" / "food_fdc_ids.npy")

# ── LLM ────────────────────────────────────────────────────────────────
LLM_BASE_URL = "http://localhost:8080/v1/chat/completions"
LLM_MODEL = "qwen3.5-9b"

# ── Retrieval ──────────────────────────────────────────────────────────
TOP_K_FOODS = 3          # max DB matches per parsed food item
GAT_NEIGHBORS_K = 5      # GAT embedding neighbors per seed candidate
SIMILARITY_THRESHOLD = 0.60  # cosine sim below this → "no reliable match"

# ── Nutrients of interest ──────────────────────────────────────────────
KEY_NUTRIENTS = [
    "Carbohydrate, by difference",
    "Protein",
    "Total lipid (fat)",
    "Energy",
    "Energy (Atwater General Factors)",
    "Energy (Atwater Specific Factors)",
]
