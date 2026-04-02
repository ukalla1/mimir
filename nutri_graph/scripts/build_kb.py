from pathlib import Path
from nutri_graph.kb.builder import build_kb

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent

    sr_legacy_path = base / "data" / "SR-Leg_ASC"

    build_kb(
        dataset_path=str(base / "data" / "raw"),
        db_path=str(base / "data" / "nutri_kb.duckdb"),
        sr_legacy_path=str(sr_legacy_path) if sr_legacy_path.exists() else None,
    )