"""Export the cleaned USDA KB to JSON for sharing / fine-tuning."""

from pathlib import Path
from nutri_graph.kb.builder import export_kb

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    export_kb(
        db_path=str(base / "data" / "nutri_kb.duckdb"),
        output_path=str(base / "data" / "nutri_kb_export.json"),
    )
