from pathlib import Path
import kagglehub
import shutil

DATASET = "barkataliarbab/usda-fooddata-central-foundation-foods-2025"

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")
    path = kagglehub.dataset_download(DATASET)

    print("Copying dataset to project...")
    for p in Path(path).glob("**/*"):
        if p.is_file():
            target = raw_dir / p.name
            shutil.copy(p, target)

    print("Dataset ready at:", raw_dir)