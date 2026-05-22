#!/usr/bin/env python3
"""One-time script: split final_substitution.csv into train/test by source food.

Strategy: pick N_TEST source foods (default 184, matching paper), hold out ONE random
pair per test source food as the test edge. All other pairs go to training. This
mirrors PyG's RandomLinkSplit behaviour — each test source food has exactly one
held-out substitute, while its other substitution pairs remain in training.

Run once from nutri_graph/ before training:
    python scripts/split_subs.py

Outputs (written to nutri_graph/data/):
    subs_train.csv       — training pairs
    subs_test.csv        — N_TEST held-out test pairs (one per test source food)
    subs_test_foods.csv  — N_TEST source food URIs (eval query set)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR       = Path(__file__).resolve().parent
NUTRI_GRAPH_ROOT = SCRIPT_DIR.parent
MIMIR_ROOT       = NUTRI_GRAPH_ROOT.parent

SUBS_CSV = MIMIR_ROOT / "HealthyFoodSubs" / "Input Data" / "final_substitution.csv"
DATA_DIR = NUTRI_GRAPH_ROOT / "data"

SPLIT_SEED = 42
N_TEST     = 184   # number of test query foods (matches paper)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SPLIT_SEED)

    df = pd.read_csv(SUBS_CSV, sep=";")
    pair_counts = df["Food id"].value_counts()
    eligible_sources = pair_counts[pair_counts >= 2].index.tolist()

    if len(eligible_sources) < N_TEST:
        raise SystemExit(f"Only {len(eligible_sources)} sources have ≥2 pairs (need {N_TEST})")

    # sample N_TEST unique source foods
    test_sources = rng.choice(eligible_sources, size=N_TEST, replace=False)

    # for each test source, randomly hold out ONE pair as test
    test_indices = []
    for source in test_sources:
        candidate_indices = df.index[df["Food id"] == source].to_numpy()
        chosen = rng.choice(candidate_indices, size=1)[0]
        test_indices.append(chosen)

    test_df  = df.loc[test_indices]
    train_df = df.drop(test_indices)

    train_df.to_csv(DATA_DIR / "subs_train.csv", sep=";", index=False)
    test_df.to_csv( DATA_DIR / "subs_test.csv",  sep=";", index=False)

    test_foods = pd.DataFrame({"id": test_sources})
    test_foods.to_csv(DATA_DIR / "subs_test_foods.csv", index=False)

    print(f"Total pairs:        {len(df)}")
    print(f"Train:              {len(train_df)}  → data/subs_train.csv")
    print(f"Test:               {len(test_df)}    → data/subs_test.csv")
    print(f"Test query foods:   {len(test_foods)}    → data/subs_test_foods.csv")
