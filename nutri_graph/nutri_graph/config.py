class Config:
    SEED = 82  # best seed from search (MAP=0.399)

    EMB_DIM = 64
    HIDDEN = 64
    HEADS = 4
    DROPOUT = 0.2

    LR = 2e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 90
    TRAIN_SPLIT = 0.85
    VAL_SPLIT = 0.92

    # Recipe graph integration (Phase 3 upgrade)
    INCLUDE_RECIPES = True

    # Substitution supervision (Method 2)
    LAMBDA_SUBS = 1.0
    SUBS_CSV = "../HealthyFoodSubs/Input Data/final_substitution.csv"