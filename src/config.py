"""Общая конфигурация для обучения и инференса."""
from pathlib import Path

RANDOM_STATE = 42
SEED = RANDOM_STATE

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "catboost_final.cbm"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.json"

SUBMISSION_PATH = Path("submission.csv")

CATBOOST_PARAMS = dict(
    iterations=700,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    auto_class_weights="Balanced",
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
)