"""
Обучение финальной модели на полном train и сохранение артефактов.

Запуск:
    python -m src.train_model

Артефакты:
    models/catboost_final.cbm   — обученная модель
    models/feature_cols.json    — список колонок
    models/train_meta.json      — метаданные (seed, params, размеры)
"""
from __future__ import annotations

import json
from pathlib import Path

from catboost import CatBoostClassifier

from src.config import (
    CATBOOST_PARAMS, SEED,
    MODELS_DIR, MODEL_PATH, FEATURE_COLS_PATH,
)
from src.data_loader import load_train, load_events
from src.features_final import build_features


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # Загрузка данных
    print("[1/3] Загрузка train и events...")
    train = load_train()
    events = load_events()

    # Получаем фичи
    print("[2/3] Сборка признаков...")
    feats = build_features(events, train, with_target=True)

    feature_cols = [c for c in feats.columns if c not in ("cookie_id", "target")]
    X = feats[feature_cols]
    y = feats["target"].astype(int)

    # Обучение
    print("[3/3] Обучение CatBoost...")
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X, y)

    # Артефакты
    model.save_model(str(MODEL_PATH))
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    meta = {
        "random_state": SEED,
        "catboost_params": CATBOOST_PARAMS,
        "n_features": len(feature_cols),
        "n_train_rows": int(len(X)),
    }
    with open(MODELS_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nСохранено:")
    print(f"  модель:  {MODEL_PATH}")
    print(f"  фичи:    {FEATURE_COLS_PATH} ({len(feature_cols)} шт.)")
    print(f"  мета:    {MODELS_DIR / 'train_meta.json'}")


if __name__ == "__main__":
    main()