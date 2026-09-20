"""
Инференс финальной модели: собирает фичи для test,
пишет submission.csv в требуемом формате.

Запуск:
    python -m src.predict

Требования:
    - модель обучена: models/catboost_final.cbm
    - сохранён список фичей: models/feature_cols.json

Артефакт:
    submission.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import (
    MODEL_PATH, FEATURE_COLS_PATH, SUBMISSION_PATH,
)
from src.data_loader import load_test, load_events
from src.features_final import build_features


def _validate_submission(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    """Жёсткие проверки формата перед записью."""
    assert len(sub) == len(test), \
        f"длина submission ({len(sub)}) != длины test ({len(test)})"
    assert sub["cookie_id"].is_unique, "дубликаты cookie_id в submission"
    assert sub["score"].notna().all(), "NaN в score"
    assert sub["score"].between(0, 1).all(), "score вне диапазона [0, 1]"
    assert set(sub["cookie_id"]) == set(test["cookie_id"]), \
        "cookie_id в submission не совпадают с test"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}. "
            f"Сначала запусти `python -m src.train_final`."
        )
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(
            f"Список фичей не найден: {FEATURE_COLS_PATH}. "
            f"Сначала запусти `python -m src.train_final`."
        )

    # Загрузка модели и списка фичей
    print("[1/4] Загрузка модели и списка фичей...")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    # Данные и фичи
    print("[2/4] Загрузка test и events...")
    test = load_test()
    events = load_events()

    print("[3/4] Сборка признаков...")
    feats = build_features(events, test, with_target=False)

    missing = set(feature_cols) - set(feats.columns)
    if missing:
        raise ValueError(f"В test отсутствуют признаки: {sorted(missing)}")

    X_test = feats[feature_cols]

    # Предсказание и submission
    print("[4/4] Предсказание и запись submission...")
    scores = model.predict_proba(X_test)[:, 1]

    sub = pd.DataFrame({
        "cookie_id": feats["cookie_id"].values,
        "score": scores,
    })

    _validate_submission(sub, test)
    sub.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nГотово: {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()