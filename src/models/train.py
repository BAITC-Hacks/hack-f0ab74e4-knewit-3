"""Обучение: физический baseline + LightGBM, бленд по holdout, сохранение артефактов.

Split строго по времени (ADR: никакой утечки будущего):
train 2024-03..2025-11, holdout 2025-12..2026-01 — та же зима, что и тест.
"""
from __future__ import annotations

import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config import ARTIFACTS, HOLDOUT_START, TRAIN_END, TRAIN_START
from src.features.build import training_matrix
from src.features.dataset import load_hourly

LGB_PARAMS = dict(
    objective="regression_l1",  # MAE устойчивее к простоям, не вычищенным фильтром
    n_estimators=3000, learning_rate=0.03,
    num_leaves=63, min_child_samples=40,
    colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
    reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1,
)


def metrics(y: pd.Series, p: np.ndarray) -> dict:
    e = p - y.values
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2)))}


def train_turbine(turbine: int, weather: pd.DataFrame) -> dict:
    hourly = load_hourly(turbine)
    X, y = training_matrix(weather, hourly["target"])

    tr = (X.index >= TRAIN_START) & (X.index <= f"{TRAIN_END} 23:59")
    ho = X.index >= HOLDOUT_START
    Xtr, ytr, Xho, yho = X[tr], y[tr], X[ho], y[ho]

    # Baseline: изотоническая кривая мощности от ансамблевой скорости ветра
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(Xtr["ens_ws_mean"], ytr)
    base_ho = iso.predict(Xho["ens_ws_mean"])

    # Основная модель: LightGBM на всех прогнозных фичах, ранняя остановка по holdout
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(Xtr, ytr, eval_set=[(Xho, yho)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
    lgb_ho = np.clip(model.predict(Xho), 0, 1)

    # Бленд: вес по MAE на holdout (сетка 0..1)
    best_w, best_mae = 1.0, np.inf
    for w in np.linspace(0, 1, 21):
        mae = np.mean(np.abs(w * lgb_ho + (1 - w) * base_ho - yho.values))
        if mae < best_mae:
            best_w, best_mae = float(w), float(mae)
    blend_ho = np.clip(best_w * lgb_ho + (1 - best_w) * base_ho, 0, 1)

    report = {
        "turbine": turbine,
        "rows_train": int(len(Xtr)), "rows_holdout": int(len(Xho)),
        "baseline": metrics(yho, base_ho),
        "lightgbm": metrics(yho, lgb_ho),
        "blend": {**metrics(yho, blend_ho), "w_lgb": best_w},
        "by_lead": {
            int(l): metrics(yho[Xho["lead_day"] == l],
                            blend_ho[(Xho["lead_day"] == l).values])
            for l in (1, 2)
        },
        "best_iteration": int(model.best_iteration_ or 0),
    }

    # Дообучение на train+holdout с найденным числом деревьев — финальный артефакт
    final_params = {**LGB_PARAMS, "n_estimators": max(model.best_iteration_ or 500, 200)}
    final = lgb.LGBMRegressor(**final_params)
    final.fit(X, y)
    iso_full = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_full.fit(X["ens_ws_mean"], y)

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / f"turbine_{turbine}.pkl", "wb") as f:
        pickle.dump({"lgb": final, "iso": iso_full, "w_lgb": best_w,
                     "features": list(X.columns)}, f)
    return report


def train_all(weather: pd.DataFrame) -> dict:
    reports = {t: train_turbine(t, weather) for t in (1, 2)}
    (ARTIFACTS / "train_report.json").write_text(json.dumps(reports, indent=2))
    return reports
