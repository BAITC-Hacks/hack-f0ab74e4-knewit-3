"""Обучение: физический baseline + LightGBM, бленд, сохранение артефактов.

Ядро (`select_and_fit`) разделяет три роли данных и не видит ничего, кроме
переданных масок: подбор числа деревьев и веса бленда идёт только на fit/tune,
финальная посадка ансамбля — только на final_mask. Ретроспективная оценка
(src/backtest/evaluate.py) и канонический артефакт сдачи используют одно и то же
ядро с разными масками, поэтому «оценивается ровно та конфигурация, что сдаётся».

Канонический сценарий (train_all): fit 2024-03..2025-11, tune 2025-12..2026-01,
финальная посадка на всех целях по 31.01.2026. Метрики в train_report.json — это
метрики НА ДАННЫХ НАСТРОЙКИ (tune), а не нетронутый тест: декабрь–январь уже
участвовали в ранней остановке, бленде и отборе источников. Честная ретроспективная
оценка — python -m src.backtest.evaluate.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config import (ARTIFACTS, HOLDOUT_END, HOLDOUT_START, TRAIN_END,
                        TRAIN_START)
from src.features.build import training_matrix
from src.features.dataset import load_hourly

LGB_PARAMS = dict(
    objective="regression_l1",  # MAE устойчивее к простоям, не вычищенным фильтром
    n_estimators=3000, learning_rate=0.03,
    num_leaves=63, min_child_samples=40,
    colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
    reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1,
)

SEEDS = (1, 7, 13, 42, 99)
QUANTILE_ALPHAS = (0.1, 0.9)
EARLY_STOPPING_ROUNDS = 150


def metrics(y: pd.Series, p: np.ndarray) -> dict:
    e = p - y.values
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2)))}


def select_and_fit(X: pd.DataFrame, y: pd.Series,
                   fit_mask, tune_mask, final_mask,
                   params: dict | None = None,
                   seeds: tuple = SEEDS) -> tuple[dict, dict]:
    """Подбор конфигурации на fit/tune и посадка итогового ансамбля на final_mask.

    Хронологическая гарантия по построению: функция не получает никаких данных,
    кроме строк под тремя масками, и не заглядывает за final_mask. Оценочные цели
    в неё просто не попадают — их изменение не может сдвинуть ни деревья, ни вес
    бленда (проверяется тестом test_eval_targets_cannot_affect_fitted_model).
    """
    params = dict(params or LGB_PARAMS)
    Xf, yf = X[fit_mask], y[fit_mask]
    Xt, yt = X[tune_mask], y[tune_mask]

    # Baseline: изотоническая кривая мощности от ансамблевой скорости ветра
    iso_sel = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_sel.fit(Xf["ens_ws_mean"], yf)
    base_tune = iso_sel.predict(Xt["ens_ws_mean"])

    # Число деревьев: ранняя остановка по tune, посадка на fit
    probe = lgb.LGBMRegressor(**params)
    probe.fit(Xf, yf, eval_set=[(Xt, yt)],
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    lgb_tune = np.clip(probe.predict(Xt), 0, 1)

    # Вес бленда: сетка по MAE на tune
    best_w, best_mae = 1.0, np.inf
    for w in np.linspace(0, 1, 21):
        mae = float(np.mean(np.abs(w * lgb_tune + (1 - w) * base_tune - yt.values)))
        if mae < best_mae:
            best_w, best_mae = float(w), mae

    # Итоговый ансамбль: бэггинг по сидам с подобранным числом деревьев,
    # посадка строго на final_mask (для оценки это fit+tune, оценочных целей нет)
    n_est = max(probe.best_iteration_ or 500, 200)
    Xfin, yfin = X[final_mask], y[final_mask]
    finals = []
    for seed in seeds:
        m = lgb.LGBMRegressor(**{**params, "n_estimators": n_est, "random_state": seed})
        m.fit(Xfin, yfin)
        finals.append(m)
    iso_full = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_full.fit(Xfin["ens_ws_mean"], yfin)

    # Квантильные модели P10/P90 — диапазон неопределённости для оператора
    quantiles = {}
    for q in QUANTILE_ALPHAS:
        mq = lgb.LGBMRegressor(**{**params, "objective": "quantile", "alpha": q,
                                  "n_estimators": n_est})
        mq.fit(Xfin, yfin)
        quantiles[q] = mq

    artifact = {"lgbs": finals, "iso": iso_full, "w_lgb": best_w,
                "quantiles": quantiles, "features": list(X.columns)}
    selection = {
        "n_estimators": int(n_est),
        "best_iteration": int(probe.best_iteration_ or 0),
        "w_lgb": best_w,
        "rows_fit": int(fit_mask.sum()), "rows_tune": int(tune_mask.sum()),
        "rows_final": int(final_mask.sum()),
        "tune_pred_baseline": base_tune, "tune_pred_lgb": lgb_tune,
        "n_members": len(finals), "quantile_alphas": list(QUANTILE_ALPHAS),
    }
    return artifact, selection


def save_artifact(artifact: dict, turbine: int, artifacts_dir: Path | str) -> Path:
    """Сохранение под именем, которое читает predict.load_model."""
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"turbine_{turbine}.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    return path


def train_turbine(turbine: int, weather: pd.DataFrame,
                  artifacts_dir: Path | str | None = None) -> dict:
    """Канонический артефакт сдачи: подбор на 12.2025–01.2026, посадка на всём."""
    hourly = load_hourly(turbine)
    X, y = training_matrix(weather, hourly["target"])

    tr = (X.index >= TRAIN_START) & (X.index <= f"{TRAIN_END} 23:59")
    ho = X.index >= HOLDOUT_START

    artifact, sel = select_and_fit(X, y, tr, ho, tr | ho)
    save_artifact(artifact, turbine, artifacts_dir or ARTIFACTS)

    yho, Xho = y[ho], X[ho]
    base_ho, lgb_ho = sel["tune_pred_baseline"], sel["tune_pred_lgb"]
    w = sel["w_lgb"]
    blend_ho = np.clip(w * lgb_ho + (1 - w) * base_ho, 0, 1)

    # Ключи отчёта сохранены для совместимости с UI; protocol — явные метаданные
    # о роли периодов, чтобы старые поля не перечитывались как «честный тест».
    return {
        "turbine": turbine,
        "rows_train": int(tr.sum()), "rows_holdout": int(ho.sum()),
        "baseline": metrics(yho, base_ho),
        "lightgbm": metrics(yho, lgb_ho),
        "blend": {**metrics(yho, blend_ho), "w_lgb": w},
        "by_lead": {
            int(l): metrics(yho[Xho["lead_day"] == l],
                            blend_ho[(Xho["lead_day"] == l).values])
            for l in (1, 2)
        },
        "best_iteration": sel["best_iteration"],
        "protocol": {
            "role": "canonical-submission",
            "fit_period": [TRAIN_START, TRAIN_END],
            "tune_period": [HOLDOUT_START, HOLDOUT_END],
            "final_fit_through": HOLDOUT_END,
            "metrics_measured_on": "tune",
            "note": ("Метрики выше получены на данных настройки (ранняя остановка, "
                     "вес бленда) — это НЕ нетронутый тест. Ретроспективная оценка "
                     "итоговой конфигурации: python -m src.backtest.evaluate"),
        },
    }


def train_all(weather: pd.DataFrame,
              artifacts_dir: Path | str | None = None) -> dict:
    directory = Path(artifacts_dir or ARTIFACTS)
    reports = {t: train_turbine(t, weather, directory) for t in (1, 2)}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train_report.json").write_text(json.dumps(reports, indent=2))
    return reports
