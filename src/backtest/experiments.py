"""Быстрый перебор улучшений на holdout (запуск: python -m src.backtest.experiments).

Каждый вариант обучается на train (03.2024–11.2025) и меряется на holdout
(12.2025–01.2026). Никакой подбор не касается тестового февраля.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import HOLDOUT_START, TRAIN_END, TRAIN_START, TEST_END
from src.features.build import training_matrix
from src.features.dataset import load_hourly
from src.models.train import LGB_PARAMS
from src.weather.openmeteo import get_spatial_weather, get_weather


def mae(y, p): return float(np.mean(np.abs(p - y)))


def run():
    weather = get_weather(TRAIN_START, TEST_END)
    spatial = get_spatial_weather(TRAIN_START, TEST_END)
    both = pd.concat([weather, spatial], axis=1)
    print(f"weather: {weather.shape}, spatial: {spatial.shape}")

    for turbine in (1, 2):
        hourly = load_hourly(turbine)
        for name, wdf in [("A: 5 сильных моделей", weather),
                          ("B: A + пространственные градиенты", both)]:
            X, y = training_matrix(wdf, hourly["target"])
            tr = (X.index >= TRAIN_START) & (X.index <= f"{TRAIN_END} 23:59")
            ho = X.index >= HOLDOUT_START
            Xtr, ytr, Xho, yho = X[tr], y[tr], X[ho], y[ho]

            m = lgb.LGBMRegressor(**LGB_PARAMS)
            m.fit(Xtr, ytr, eval_set=[(Xho, yho)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            p = np.clip(m.predict(Xho), 0, 1)
            print(f"T{turbine} {name:38s} {X.shape[1]:4d} фич  "
                  f"MAE={mae(yho.values, p):.4f} (iter {m.best_iteration_})")

            if name.startswith("B"):
                # бэггинг по сидам: среднее 5 моделей
                preds = []
                for seed in (1, 7, 13, 42, 99):
                    ms = lgb.LGBMRegressor(**{**LGB_PARAMS, "random_state": seed,
                                              "n_estimators": m.best_iteration_ or 300})
                    ms.fit(Xtr, ytr)
                    preds.append(np.clip(ms.predict(Xho), 0, 1))
                pbag = np.mean(preds, axis=0)
                print(f"T{turbine} B + бэггинг 5 сидов{'':21s}      "
                      f"MAE={mae(yho.values, pbag):.4f}")


if __name__ == "__main__":
    run()
