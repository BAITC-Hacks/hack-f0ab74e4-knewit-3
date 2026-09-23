"""Проверка двух гипотез из рецензии (запуск: python -m src.backtest.prof_ideas).

Гипотеза A — двухступенчатая схема с калибровкой:
    NWP -> предсказанный ветер на площадке -> мощность,
вместо прямого NWP -> мощность. Выигрыш не в самой композиции (её бустинг выучивает
и напрямую), а в объёме данных: кривую мощности можно учить на ВСЕЙ истории с марта
2023, тогда как прямая схема ограничена архивом прогнозов с марта 2024.

Гипотеза B — временная персистентность: добавить признаки недавней фактической
выработки, доступной на момент выпуска прогноза (до конца дня D включительно).
Для цели в дне D+L берётся агрегат по 48 часам, закончившимся в D 23:00.
Применение к подаче требует фактической мощности на эти даты, которой в феврале нет.

Обе сравниваются на периоде настройки 12.2025–01.2026; это ретроспективный эксперимент.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import HOLDOUT_END, HOLDOUT_START, TEST_END, TRAIN_END, TRAIN_START
from src.features.build import issued_feature_stack, training_matrix
from src.features.dataset import load_hourly
from src.models.train import LGB_PARAMS
from src.weather.openmeteo import get_weather


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def recent_power_features(index: pd.DatetimeIndex, lead: pd.Series,
                          power: pd.Series) -> pd.DataFrame:
    """Агрегаты фактической выработки, известные на момент выпуска прогноза.

    Для цели в момент T с горизонтом L суток отсечка — конец дня (T.date() - L).
    Берём 48 часов до отсечки: так признак одинаков для всех часов одного запуска
    и физически доступен оператору.
    """
    daily_end = power.copy()
    days = pd.Series(index.normalize(), index=index) - pd.to_timedelta(lead.values, unit="D")
    out = pd.DataFrame(index=index)
    # предвычисляем агрегаты по каждой дате отсечки
    uniq = pd.DatetimeIndex(days.unique())
    stats = {}
    for d in uniq:
        window = daily_end.loc[d - pd.Timedelta(days=1): d + pd.Timedelta(hours=23)]
        stats[d] = (window.mean(), window.tail(1).mean() if len(window) else np.nan,
                    window.std())
    out["recent_mean48"] = [stats[d][0] for d in days]
    out["recent_last"] = [stats[d][1] for d in days]
    out["recent_std48"] = [stats[d][2] for d in days]
    return out


def run() -> None:
    weather = get_weather(TRAIN_START, TEST_END)
    stack = issued_feature_stack(weather)
    print(f"погода: {weather.shape}\n")

    for turbine in (1, 2):
        hourly = load_hourly(turbine)
        X, y = training_matrix(weather, hourly["target"], stack=stack,
                               target_end=f"{HOLDOUT_END} 23:00")
        tr = (X.index >= TRAIN_START) & (X.index <= f"{TRAIN_END} 23:59")
        ho = X.index >= HOLDOUT_START
        Xtr, ytr, Xho, yho = X[tr], y[tr], X[ho], y[ho]
        print(f"=== Турбина {turbine}: train {len(Xtr)}, holdout {len(Xho)} ===")

        # --- эталон: прямая схема NWP -> мощность (наша текущая модель) ---
        direct = lgb.LGBMRegressor(**LGB_PARAMS)
        direct.fit(Xtr, ytr, eval_set=[(Xho, yho)],
                   callbacks=[lgb.early_stopping(150, verbose=False)])
        p_direct = np.clip(direct.predict(Xho), 0, 1)
        print(f"  эталон  прямая NWP -> мощность          MAE={mae(yho, p_direct):.4f}")

        # --- гипотеза A: калибровка NWP -> ветер, затем ветер -> мощность ---
        wind_meas = hourly["wind_meas"].reindex(X.index)
        ok_tr = tr & wind_meas.notna().values
        calib = lgb.LGBMRegressor(**{**LGB_PARAMS, "objective": "regression"})
        calib.fit(X[ok_tr], wind_meas[ok_tr])
        wind_ho = calib.predict(Xho)
        print(f"          калибровка ветра: MAE={mae(wind_meas[ho], wind_ho):.3f} м/с, "
              f"сырой NWP: {mae(wind_meas[ho], Xho['ens_ws_mean']):.3f} м/с")

        # ступень 2 учится на ВСЕЙ истории (с марта 2023), а не только с архива прогнозов
        full = hourly.dropna(subset=["target"])
        full_tr = full[full.index <= f"{TRAIN_END} 23:59"]
        curve = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": 800})
        curve.fit(full_tr[["wind_meas", "temp_meas"]], full_tr["target"])

        stage2_in = pd.DataFrame({"wind_meas": wind_ho,
                                  "temp_meas": Xho[[c for c in Xho.columns
                                                    if c.endswith("_temp")]].mean(axis=1).values},
                                 index=Xho.index)
        p_two = np.clip(curve.predict(stage2_in), 0, 1)
        print(f"  A       двухступенчатая (калибровка)     MAE={mae(yho, p_two):.4f}  "
              f"[ступень 2 обучена на {len(full_tr)} часах с 03.2023]")

        # бленд прямой и двухступенчатой
        best = min(((mae(yho, w * p_direct + (1 - w) * p_two), w)
                    for w in np.linspace(0, 1, 21)))
        print(f"  A+      бленд прямой и двухступенчатой   MAE={best[0]:.4f} (вес прямой {best[1]:.2f})")

        # --- гипотеза B: признаки недавней фактической выработки ---
        rec = recent_power_features(X.index, X["lead_day"], hourly["power"])
        Xb = pd.concat([X, rec], axis=1)
        mb = lgb.LGBMRegressor(**LGB_PARAMS)
        mb.fit(Xb[tr], ytr, eval_set=[(Xb[ho], yho)],
               callbacks=[lgb.early_stopping(150, verbose=False)])
        p_b = np.clip(mb.predict(Xb[ho]), 0, 1)
        print(f"  B       + недавняя выработка (3 признака) MAE={mae(yho, p_b):.4f}\n")


if __name__ == "__main__":
    run()
