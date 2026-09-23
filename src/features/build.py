"""Матрица фич из архивных прогнозов погоды.

Вход инференса — только прогноз (previous-runs API); измеренные на турбине величины
в фичи не входят (на инференсе их нет). Обучение на тех же фичах устраняет сдвиг
распределений train/inference и учит модель поправлять ошибки погодной модели.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import WEATHER_MODELS

R_AIR = 287.05  # Дж/(кг·К)


def _model_block(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Фичи одного погодного источника."""
    p = f"{model}__"
    out = pd.DataFrame(index=df.index)
    for h in (10, 80, 100, 120):
        out[f"{model}_ws{h}"] = df[f"{p}wind_speed_{h}m"]
    ws = df[f"{p}wind_speed_100m"]
    out[f"{model}_ws100_cube"] = ws ** 3          # физика: P ~ rho * v^3
    out[f"{model}_gust"] = df[f"{p}wind_gusts_10m"]
    out[f"{model}_shear"] = (df[f"{p}wind_speed_120m"] - df[f"{p}wind_speed_10m"]).clip(lower=-20)
    wd = np.deg2rad(df[f"{p}wind_direction_100m"])
    out[f"{model}_wd_sin"] = np.sin(wd)
    out[f"{model}_wd_cos"] = np.cos(wd)
    t_k = df[f"{p}temperature_2m"] + 273.15
    rho = (df[f"{p}surface_pressure"] * 100) / (R_AIR * t_k)
    out[f"{model}_rho"] = rho
    out[f"{model}_pwr_density"] = 0.5 * rho * ws ** 3  # Вт/м^2
    out[f"{model}_temp"] = df[f"{p}temperature_2m"]
    # инерция и фазовые ошибки фронтов: лаги и окно по основной скорости
    out[f"{model}_ws100_lag1"] = ws.shift(1)
    out[f"{model}_ws100_lead1"] = ws.shift(-1)
    out[f"{model}_ws100_roll3"] = ws.rolling(3, center=True, min_periods=1).mean()
    return out


def build_features(weather_slice: pd.DataFrame) -> pd.DataFrame:
    """weather_slice: колонки {model}__{var} (+ lead_day), индекс datetime почасовой."""
    blocks = [_model_block(weather_slice, m) for m in WEATHER_MODELS
              if f"{m}__wind_speed_100m" in weather_slice.columns]
    X = pd.concat(blocks, axis=1)

    # согласие источников: среднее и разброс ансамбля по ws100
    ws_cols = [c for c in X.columns if c.endswith("_ws100")]
    X["ens_ws100_mean"] = X[ws_cols].mean(axis=1)
    X["ens_ws100_std"] = X[ws_cols].std(axis=1)
    X["ens_ws100_cube"] = X["ens_ws100_mean"] ** 3

    idx = X.index
    X["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    X["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    X["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)
    if "lead_day" in weather_slice.columns:
        X["lead_day"] = weather_slice["lead_day"]
    return X


def training_matrix(weather: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Стек по lead_day: каждый час встречается с прогнозом за 1 и за 2 дня."""
    parts_X, parts_y = [], []
    for lead in (1, 2):
        cols = {c: c.replace(f"__d{lead}", "") for c in weather.columns if c.endswith(f"__d{lead}")}
        sl = weather[list(cols)].rename(columns=cols)
        sl["lead_day"] = lead
        X = build_features(sl)
        y = target.reindex(X.index)
        ok = y.notna() & X[[c for c in X.columns if c.endswith("_ws100")]].notna().any(axis=1)
        parts_X.append(X[ok])
        parts_y.append(y[ok])
    return pd.concat(parts_X), pd.concat(parts_y)
