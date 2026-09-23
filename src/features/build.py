"""Матрица фич из архивных прогнозов погоды.

Вход инференса — только прогноз (previous-runs API); измеренные на турбине величины
в фичи не входят (на инференсе их нет). Обучение на тех же фичах устраняет сдвиг
распределений train/inference и учит модель поправлять ошибки погодной модели.

Не каждая погодная модель отдаёт все переменные (ECMWF — без 80/120 м, UKMO/JMA/CMA —
только 10 м): блок строится из доступного, пропуски LightGBM обрабатывает нативно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import WEATHER_MODELS

R_AIR = 287.05  # Дж/(кг·К)


def _model_block(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Фичи одного погодного источника — из тех переменных, что у него есть."""
    p = f"{model}__"
    have = lambda v: f"{p}{v}" in df.columns and df[f"{p}{v}"].notna().any()
    out = pd.DataFrame(index=df.index)
    for h in (10, 80, 100, 120):
        if have(f"wind_speed_{h}m"):
            out[f"{model}_ws{h}"] = df[f"{p}wind_speed_{h}m"]
    # основная скорость: высота ступицы ~100 м, иначе лучшее из доступного
    ws = None
    for v in ("wind_speed_100m", "wind_speed_120m", "wind_speed_80m", "wind_speed_10m"):
        if have(v):
            ws = df[f"{p}{v}"]
            break
    if ws is None:
        return out
    out[f"{model}_ws_main"] = ws
    out[f"{model}_ws_cube"] = ws ** 3            # физика: P ~ rho * v^3
    if have("wind_gusts_10m"):
        out[f"{model}_gust"] = df[f"{p}wind_gusts_10m"]
    if have("wind_speed_120m") and have("wind_speed_10m"):
        out[f"{model}_shear"] = (df[f"{p}wind_speed_120m"] - df[f"{p}wind_speed_10m"]).clip(lower=-20)
    if have("wind_direction_100m"):
        wd = np.deg2rad(df[f"{p}wind_direction_100m"])
        out[f"{model}_wd_sin"] = np.sin(wd)
        out[f"{model}_wd_cos"] = np.cos(wd)
    if have("temperature_2m"):
        out[f"{model}_temp"] = df[f"{p}temperature_2m"]
        if have("surface_pressure"):
            t_k = df[f"{p}temperature_2m"] + 273.15
            rho = (df[f"{p}surface_pressure"] * 100) / (R_AIR * t_k)
            out[f"{model}_rho"] = rho
            out[f"{model}_pwr_density"] = 0.5 * rho * ws ** 3  # Вт/м^2
    # инерция и фазовые ошибки фронтов: лаги и окна по основной скорости
    for lag in (1, 2, 3):
        out[f"{model}_ws_lag{lag}"] = ws.shift(lag)
        out[f"{model}_ws_lead{lag}"] = ws.shift(-lag)
    out[f"{model}_ws_roll3"] = ws.rolling(3, center=True, min_periods=1).mean()
    out[f"{model}_ws_roll6"] = ws.rolling(6, center=True, min_periods=1).mean()
    out[f"{model}_ws_rollstd6"] = ws.rolling(6, center=True, min_periods=2).std()
    return out


def _spatial_block(df: pd.DataFrame) -> pd.DataFrame:
    """Градиенты давления/температуры через станцию (движущая сила gap wind в коридоре)
    и ветер в окрестных точках. Колонки входа: sp{N|S|E|W}_{model}__{var}."""
    out = pd.DataFrame(index=df.index)
    models = sorted({c.split("__")[0].split("_", 1)[1] for c in df.columns
                     if c.startswith("sp") and "__" in c})
    for m in models:
        col = lambda d, v: f"sp{d}_{m}__{v}"
        have = lambda d, v: col(d, v) in df.columns and df[col(d, v)].notna().any()
        if all(have(d, "surface_pressure") for d in "NSEW"):
            out[f"sp_{m}_dp_ns"] = df[col("N", "surface_pressure")] - df[col("S", "surface_pressure")]
            out[f"sp_{m}_dp_ew"] = df[col("E", "surface_pressure")] - df[col("W", "surface_pressure")]
        if all(have(d, "temperature_2m") for d in "NS"):
            out[f"sp_{m}_dt_ns"] = df[col("N", "temperature_2m")] - df[col("S", "temperature_2m")]
        ws_pts = [df[col(d, "wind_speed_100m")] for d in "NSEW" if have(d, "wind_speed_100m")]
        if ws_pts:
            out[f"sp_{m}_ws_around"] = pd.concat(ws_pts, axis=1).mean(axis=1)
    return out


def build_features(weather_slice: pd.DataFrame) -> pd.DataFrame:
    """weather_slice: колонки {model}__{var} (+ lead_day), индекс datetime почасовой."""
    models = sorted({c.split("__")[0] for c in weather_slice.columns
                     if "__" in c and not c.startswith("sp")})
    blocks = [b for m in models if not (b := _model_block(weather_slice, m)).empty]
    sp = _spatial_block(weather_slice)
    if not sp.empty:
        blocks.append(sp)
    X = pd.concat(blocks, axis=1)

    # согласие источников: среднее и разброс ансамбля по основной скорости
    ws_cols = [c for c in X.columns if c.endswith("_ws_main")]
    X["ens_ws_mean"] = X[ws_cols].mean(axis=1)
    X["ens_ws_std"] = X[ws_cols].std(axis=1)
    X["ens_ws_cube"] = X["ens_ws_mean"] ** 3

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
        ok = y.notna() & X["ens_ws_mean"].notna()
        parts_X.append(X[ok])
        parts_y.append(y[ok])
    return pd.concat(parts_X), pd.concat(parts_y)
