"""Матрица фич из архивных прогнозов погоды.

Вход инференса — только прогноз (previous-runs API); измеренные на турбине величины
в фичи не входят (на инференсе их нет). Обучение и инференс строят признаки одинаково
из среза выпуска; это устраняет расхождение расчёта лагов и окон на его границах.

Не каждая погодная модель отдаёт все переменные (ECMWF — без 80/120 м, UKMO/JMA/CMA —
только 10 м): блок строится из доступного, пропуски LightGBM обрабатывает нативно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.weather.openmeteo import get_issued_forecast, issue_dates

R_AIR = 287.05  # Дж/(кг·К)
FEATURE_DECIMALS = 10


def _model_block(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Фичи одного погодного источника — из тех переменных, что у него есть.

    Считается по переданному срезу: лаги/опережения/окна не выходят за его границы.
    Обучение и инференс передают сюда срез одного выпуска (get_issued_forecast).
    """
    p = f"{model}__"
    have = lambda v: f"{p}{v}" in df.columns and df[f"{p}{v}"].notna().any()
    cols: dict[str, pd.Series] = {}
    for h in (10, 80, 100, 120):
        if have(f"wind_speed_{h}m"):
            cols[f"{model}_ws{h}"] = df[f"{p}wind_speed_{h}m"]
    # основная скорость: высота ступицы ~100 м, иначе лучшее из доступного
    ws = None
    for v in ("wind_speed_100m", "wind_speed_120m", "wind_speed_80m", "wind_speed_10m"):
        if have(v):
            ws = df[f"{p}{v}"]
            break
    if ws is None:
        return pd.DataFrame(cols, index=df.index)
    cols[f"{model}_ws_main"] = ws
    cols[f"{model}_ws_cube"] = ws ** 3            # физика: P ~ rho * v^3
    if have("wind_gusts_10m"):
        cols[f"{model}_gust"] = df[f"{p}wind_gusts_10m"]
    if have("wind_speed_120m") and have("wind_speed_10m"):
        cols[f"{model}_shear"] = (df[f"{p}wind_speed_120m"] - df[f"{p}wind_speed_10m"]).clip(lower=-20)
    if have("wind_direction_100m"):
        wd = np.deg2rad(df[f"{p}wind_direction_100m"])
        cols[f"{model}_wd_sin"] = np.sin(wd)
        cols[f"{model}_wd_cos"] = np.cos(wd)
    if have("temperature_2m"):
        cols[f"{model}_temp"] = df[f"{p}temperature_2m"]
        if have("surface_pressure"):
            t_k = df[f"{p}temperature_2m"] + 273.15
            rho = (df[f"{p}surface_pressure"] * 100) / (R_AIR * t_k)
            cols[f"{model}_rho"] = rho
            cols[f"{model}_pwr_density"] = 0.5 * rho * ws ** 3  # Вт/м^2
    # инерция и фазовые ошибки фронтов: лаги и окна по основной скорости
    for lag in (1, 2, 3):
        cols[f"{model}_ws_lag{lag}"] = ws.shift(lag)
        cols[f"{model}_ws_lead{lag}"] = ws.shift(-lag)
    cols[f"{model}_ws_roll3"] = ws.rolling(3, center=True, min_periods=1).mean()
    cols[f"{model}_ws_roll6"] = ws.rolling(6, center=True, min_periods=1).mean()
    cols[f"{model}_ws_rollstd6"] = ws.rolling(6, center=True, min_periods=2).std()
    return pd.DataFrame(cols, index=df.index)


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
    # libm разных платформ даёт младшие биты, на которых дерево может разделить
    # математически равные значения. Один контракт точности для обучения и прогноза;
    # изменение требует переобучения. 1e-10 намного меньше точности погодных входов.
    return X.round(FEATURE_DECIMALS)


def issued_feature_stack(weather: pd.DataFrame,
                         issues: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Признаки всех выпусков архива — ровно так, как их строит ежедневный прогноз.

    Для каждого дня выпуска D: build_features(get_issued_forecast(D, weather)). Строка
    (D, целевой час) встречается один раз; каждый целевой час — дважды (lead 1 от D-1,
    lead 2 от D-2), кроме краёв архива. Не зависит от турбины, поэтому считается один
    раз на процесс и передаётся в training_matrix(stack=...).

    Возвращает (X, meta): индекс X — целевой час; meta той же длины и порядка с колонками
    issue_date (str), datetime, lead_day — происхождение каждой строки.
    """
    frames, metas = [], []
    for issue_date in (issues if issues is not None else issue_dates(weather)):
        X = build_features(get_issued_forecast(issue_date, weather))
        frames.append(X)
        metas.append(pd.DataFrame({"issue_date": issue_date, "datetime": X.index,
                                   "lead_day": X["lead_day"].values}))
    if not frames:
        raise ValueError("в архиве нет ни одного выпуска")
    # канонический порядок колонок: срез с наибольшим набором источников, затем остальные
    canonical = list(max(frames, key=lambda f: f.shape[1]).columns)
    extra = [c for f in frames for c in f.columns if c not in canonical]
    canonical += list(dict.fromkeys(extra))
    X_all = pd.concat([f.reindex(columns=canonical) for f in frames])
    meta = pd.concat(metas, ignore_index=True)
    return X_all, meta


def training_matrix(weather: pd.DataFrame, target: pd.Series, *,
                    target_end: str | pd.Timestamp | None = None,
                    stack: tuple[pd.DataFrame, pd.DataFrame] | None = None,
                    with_meta: bool = False):
    """Обучающая матрица из тех же срезов выпусков, что и инференс.

    Строка остаётся, если известен таргет и есть ансамблевая скорость ветра; таргет не
    заполняется. target_end — верхняя граница целевых часов (данные позже отсечки в
    обучение не попадают). stack — результат issued_feature_stack (повторное использование).
    Возвращает (X, y) с одинаковым индексом целевых часов; with_meta=True добавляет meta.
    """
    X_all, meta = stack if stack is not None else issued_feature_stack(weather)
    y = target.reindex(X_all.index)
    ok = np.asarray(y.notna() & X_all["ens_ws_mean"].notna(), dtype=bool)
    if target_end is not None:
        ok = ok & np.asarray(X_all.index <= pd.Timestamp(target_end))
    X, y, meta = X_all[ok], y[ok], meta[ok].reset_index(drop=True)
    return (X, y, meta) if with_meta else (X, y)


def training_provenance(weather: pd.DataFrame, target: pd.Series, *,
                        target_end=None, stack=None) -> pd.DataFrame:
    """Происхождение строк training_matrix: issue_date, datetime, lead_day (тот же порядок)."""
    return training_matrix(weather, target, target_end=target_end, stack=stack, with_meta=True)[2]
