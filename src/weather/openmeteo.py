"""Клиент Open-Meteo Previous Runs API: архивные прогнозы, доступные на момент прогнозирования.

Семантика по документации Open-Meteo (docs/FEATURE_AVAILABILITY.md): значение колонки
`<var>_previous_dayN` в час T — «the value that was predicted N*24 hours before valid time»,
то есть из прогона, инициализированного не позже чем за N суток до T. Выпуск «в день D
на 48 часов» = previous_day1 для часов дня D+1 и previous_day2 для часов дня D+2; это
скользящий набор прогонов дня D, а не один прогон. Фактическая погода сюда попасть не может.
Все ответы кэшируются в data/weather_cache/ — повторные запуски работают офлайн.
"""
from __future__ import annotations

import json
import hashlib
from datetime import date, timedelta

import httpx
import pandas as pd

from src.config import (LEAD_DAYS, TIMEZONE, WEATHER_CACHE, WEATHER_MODELS,
                        WEATHER_POINT, WEATHER_VARS)

API = "https://previous-runs-api.open-meteo.com/v1/forecast"


def _cache_path(model: str, start: str, end: str) -> "Path":
    key = hashlib.md5(f"{model}|{start}|{end}|{','.join(WEATHER_VARS)}".encode()).hexdigest()[:10]
    return WEATHER_CACHE / f"{model}_{start}_{end}_{key}.json"


def _fetch_chunk(model: str, start: str, end: str) -> dict:
    """Один запрос к API за период; результат кэшируется на диске."""
    path = _cache_path(model, start, end)
    if path.exists():
        return json.loads(path.read_text())
    hourly = [f"{v}_previous_day{n}" for v in WEATHER_VARS for n in LEAD_DAYS]
    params = {
        "latitude": WEATHER_POINT[0], "longitude": WEATHER_POINT[1],
        "hourly": ",".join(hourly), "models": model,
        "start_date": start, "end_date": end, "timezone": TIMEZONE,
        "wind_speed_unit": "ms",
    }
    r = httpx.get(API, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo без hourly: {data}")
    WEATHER_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Полугодовые куски, чтобы не упираться в лимит размера ответа."""
    out, cur = [], date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while cur <= stop:
        nxt = min(cur + timedelta(days=182), stop)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt + timedelta(days=1)
    return out


def get_weather(start: str, end: str, models: list[str] | None = None) -> pd.DataFrame:
    """Архивные прогнозы за период, все модели и оба lead time.

    Индекс — naive datetime в Asia/Almaty (как в датасете). Колонки:
    {model}__{var}__d{N}, ветер в м/с.
    """
    frames = []
    for model in (models or WEATHER_MODELS):
        parts = []
        for s, e in _chunks(start, end):
            data = _fetch_chunk(model, s, e)
            h = data["hourly"]
            df = pd.DataFrame(h).assign(time=lambda d: pd.to_datetime(d["time"]))
            df = df.set_index("time")
            parts.append(df)
        mdf = pd.concat(parts)
        mdf.columns = [
            f"{model}__{c.replace('_previous_day', '__d')}" for c in mdf.columns
        ]
        frames.append(mdf)
    out = pd.concat(frames, axis=1)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    # не все модели отдают все переменные (у ECMWF нет части высот) — None -> NaN,
    # LightGBM обрабатывает пропуски нативно
    return out.apply(pd.to_numeric, errors="coerce")


def _fetch_point_chunk(model: str, start: str, end: str, lat: float, lon: float,
                       variables: list[str]) -> dict:
    """Запрос для произвольной точки/набора переменных (пространственные фичи)."""
    key = hashlib.md5(f"{model}|{start}|{end}|{lat}|{lon}|{','.join(variables)}".encode()).hexdigest()[:10]
    path = WEATHER_CACHE / f"sp_{model}_{start}_{end}_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    hourly = [f"{v}_previous_day{n}" for v in variables for n in LEAD_DAYS]
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly),
              "models": model, "start_date": start, "end_date": end,
              "timezone": TIMEZONE, "wind_speed_unit": "ms"}
    r = httpx.get(API, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo без hourly: {data}")
    WEATHER_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def get_spatial_weather(start: str, end: str) -> pd.DataFrame:
    """Прогнозы в 4 точках вокруг станции (N/S/E/W): колонки sp{dir}_{model}__{var}__dN."""
    from src.config import SPATIAL_MODELS, SPATIAL_POINTS, SPATIAL_VARS
    frames = []
    for direction, (lat, lon) in SPATIAL_POINTS.items():
        for model in SPATIAL_MODELS:
            parts = []
            for s, e in _chunks(start, end):
                data = _fetch_point_chunk(model, s, e, lat, lon, SPATIAL_VARS)
                df = pd.DataFrame(data["hourly"]).assign(
                    time=lambda d: pd.to_datetime(d["time"])).set_index("time")
                parts.append(df)
            mdf = pd.concat(parts)
            mdf.columns = [f"sp{direction}_{model}__{c.replace('_previous_day', '__d')}"
                           for c in mdf.columns]
            frames.append(mdf)
    out = pd.concat(frames, axis=1)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.apply(pd.to_numeric, errors="coerce")


def _check_archive_index(weather: pd.DataFrame) -> None:
    """Архив — почасовой naive-индекс в локальном времени Asia/Almaty (ADR-006)."""
    idx = weather.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("индекс архива погоды должен быть DatetimeIndex (локальное время)")
    if idx.tz is not None:
        raise ValueError("индекс архива должен быть naive в Asia/Almaty, а не tz-aware "
                         f"({idx.tz}); склейка с датасетом идёт по локальному времени")
    if not idx.is_monotonic_increasing:
        raise ValueError("индекс архива погоды должен быть отсортирован по времени")


def _day_index(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(day, periods=24, freq="h")


def _issued_day(weather: pd.DataFrame, day: pd.Timestamp, lead: int) -> pd.DataFrame | None:
    """Полные 24 часа дня `day` из колонок lead `lead`; None — если дня в архиве нет вовсе.

    Пропавшие внутри дня часы становятся NaN-строками, а не склеиваются с соседями:
    иначе лаги и окна в build_features сдвинулись бы по времени незаметно.
    """
    hours = _day_index(day)
    sel = weather.loc[hours[0]:hours[-1]]
    if sel.empty:
        return None
    suffix = f"__d{lead}"
    cols = {c: c[: -len(suffix)] for c in sel.columns if c.endswith(suffix)}
    part = sel[list(cols)].rename(columns=cols).reindex(hours)
    part["lead_day"] = lead
    return part


def issue_dates(weather: pd.DataFrame) -> list[str]:
    """Дни выпуска D, для которых архив содержит день D+1 (lead 1).

    Первый выпуск — за день до начала архива (у него есть D+1 и D+2), последний — за день
    до конца архива (только D+1, 24 часа: край архива). Выпуск за два дня до начала архива
    имел бы только lead 2 — такой формы инференс не производит, он исключён.
    """
    _check_archive_index(weather)
    days = pd.DatetimeIndex(weather.index.normalize().unique())
    return [(d - timedelta(days=1)).date().isoformat() for d in days]


def get_issued_forecast(issue_date: str, weather: pd.DataFrame) -> pd.DataFrame:
    """Срез «что было доступно в день issue_date»: 48 часов D+1 (lead 1) и D+2 (lead 2).

    Возвращает длинный DataFrame: индекс datetime (naive, Asia/Almaty), колонки
    {model}__{var} + lead_day (int). Каждый присутствующий день — ровно 24 строки;
    отсутствующий в архиве день (край архива) пропускается, поэтому на последнем дне
    архива срез состоит из 24 часов lead 1. Ровно эту функцию использует и обучение
    (src/features/build.py), чтобы признаки строились из одного и того же выпуска.
    """
    _check_archive_index(weather)
    d = pd.Timestamp(date.fromisoformat(issue_date))
    rows = []
    for lead in LEAD_DAYS:
        part = _issued_day(weather, d + timedelta(days=lead), lead)
        if part is not None:
            rows.append(part)
    if not rows:
        raise ValueError(f"нет погодных данных для запуска {issue_date}")
    return pd.concat(rows)
