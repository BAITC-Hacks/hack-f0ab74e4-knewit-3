"""Клиент Open-Meteo Previous Runs API: архивные прогнозы, доступные на момент прогнозирования.

Семантика: значение колонки `<var>_previous_dayN` в час T — из прогона погодной модели,
выпущенного за N суток до T. Прогноз «сделанный в день D на 48 часов» = previous_day1
для часов дня D+1 и previous_day2 для часов дня D+2. Фактическая погода сюда попасть не может.
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


def get_issued_forecast(issue_date: str, weather: pd.DataFrame) -> pd.DataFrame:
    """Срез «что было доступно в день issue_date»: 48 часов D+1 (lead 1) и D+2 (lead 2).

    Возвращает длинный DataFrame: индекс datetime, колонки {model}__{var} + lead_day.
    """
    d = date.fromisoformat(issue_date)
    rows = []
    for lead in LEAD_DAYS:
        day = d + timedelta(days=lead)
        sel = weather.loc[str(day)]
        cols = {c: c.replace(f"__d{lead}", "") for c in sel.columns if c.endswith(f"__d{lead}")}
        part = sel[list(cols)].rename(columns=cols)
        part["lead_day"] = lead
        rows.append(part)
    return pd.concat(rows)
