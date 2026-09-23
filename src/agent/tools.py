"""Тулы агента. Каждый — обычная функция над контекстом прогнозного дня;
работают и без LLM (режим --no-llm вызывает их в штатном порядке)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import FORECASTS, TURBINES
from src.features.build import build_features
from src.models.predict import predict
from src.weather.openmeteo import get_issued_forecast


class DayContext:
    """Состояние одного прогнозного запуска (issue_date -> 48 часов вперёд)."""

    def __init__(self, issue_date: str, weather: pd.DataFrame):
        self.issue_date = issue_date
        self.weather = weather          # полный архив (previous-runs, кэш)
        self.slice: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.preds: dict[int, pd.DataFrame] = {}
        self.validation: dict | None = None


def fetch_weather(ctx: DayContext) -> dict:
    """Архивный прогноз, доступный в issue_date: 24ч lead-1 + 24ч lead-2."""
    ctx.slice = get_issued_forecast(ctx.issue_date, ctx.weather)
    ws = ctx.slice[[c for c in ctx.slice.columns if c.endswith("wind_speed_100m")]]
    n_missing = int(ws.isna().all(axis=1).sum())
    return {
        "issue_date": ctx.issue_date,
        "hours": len(ctx.slice),
        "period": [str(ctx.slice.index.min()), str(ctx.slice.index.max())],
        "wind100_mean_ms": round(float(ws.mean(axis=1).mean()), 2),
        "wind100_max_ms": round(float(ws.mean(axis=1).max()), 2),
        "hours_without_any_model": n_missing,
        "models_present": sorted({c.split("__")[0] for c in ctx.slice.columns if "__" in c}),
    }


def prepare_features(ctx: DayContext) -> dict:
    assert ctx.slice is not None, "сначала fetch_weather"
    ctx.features = build_features(ctx.slice)
    na_frac = float(ctx.features.isna().mean().mean())
    return {"rows": len(ctx.features), "n_features": ctx.features.shape[1],
            "na_fraction": round(na_frac, 4)}


def run_model(ctx: DayContext) -> dict:
    assert ctx.features is not None, "сначала prepare_features"
    out = {}
    for t in TURBINES:
        ctx.preds[t] = predict(t, ctx.features)
        p = ctx.preds[t]["power_pred"]
        out[f"turbine_{t}"] = {
            "mean": round(float(p.mean()), 3), "max": round(float(p.max()), 3),
            "hours": len(p),
            "expected_energy_norm_24h": round(float(p[ctx.preds[t]["lead_day"] == 1].sum()), 2),
        }
    return out


def validate_forecast(ctx: DayContext) -> dict:
    """Физические границы, полнота 48 часов, деградация со вторым горизонтом."""
    assert ctx.preds, "сначала run_model"
    checks = {}
    for t, p in ctx.preds.items():
        pr = p["power_pred"]
        checks[f"turbine_{t}"] = {
            "in_bounds_0_1": bool((pr.between(0, 1)).all()),
            # 48 ч штатно; 24 ч допустимо на краю архива (последний день периода)
            "hours_ok": bool(len(pr) in (24, 48)),
            "hours": int(len(pr)),
            "has_nan": bool(pr.isna().any()),
            "flatline": bool(pr.std() < 1e-4),  # подозрительно постоянный прогноз
        }
    ok = all(c["in_bounds_0_1"] and c["hours_ok"] and not c["has_nan"] and not c["flatline"]
             for c in checks.values())
    ctx.validation = {"ok": ok, "checks": checks}
    return ctx.validation


def compare_with_previous(ctx: DayContext) -> dict:
    """Дрейф против вчерашнего запуска: вчера эти же часы были предсказаны с lead 2."""
    prev_date = (date.fromisoformat(ctx.issue_date) - timedelta(days=1)).isoformat()
    out = {}
    for t in TURBINES:
        prev_file = FORECASTS / f"forecast_t{t}_{prev_date}.csv"
        if not prev_file.exists():
            out[f"turbine_{t}"] = "нет предыдущего запуска"
            continue
        prev = pd.read_csv(prev_file, parse_dates=["datetime"]).set_index("datetime")
        cur = ctx.preds[t]
        overlap = cur.index.intersection(prev.index)
        if len(overlap) == 0:
            out[f"turbine_{t}"] = "нет пересечения часов"
            continue
        diff = (cur.loc[overlap, "power_pred"] - prev.loc[overlap, "power_pred"])
        out[f"turbine_{t}"] = {
            "hours_overlap": len(overlap),
            "mean_abs_update": round(float(diff.abs().mean()), 3),
            "max_abs_update": round(float(diff.abs().max()), 3),
            "significant_update": bool(diff.abs().mean() > 0.08),
        }
    return out


def write_outputs(ctx: DayContext, analysis: str) -> dict:
    """CSV прогноза на каждый день + markdown-отчёт агента."""
    FORECASTS.mkdir(exist_ok=True)
    files = []
    for t, p in ctx.preds.items():
        df = p.reset_index().rename(columns={"index": "datetime", "time": "datetime"})
        df.insert(0, "turbine", t)
        df["horizon_h"] = range(1, len(df) + 1)
        path = FORECASTS / f"forecast_t{t}_{ctx.issue_date}.csv"
        df[["turbine", "datetime", "horizon_h", "lead_day",
            "power_pred", "power_baseline", "power_lgb"]].to_csv(path, index=False)
        files.append(path.name)
    report = FORECASTS / f"report_{ctx.issue_date}.md"
    report.write_text(f"# Отчёт агента — запуск {ctx.issue_date}\n\n{analysis}\n")
    return {"files": files, "report": report.name}
