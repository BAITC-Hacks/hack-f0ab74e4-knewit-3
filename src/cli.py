"""Точки входа.

  python -m src.cli train                                       # обучение + метрики holdout
  python -m src.cli run-agent --start 2026-01-31 --end 2026-02-28 [--no-llm]
  python -m src.cli check-tz                                    # тест склейки таймзон
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import pandas as pd

from src.config import FORECASTS, TEST_END, TRAIN_START, TURBINES


def _load_weather() -> pd.DataFrame:
    from src.weather.openmeteo import get_weather
    return get_weather(TRAIN_START, TEST_END)


def cmd_train(_args) -> None:
    from src.models.train import train_all
    print("Загрузка архивных прогнозов (кэш data/weather_cache/)...")
    weather = _load_weather()
    print(f"Погода: {weather.shape[0]} часов x {weather.shape[1]} колонок. Обучение...")
    reports = train_all(weather)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


def cmd_run_agent(args) -> None:
    from src.agent.loop import run_day_llm, run_day_no_llm
    weather = _load_weather()
    d, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    runner = run_day_no_llm if args.no_llm else run_day_llm
    while d <= end:
        res = runner(d.isoformat(), weather)
        print(f"[{d}] ok={res['validation_ok']} -> {', '.join(res['files'])}")
        d += timedelta(days=1)
    _build_submission()


def _build_submission() -> None:
    """Сводный файл: на каждый час — прогноз минимального lead time (свежайший запуск)."""
    frames = []
    for t in TURBINES:
        for f in sorted(FORECASTS.glob(f"forecast_t{t}_*.csv")):
            frames.append(pd.read_csv(f, parse_dates=["datetime"]))
    if not frames:
        return
    allp = pd.concat(frames)
    allp = (allp.sort_values(["turbine", "datetime", "lead_day"])
                .groupby(["turbine", "datetime"], as_index=False).first())
    allp = allp[(allp["datetime"] >= "2026-02-01") & (allp["datetime"] <= f"{TEST_END} 23:59")]
    out = FORECASTS / "submission.csv"
    allp[["turbine", "datetime", "lead_day", "power_pred"]].to_csv(out, index=False)
    print(f"Сводный прогноз: {out} ({len(allp)} строк, "
          f"{allp['datetime'].min()} -> {allp['datetime'].max()})")


def cmd_check_tz(_args) -> None:
    """Сдвиг максимальной корреляции прогнозного и измеренного ветра должен быть 0 (ADR-006)."""
    from src.features.dataset import load_hourly
    weather = _load_weather()
    ws = weather["best_match__wind_speed_100m__d1"]
    for t in TURBINES:
        meas = load_hourly(t)["wind_meas"]
        joined = pd.concat([ws, meas], axis=1).dropna()
        corrs = {lag: joined.iloc[:, 0].shift(lag).corr(joined.iloc[:, 1])
                 for lag in range(-6, 7)}
        best = max(corrs, key=corrs.get)
        status = "OK" if best == 0 else "ОШИБКА СКЛЕЙКИ ВРЕМЕНИ"
        print(f"Турбина {t}: лучший лаг {best} (corr={corrs[best]:.3f}) — {status}")


def main() -> None:
    p = argparse.ArgumentParser(prog="windcast")
    sub = p.add_subparsers(required=True)
    sub.add_parser("train").set_defaults(func=cmd_train)
    ra = sub.add_parser("run-agent")
    ra.add_argument("--start", required=True)
    ra.add_argument("--end", required=True)
    ra.add_argument("--no-llm", action="store_true")
    ra.set_defaults(func=cmd_run_agent)
    sub.add_parser("check-tz").set_defaults(func=cmd_check_tz)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
