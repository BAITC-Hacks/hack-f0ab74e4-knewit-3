"""Ретроспективная оценка точной прогнозной конфигурации через продовый путь инференса.

    python -m src.backtest.evaluate --output-dir runs/evaluation-v1

Протокол retrospective-issue-replay-v1:

- fit      2024-03-01..2025-09-30 — посадка модели при подборе;
- tune     2025-10-01..2025-11-30 — ранняя остановка и вес бленда;
- evaluate 2025-12-01..2026-01-31 — воспроизведение по датам выпуска.

Оцениваемый ансамбль (5 LightGBM + изотонический baseline + квантили P10/P90)
сажается ТОЛЬКО на целях по 2025-11-30 и предсказывает оценочный период ровно тем
же путём, что и февральская подача: get_issued_forecast -> build_features ->
predict. Это РЕТРОСПЕКТИВНАЯ оценка, а не нетронутый тест: декабрь-январь ранее
использовались для отбора источников и настроек, что могло оптимистично сместить
выбор конфигурации. Февральского факта у команды нет — точность февраля не
изобретается.

Оценочные часы встречаются с lead 1 и lead 2, кроме lead 2 за 01.12: его выпуск
29.11 предшествует отсечке обучения модели и исключён. Строки не усредняются;
метрики группируются по lead_day, полнота воспроизведения указана в coverage.

Команда пишет только в --output-dir: канонические forecasts/ и models_artifacts/
не трогаются, LLM не вызывается, сеть не нужна (погодный кэш в репозитории).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TURBINES, WEATHER_MODELS

SCHEMA_VERSION = 1
PROTOCOL = "retrospective-issue-replay-v1"
FIT_PERIOD = ("2024-03-01", "2025-09-30")
TUNE_PERIOD = ("2025-10-01", "2025-11-30")
EVAL_PERIOD = ("2025-12-01", "2026-01-31")
TRAINED_THROUGH = TUNE_PERIOD[1]

CSV_NAME = "evaluation_predictions.csv"
REPORT_NAME = "evaluation_report.json"


def split_masks(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Маски fit/tune/evaluate по времени ЦЕЛИ. Дубли одного часа (lead 1 и 2)
    получают одинаковое назначение; случайного разбиения нет по построению."""
    fit = (index >= FIT_PERIOD[0]) & (index <= f"{FIT_PERIOD[1]} 23:59")
    tune = (index >= TUNE_PERIOD[0]) & (index <= f"{TUNE_PERIOD[1]} 23:59")
    ev = (index >= EVAL_PERIOD[0]) & (index <= f"{EVAL_PERIOD[1]} 23:59")
    return np.asarray(fit), np.asarray(tune), np.asarray(ev)


def _fit_evaluation_models(weather: pd.DataFrame, output_dir: Path) -> dict:
    """Посадка оцениваемой конфигурации с отсечкой 2025-11-30, артефакты — в output_dir."""
    from src.features.build import issued_feature_stack, training_matrix
    from src.features.dataset import load_hourly
    from src.models.train import save_artifact, select_and_fit

    stack = issued_feature_stack(weather)
    selections = {}
    for turbine in TURBINES:
        hourly = load_hourly(turbine)
        X, y = training_matrix(weather, hourly["target"], stack=stack,
                               target_end=f"{TRAINED_THROUGH} 23:00")
        fit, tune, ev = split_masks(X.index)
        assert not ((fit | tune) & ev).any(), "оценочные цели попали в обучение"
        artifact, sel = select_and_fit(X, y, fit, tune, fit | tune)
        save_artifact(artifact, turbine, output_dir)
        sel.pop("tune_pred_baseline"), sel.pop("tune_pred_lgb")
        sel["features"] = artifact["features"]
        selections[str(turbine)] = sel
    return selections


def _issue_range() -> tuple[date, date]:
    # Условный выпуск в конце дня: цели всего дня trained_through уже известны.
    # Более ранний выпуск не мог пользоваться этой обученной моделью.
    first = max(date.fromisoformat(EVAL_PERIOD[0]) - timedelta(days=2),
                date.fromisoformat(TRAINED_THROUGH))
    return first, date.fromisoformat(EVAL_PERIOD[1]) - timedelta(days=1)


def _replay(weather: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Воспроизведение по датам выпуска: только срезы, доступные в день выпуска."""
    from src.features.build import build_features
    from src.features.dataset import load_hourly
    from src.models.predict import predict
    from src.weather.openmeteo import get_issued_forecast

    eval_lo = pd.Timestamp(EVAL_PERIOD[0])
    eval_hi = pd.Timestamp(f"{EVAL_PERIOD[1]} 23:59")
    first, last = _issue_range()

    truth = {
        t: load_hourly(t)[["power", "target"]]  # power — все наблюдаемые, target — чистые
        for t in TURBINES
    }

    rows, skipped = [], []
    issue = first
    while issue <= last:
        issue_str = issue.isoformat()
        try:
            slice_ = get_issued_forecast(issue_str, weather)
            features = build_features(slice_)
        except Exception as exc:  # пустой/битый срез фиксируется, не замалчивается
            skipped.append({"issue_date": issue_str, "reason": str(exc)})
            issue += timedelta(days=1)
            continue
        for turbine in TURBINES:
            pred = predict(turbine, features, artifacts_dir=output_dir)
            keep = (pred.index >= eval_lo) & (pred.index <= eval_hi)
            part = pred[keep]
            if part.empty:
                continue
            tr = truth[turbine].reindex(part.index)
            out = pd.DataFrame({
                "turbine": turbine,
                "issue_date": issue_str,
                "datetime": part.index.strftime("%Y-%m-%d %H:%M:%S"),
                "lead_day": part["lead_day"].astype(int).values,
                "power_true": tr["power"].values,        # NaN остаётся NaN: не заполняем
                "power_pred": part["power_pred"].values,
                "power_baseline": part["power_baseline"].values,
                "power_p10": part.get("power_p10", pd.Series(np.nan, index=part.index)).values,
                "power_p90": part.get("power_p90", pd.Series(np.nan, index=part.index)).values,
                "target_eligible": tr["target"].notna().values,
                "trained_through": TRAINED_THROUGH,
            })
            rows.append(out)
        issue += timedelta(days=1)
    return pd.concat(rows, ignore_index=True), skipped


def _group_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for turbine, tdf in df.groupby("turbine"):
        tkey = f"turbine_{turbine}"
        out[tkey] = {}
        for lead, ldf in tdf.groupby("lead_day"):
            err = ldf["power_pred"] - ldf["power_true"]
            base = ldf["power_baseline"] - ldf["power_true"]
            out[tkey][f"lead_{lead}"] = {
                "n": int(len(ldf)),
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt((err ** 2).mean())),
                "mae_baseline": float(base.abs().mean()),
            }
        err_all = tdf["power_pred"] - tdf["power_true"]
        out[tkey]["all_leads"] = {"n": int(len(tdf)),
                                  "mae": float(err_all.abs().mean()),
                                  "rmse": float(np.sqrt((err_all ** 2).mean()))}
    return out


def compute_metrics(csv_path: Path | str) -> dict:
    """Метрики строго из сохранённого CSV — то, что записано, то и оценивается."""
    df = pd.read_csv(csv_path)
    for column in ("power_pred", "power_baseline"):
        invalid = ~np.isfinite(df[column].to_numpy(dtype=float))
        if invalid.any():
            raise ValueError(f"{column}: {int(invalid.sum())} нечисловых прогнозов; "
                             "оценка остановлена, строки не исключаются из MAE молча")
    observed = df[df["power_true"].notna()]
    clean = observed[observed["target_eligible"]]

    metrics = {
        "rows": {
            "total": int(len(df)),
            "missing_truth": int(df["power_true"].isna().sum()),
            "observed": int(len(observed)),
            "clean_targets": int(len(clean)),
            "excluded_as_downtime_or_gap": int(len(observed) - len(clean)),
        },
        "clean_targets": _group_metrics(clean),
        "all_observed": _group_metrics(observed),
    }
    band = clean[clean["power_p10"].notna() & clean["power_p90"].notna()]
    if len(band):
        covered = ((band["power_true"] >= band["power_p10"])
                   & (band["power_true"] <= band["power_p90"])).mean()
        # измеренное эмпирическое покрытие; калиброванность не заявляется
        metrics["clean_targets"]["quantile_coverage_p10_p90"] = float(covered)
        metrics["clean_targets"]["quantile_band_rows"] = int(len(band))
    return metrics


def _replay_coverage(predictions: pd.DataFrame) -> dict:
    """Ожидаемые пары из протокола, независимо от наличия погоды и факта SCADA."""
    first, last = _issue_range()
    hours = pd.date_range(EVAL_PERIOD[0], f"{EVAL_PERIOD[1]} 23:00", freq="h")
    expected = {
        (t, issue.isoformat(), hour.strftime("%Y-%m-%d %H:%M:%S"), lead)
        for hour in hours for lead in (1, 2) for t in TURBINES
        if first <= (issue := hour.date() - timedelta(days=lead)) <= last
    }
    keys = ["turbine", "issue_date", "datetime", "lead_day"]
    actual = set(predictions[keys].itertuples(index=False, name=None))
    if len(actual) != len(predictions):
        raise ValueError("дубли пар (турбина, выпуск, целевой час, горизонт) в оценке")
    if actual - expected:
        raise ValueError("оценочные строки вне периодов или горизонтов протокола")
    return {
        "expected_rows": len(expected), "actual_rows": len(actual),
        "missing_rows": len(expected - actual),
        "excluded_before_model_cutoff": len(hours) * 2 * len(TURBINES) - len(expected),
        "by_lead": {str(lead): {
            "expected_rows": sum(key[3] == lead for key in expected),
            "actual_rows": sum(key[3] == lead for key in actual),
        } for lead in (1, 2)},
    }


def run_evaluation(output_dir: Path | str, weather: pd.DataFrame | None = None) -> dict:
    import lightgbm
    import sklearn

    from src.config import TEST_END, TIMEZONE, TRAIN_START, WEATHER_POINT
    from src.models.train import (EARLY_STOPPING_ROUNDS, LGB_PARAMS,
                                  QUANTILE_ALPHAS, SEEDS)
    from src.weather.openmeteo import get_weather

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if weather is None:
        weather = get_weather(TRAIN_START, TEST_END)

    print("Посадка оцениваемой конфигурации (отсечка обучения 2025-11-30)...")
    selections = _fit_evaluation_models(weather, output)

    print("Воспроизведение по датам выпуска...")
    predictions, skipped = _replay(weather, output)
    coverage = _replay_coverage(predictions)
    csv_path = output / CSV_NAME
    predictions.to_csv(csv_path, index=False)

    metrics = compute_metrics(csv_path)

    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "periods": {"fit": list(FIT_PERIOD), "tune": list(TUNE_PERIOD),
                    "evaluate": list(EVAL_PERIOD)},
        "trained_through": TRAINED_THROUGH,
        "aggregation_rule": ("нет усреднения: каждая пара (выпуск, целевой час) — "
                             "отдельная строка; метрики группируются по lead_day"),
        "issue_range": [day.isoformat() for day in _issue_range()],
        "issue_time_assumption": "конец дня; задержки публикации SCADA/NWP не моделируются",
        "coverage": coverage,
        "config": {
            "weather_models": WEATHER_MODELS,
            "weather_point": list(WEATHER_POINT), "timezone": TIMEZONE,
            "weather_columns": list(weather.columns),
            "feature_builder": "build_features(get_issued_forecast(issue_date, weather))",
            "turbines": list(TURBINES),
            "lgb_params": LGB_PARAMS,
            "ensemble_seeds": list(SEEDS),
            "quantile_alphas": list(QUANTILE_ALPHAS),
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "selection": selections,
        },
        "model_artifacts": {str(t): f"turbine_{t}.pkl" for t in TURBINES},
        "skipped_issues": skipped,
        "metrics": metrics,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__,
            "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__,
        },
        "duration_s": round(time.time() - started, 1),
    }
    (output / REPORT_NAME).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    for t in TURBINES:
        clean = metrics["clean_targets"].get(f"turbine_{t}", {})
        line = " | ".join(f"{k}: MAE {v['mae']:.4f} (n={v['n']})"
                          for k, v in clean.items() if isinstance(v, dict))
        print(f"T{t}  {line}")
    print(f"Готово за {report['duration_s']} с -> {csv_path} и {output / REPORT_NAME}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="каталог результатов; никакие другие пути не изменяются")
    args = parser.parse_args()
    run_evaluation(args.output_dir)


if __name__ == "__main__":
    main()
