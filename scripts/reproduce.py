"""Одна команда репетиции релиза: сохранённые модели и кэш → 28 выпусков → проверки.

    python -m scripts.reproduce --output-dir runs/reproduction

Что происходит (offline, без обучения, без LLM, без изменения файлов команды):

1. Каталог результатов проверяется: не защищённый каталог репозитория (в том числе
   через symlink) и не непустой каталог. Рекурсивной очистки нет — только пустой или новый.
2. SHA256 канонических входов и результатов снимается до и после прогона и сверяется
   с `models_artifacts/manifest.json`: команда обязана ничего не изменить.
3. Сеть блокируется в процессе, а полнота погодного кэша проверяется заранее:
   недостающий кэш — ошибка, а не незаметная докачка.
4. 28 выпусков 31.01–27.02.2026 выполняются сохранёнными моделями в режиме без LLM.
5. Подача проверяется верификатором, считаются 1344 turbine-hours, числа сравниваются
   с каноническими `forecasts/` с явным допуском.
6. Метрики оценки пересчитываются из сохранённого CSV и сверяются с
   `evaluation_report.json`; оценочные прогнозы воспроизводятся сохранёнными
   оценочными моделями (переобучения нет).
7. Сводный `reproduction_report.json` пишется только в output-dir. Любая проблема —
   ненулевой код выхода.

Совпадение чисел подтверждает воспроизводимость сохранённого комплекта, а не точность
февраля и не доступность погоды к часу выпуска (docs/FEATURE_AVAILABILITY.md).

Допуск `--tolerance` относится только к сравнению прогнозов с каноническими файлами.
На той же платформе они совпадают побитово; на другой ОС тригонометрические признаки
отличаются на 1 ulp (libm), и часть порогов LightGBM переключается — измеренные
отклонения и лечение описаны в docs/REPRODUCIBILITY.md. Отчёт всегда содержит число
строк, совпавших точнее 1e-9, чтобы ослабленный допуск не скрывал масштаб различий.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from src.config import (ARTIFACTS, DATA_RAW, FORECASTS, ROOT, TEST_END, TRAIN_START,
                        TURBINES, WEATHER_CACHE, WEATHER_MODELS)

EVALUATION_DIR = ARTIFACTS / "evaluation"
MANIFEST = ARTIFACTS / "manifest.json"
REPORT_NAME = "reproduction_report.json"

DEFAULT_START = "2026-01-31"   # первый выпуск: закрывает 1 февраля
DEFAULT_END = "2026-02-27"     # последний выпуск: 24 часа на 28 февраля, край архива
DEFAULT_TOLERANCE = 1e-9       # абсолютный допуск сравнения прогнозов с каноническими
EXACT_TOLERANCE = 1e-9         # порог «побитового» совпадения для диагностики в отчёте
METRIC_TOLERANCE = 1e-9        # метрики пересчитываются из того же CSV: платформа не влияет

# Каталоги, в которые репетиция писать не имеет права; сравниваются по realpath.
PROTECTED_DIRS = ("data", "models_artifacts", "forecasts", "src", "tests", "scripts",
                  "docs", ".git", ".github")
FORECAST_VALUE_COLUMNS = ("power_pred", "power_baseline", "power_lgb", "power_p10",
                          "power_p90", "lead_day", "horizon_h")
EVALUATION_VALUE_COLUMNS = ("power_pred", "power_baseline", "power_p10", "power_p90")


class ReproductionError(Exception):
    """Проблема, из-за которой репетиция считается неудачной."""


# ------------------------------------------------------------------ каталог результатов

def _realpath(path: Path) -> Path:
    # resolve() раскрывает symlink и в существующей части пути, и в родителях ещё не
    # созданного каталога, поэтому ссылка внутрь forecasts/ тоже будет отклонена
    return Path(path).expanduser().resolve()


def protected_roots() -> list[Path]:
    return [_realpath(ROOT)] + [_realpath(ROOT / name) for name in PROTECTED_DIRS]


def prepare_output_dir(path: Path) -> Path:
    """Вернуть готовый пустой каталог или отказаться: защищённый, файл, непустой."""
    target = _realpath(path)
    roots = protected_roots()
    repo = roots[0]
    if target == repo:
        raise ReproductionError(f"каталог результатов совпадает с корнем репозитория: {target}")
    for root in roots[1:]:
        if target == root or root in target.parents:
            raise ReproductionError(f"каталог результатов {target} лежит в защищённом {root}; "
                                    "используйте, например, runs/reproduction")
    if target.exists():
        if not target.is_dir():
            raise ReproductionError(f"{target} существует и не является каталогом")
        if any(target.iterdir()):
            raise ReproductionError(f"{target} не пуст; рекурсивная очистка не выполняется — "
                                    "укажите новый или пустой каталог")
    else:
        target.mkdir(parents=True)
    return target


# ------------------------------------------------------------------ неизменность файлов

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_dirs() -> list[Path]:
    return [DATA_RAW, WEATHER_CACHE, ARTIFACTS, FORECASTS]


def snapshot_canonical() -> dict[str, str]:
    """SHA256 каждого канонического файла по пути относительно корня репозитория."""
    hashes = {}
    for directory in canonical_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            # служебные файлы ОС и байткод не входят в комплект и не сравниваются
            if not path.is_file() or path.name.startswith(".") or "__pycache__" in path.parts:
                continue
            hashes[path.relative_to(ROOT).as_posix()] = _sha256(path)
    return hashes


def diff_snapshots(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {
        "changed": sorted(k for k in before if k in after and before[k] != after[k]),
        "removed": sorted(k for k in before if k not in after),
        "added": sorted(k for k in after if k not in before),
    }


def manifest_mismatches(snapshot: dict[str, str], manifest_path: Path) -> dict[str, list[str]]:
    """Расхождения текущих файлов с input_sha256/output_sha256 манифеста интегратора.

    Хэши исходников не сверяются: репетиция сама вправе менять код, а комплект данных,
    моделей и прогнозов должен совпадать с тем, который интегратор проверил.
    """
    manifest = json.loads(manifest_path.read_text())
    expected = {**manifest.get("input_sha256", {}), **manifest.get("output_sha256", {})}
    return {
        "missing": sorted(k for k in expected if k not in snapshot),
        "changed": sorted(k for k in expected if k in snapshot and snapshot[k] != expected[k]),
        "unlisted": sorted(k for k in snapshot if k not in expected),
    }


# ------------------------------------------------------------------ offline

def install_network_guard() -> None:
    """Любое сетевое соединение процесса — исключение; кэш должен быть полным заранее."""
    def refuse(*_args, **_kwargs):
        raise RuntimeError("репетиция выполняется без сети: сетевой запрос запрещён")

    import httpx
    httpx.get = refuse
    httpx.post = refuse
    httpx.request = refuse
    httpx.Client.send = refuse
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse


def missing_weather_cache() -> list[str]:
    """Файлы кэша, которые понадобятся get_weather для полного диапазона."""
    from src.weather.openmeteo import _cache_path, _chunks
    return [
        _cache_path(model, start, end).name
        for model in WEATHER_MODELS
        for start, end in _chunks(TRAIN_START, TEST_END)
        if not _cache_path(model, start, end).is_file()
    ]


# ------------------------------------------------------------------ прогон и сравнения

def load_weather():
    from src.weather.openmeteo import get_weather
    return get_weather(TRAIN_START, TEST_END)


def run_forecasts(output_dir: Path, weather, start: date, end: date) -> list[dict]:
    """28 выпусков без LLM сохранёнными моделями; сводная подача — в output_dir."""
    from src import cli
    from src.agent.loop import run_day_no_llm

    days = []
    issue = start
    while issue <= end:
        result = run_day_no_llm(issue.isoformat(), weather, output_dir=output_dir)
        days.append({
            "issue_date": issue.isoformat(),
            "completed": bool(result.get("completed")),
            "validation_ok": bool(result.get("validation_ok")),
            "retried": bool(result.get("retried")),
            "fallback": bool(result.get("fallback")),
            "error": result.get("error"),
        })
        print(f"[{issue}] completed={days[-1]['completed']} validation_ok={days[-1]['validation_ok']}")
        issue += timedelta(days=1)
    cli._build_submission(output_dir=output_dir)
    return days


def _read_csv(path: Path):
    import pandas as pd
    if not path.is_file():
        raise ReproductionError(f"нет файла {path}")
    return pd.read_csv(path)


def _abs_diff(a, b):
    import numpy as np
    x = a.to_numpy(dtype=float)
    y = b.to_numpy(dtype=float)
    both_nan = np.isnan(x) & np.isnan(y)
    diff = np.abs(x - y)
    diff[both_nan] = 0.0
    # NaN только с одной стороны — это расхождение, а не «нечего сравнивать»
    diff[np.isnan(diff)] = math.inf
    return diff


def _max_abs_diff(a, b) -> float:
    diff = _abs_diff(a, b)
    return float(diff.max()) if len(diff) else 0.0


def _inexact_rows(a, b) -> int:
    """Сколько строк отличаются сильнее порога побитового совпадения."""
    return int((_abs_diff(a, b) > EXACT_TOLERANCE).sum())


def compare_forecast_dirs(run_dir: Path, canonical_dir: Path, start: date, end: date) -> dict:
    """Максимальное отклонение подачи и дневных CSV прогона от канонических файлов."""
    problems, max_dev, inexact = [], {}, None

    canonical = _read_csv(canonical_dir / "submission.csv")
    produced = _read_csv(run_dir / "submission.csv")
    keys = ["turbine", "datetime"]
    if len(canonical) != len(produced) or not canonical[keys].equals(produced[keys]):
        merged = canonical.merge(produced, on=keys, how="outer", indicator=True)
        problems.append("submission.csv: набор turbine/datetime отличается от канонического "
                        f"({int((merged['_merge'] != 'both').sum())} несовпадающих строк)")
    else:
        max_dev["submission.csv:power_pred"] = _max_abs_diff(canonical["power_pred"], produced["power_pred"])
        inexact = _inexact_rows(canonical["power_pred"], produced["power_pred"])
        if not canonical["lead_day"].equals(produced["lead_day"]):
            problems.append("submission.csv: lead_day отличается от канонического")

    issue = start
    while issue <= end:
        for turbine in TURBINES:
            name = f"forecast_t{turbine}_{issue.isoformat()}.csv"
            if not (canonical_dir / name).is_file():
                problems.append(f"{name}: нет канонического файла для сравнения")
            elif not (run_dir / name).is_file():
                problems.append(f"{name}: прогон не создал файл")
            else:
                a, b = _read_csv(canonical_dir / name), _read_csv(run_dir / name)
                if list(a["datetime"]) != list(b["datetime"]):
                    problems.append(f"{name}: набор часов отличается от канонического")
                    continue
                for column in FORECAST_VALUE_COLUMNS:
                    if column in a.columns and column in b.columns:
                        dev = _max_abs_diff(a[column], b[column])
                        max_dev[f"daily:{column}"] = max(max_dev.get(f"daily:{column}", 0.0), dev)
                    elif (column in a.columns) != (column in b.columns):
                        problems.append(f"{name}: колонка {column} есть только с одной стороны")
        issue += timedelta(days=1)
    return {"problems": problems, "max_abs_deviation": max_dev,
            "submission_rows": int(len(canonical)),
            "submission_rows_beyond_exact": inexact}


def _flatten(tree: dict, prefix: str = "") -> dict[str, object]:
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def compare_metric_trees(saved: dict, recomputed: dict, tolerance: float) -> dict:
    """Целые — точно, вещественные — по абсолютному допуску; отсутствие ключа — ошибка."""
    a, b = _flatten(saved), _flatten(recomputed)
    problems, max_dev = [], 0.0
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            problems.append(f"метрика {key} есть только {'в JSON' if key in a else 'в пересчёте'}")
            continue
        x, y = a[key], b[key]
        exact = (isinstance(x, bool) or isinstance(y, bool)
                 or (isinstance(x, int) and isinstance(y, int)))
        if exact:
            if x != y:
                problems.append(f"метрика {key}: {x} != {y}")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            dev = abs(float(x) - float(y))
            max_dev = max(max_dev, dev)
            if not dev <= tolerance:  # NaN и inf тоже считаются расхождением
                problems.append(f"метрика {key}: {x} vs {y} (|Δ|={dev:.3e} > {tolerance:g})")
        elif x != y:
            problems.append(f"метрика {key}: {x!r} != {y!r}")
    return {"problems": problems, "max_abs_deviation": max_dev}


def check_evaluation_metrics(evaluation_dir: Path, tolerance: float = METRIC_TOLERANCE) -> dict:
    from src.backtest.evaluate import CSV_NAME, REPORT_NAME as EVAL_REPORT, compute_metrics
    report_path = evaluation_dir / EVAL_REPORT
    csv_path = evaluation_dir / CSV_NAME
    if not report_path.is_file() or not csv_path.is_file():
        raise ReproductionError(f"нет сохранённой оценки в {evaluation_dir}")
    report = json.loads(report_path.read_text())
    recomputed = compute_metrics(csv_path)
    result = compare_metric_trees(report["metrics"], recomputed, tolerance)
    rows = len(_read_csv(csv_path))
    coverage = report.get("coverage", {})
    if coverage.get("actual_rows") != rows:
        result["problems"].append(f"coverage.actual_rows={coverage.get('actual_rows')} "
                                  f"при {rows} строках CSV")
    result["rows"] = rows
    return result


def replay_evaluation(evaluation_dir: Path, weather, tolerance: float) -> dict:
    """Сохранённые оценочные модели заново предсказывают оценочный период; обучения нет."""
    from src.backtest.evaluate import CSV_NAME, _replay
    saved = _read_csv(evaluation_dir / CSV_NAME)
    produced, skipped = _replay(weather, evaluation_dir)
    keys = ["turbine", "issue_date", "datetime", "lead_day"]
    problems, max_dev = [], {}
    if skipped:
        problems.append(f"пропущенные выпуски при воспроизведении: {skipped}")
    merged = saved.merge(produced, on=keys, how="outer", suffixes=("_saved", "_new"),
                         indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        problems.append(f"оценочные строки не совпали по ключам: {unmatched}")
    matched = merged[merged["_merge"] == "both"]
    for column in EVALUATION_VALUE_COLUMNS:
        max_dev[column] = _max_abs_diff(matched[f"{column}_saved"], matched[f"{column}_new"])
    return {"problems": problems, "max_abs_deviation": max_dev,
            "rows_saved": int(len(saved)), "rows_replayed": int(len(produced)),
            "rows_beyond_exact": _inexact_rows(matched["power_pred_saved"], matched["power_pred_new"])}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None  # внутри образа .git отсутствует по .dockerignore


def _versions() -> dict:
    import lightgbm
    import numpy
    import pandas
    import sklearn
    return {"python": platform.python_version(), "pandas": pandas.__version__,
            "numpy": numpy.__version__, "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__}


# ------------------------------------------------------------------ сценарий

def reproduce(output_dir: Path, *, start: date, end: date, tolerance: float,
              skip_evaluation_replay: bool = False) -> dict:
    """Выполнить репетицию и вернуть отчёт; отчёт записывается в output_dir."""
    started = time.time()
    report: dict = {
        "schema_version": 1,
        "command": "python -m scripts.reproduce",
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "versions": _versions(),
        "issue_range": [start.isoformat(), end.isoformat()],
        "tolerance": tolerance,
        "network": "заблокирована в процессе (httpx, socket)",
        "checks": {},
        "durations_s": {},
        "ok": False,
    }
    checks = report["checks"]
    durations = report["durations_s"]
    failures: list[str] = []

    def record(name: str, ok: bool, details: dict) -> None:
        checks[name] = {"ok": ok, **details}
        if not ok:
            failures.append(name)

    t0 = time.time()
    before = snapshot_canonical()
    durations["snapshot_before"] = round(time.time() - t0, 1)
    if MANIFEST.is_file():
        mism = manifest_mismatches(before, MANIFEST)
        record("manifest", not mism["missing"] and not mism["changed"],
               {"files": len(before), **mism})
    else:
        record("manifest", False, {"error": f"нет {MANIFEST.relative_to(ROOT)}"})

    install_network_guard()
    missing = missing_weather_cache()
    record("weather_cache", not missing, {"missing": missing})

    if missing:
        # без полного кэша прогон означал бы докачку — она запрещена
        return _finish(report, failures, output_dir, started)

    t0 = time.time()
    weather = load_weather()
    durations["load_weather"] = round(time.time() - t0, 1)

    t0 = time.time()
    days = run_forecasts(output_dir, weather, start, end)
    durations["forecast_runs"] = round(time.time() - t0, 1)
    bad = [d["issue_date"] for d in days if not d["completed"] or d["error"]]
    expected_days = (end - start).days + 1
    record("forecast_runs", not bad and len(days) == expected_days,
           {"issues": len(days), "expected_issues": expected_days, "failed_issues": bad,
            "fallback_or_recovered": [d["issue_date"] for d in days
                                      if d["fallback"] or d["retried"]]})

    from scripts.verify_submission import verify_forecasts
    period_start, period_end = start + timedelta(days=1), end + timedelta(days=1)
    errors = verify_forecasts(output_dir, period_start.isoformat(), period_end.isoformat())
    expected_rows = len(TURBINES) * ((period_end - period_start).days + 1) * 24
    submission = output_dir / "submission.csv"
    rows = len(_read_csv(submission)) if submission.is_file() else 0
    record("submission", not errors and rows == expected_rows,
           {"rows": rows, "expected_rows": expected_rows, "errors": errors[:20]})

    try:
        comparison = compare_forecast_dirs(output_dir, FORECASTS, start, end)
        worst = max(comparison["max_abs_deviation"].values(), default=0.0)
        record("forecast_match", not comparison["problems"] and worst <= tolerance,
               {**comparison, "worst_abs_deviation": worst})
    except ReproductionError as exc:
        record("forecast_match", False, {"error": str(exc)})

    try:
        metrics = check_evaluation_metrics(EVALUATION_DIR)
        record("evaluation_metrics", not metrics["problems"], {**metrics, "tolerance": METRIC_TOLERANCE})
    except (ReproductionError, ValueError, KeyError) as exc:
        record("evaluation_metrics", False, {"error": str(exc)})

    if not skip_evaluation_replay:
        t0 = time.time()
        try:
            replay = replay_evaluation(EVALUATION_DIR, weather, tolerance)
            worst = max(replay["max_abs_deviation"].values(), default=0.0)
            record("evaluation_replay", not replay["problems"] and worst <= tolerance,
                   {**replay, "worst_abs_deviation": worst})
        except (ReproductionError, FileNotFoundError) as exc:
            record("evaluation_replay", False, {"error": str(exc)})
        durations["evaluation_replay"] = round(time.time() - t0, 1)

    t0 = time.time()
    after = snapshot_canonical()
    durations["snapshot_after"] = round(time.time() - t0, 1)
    changes = diff_snapshots(before, after)
    record("canonical_unchanged", not any(changes.values()), changes)

    return _finish(report, failures, output_dir, started)


def _finish(report: dict, failures: list[str], output_dir: Path, started: float) -> dict:
    report["ok"] = not failures
    report["failed_checks"] = failures
    report["duration_s"] = round(time.time() - started, 1)
    (output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def print_summary(report: dict, output_dir: Path) -> None:
    for name, check in report["checks"].items():
        status = "OK  " if check["ok"] else "FAIL"
        extra = ""
        if "worst_abs_deviation" in check:
            extra = f" (max |Δ| = {check['worst_abs_deviation']:.2e}"
            beyond = check.get("submission_rows_beyond_exact", check.get("rows_beyond_exact"))
            total = check.get("submission_rows", check.get("rows_saved"))
            if beyond is not None and total:
                extra += f", строк точнее {EXACT_TOLERANCE:g}: {total - beyond}/{total}"
            extra += ")"
        elif "max_abs_deviation" in check and isinstance(check["max_abs_deviation"], float):
            extra = f" (max |Δ| = {check['max_abs_deviation']:.2e})"
        elif "rows" in check:
            extra = f" (строк: {check['rows']})"
        print(f"{status} {name}{extra}")
        for problem in (check.get("problems") or check.get("errors") or [])[:5]:
            print(f"      - {problem}")
        if check.get("error"):
            print(f"      - {check['error']}")
        for key in ("missing", "changed", "removed", "added", "failed_issues"):
            if check.get(key):
                print(f"      - {key}: {check[key][:5]}")
    print(f"{'ГОТОВО' if report['ok'] else 'ОШИБКА'}: {report['duration_s']} с, "
          f"отчёт {output_dir / REPORT_NAME}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="пустой или новый каталог вне data/, models_artifacts/, forecasts/")
    parser.add_argument("--start", default=DEFAULT_START, help="первая дата выпуска")
    parser.add_argument("--end", default=DEFAULT_END, help="последняя дата выпуска")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="абсолютный допуск сравнения прогнозов с каноническими; метрики "
                             "оценки всегда сверяются с допуском %g" % METRIC_TOLERANCE)
    parser.add_argument("--skip-evaluation-replay", action="store_true",
                        help="не воспроизводить оценочные прогнозы сохранёнными моделями")
    args = parser.parse_args(argv)

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if start > end:
        print("Дата --start должна быть не позже --end")
        return 2
    try:
        output_dir = prepare_output_dir(args.output_dir)
    except ReproductionError as exc:
        print(f"ОТКАЗ: {exc}")
        return 2

    report = reproduce(output_dir, start=start, end=end, tolerance=args.tolerance,
                       skip_evaluation_replay=args.skip_evaluation_replay)
    print_summary(report, output_dir)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
