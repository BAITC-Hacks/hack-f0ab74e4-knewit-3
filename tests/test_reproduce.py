"""Репетиция релиза должна падать на значимых отказах, а не только на исключениях.

Тяжёлые шаги (погода, модели, 28 выпусков) подменены синтетикой того же формата;
сами проверки — реальные функции scripts/reproduce.py. Успешный путь строится
из одного генератора для «канонических» и «воспроизведённых» файлов, а отказы —
из его точечных искажений.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import reproduce as rep

START, END = date(2026, 2, 1), date(2026, 2, 2)    # выпуски: 2 дня, подача 02.02–03.02


def _daily_frame(turbine: int, issue: date, last_day: date) -> pd.DataFrame:
    """48 часов выпуска; на краю архива, как у настоящего последнего выпуска, — 24."""
    hours = pd.date_range(issue + timedelta(days=1), periods=48, freq="h")
    hours = hours[hours.normalize() <= pd.Timestamp(last_day)]
    power = ((hours.hour + turbine) / 30).round(6)
    return pd.DataFrame({
        "turbine": turbine, "datetime": hours.strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_h": range(1, len(hours) + 1),
        "lead_day": [(h.date() - issue).days for h in hours],
        "power_pred": power, "power_baseline": power + .01, "power_lgb": power - .01,
        "power_p10": power - .05, "power_p90": power + .05,
    })


def write_forecast_set(directory: Path, start: date = START, end: date = END,
                       distort: dict | None = None) -> None:
    """Дневные CSV и подача в формате run-agent; distort — точечные искажения."""
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    issue = start
    while issue <= end:
        for turbine in (1, 2):
            frame = _daily_frame(turbine, issue, end + timedelta(days=1))
            if distort and distort.get("issue") == issue.isoformat() and distort.get("turbine") == turbine:
                if distort.get("drop_hour"):
                    frame = frame.iloc[1:]
                if distort.get("shift"):
                    frame["power_pred"] = frame["power_pred"] + distort["shift"]
            frame.to_csv(directory / f"forecast_t{turbine}_{issue.isoformat()}.csv", index=False)
            frames.append(frame)
        issue += timedelta(days=1)
    allp = pd.concat(frames)
    allp = (allp.sort_values(["turbine", "datetime", "lead_day"])
                .groupby(["turbine", "datetime"], as_index=False).first())
    first_day = (start + timedelta(days=1)).isoformat()
    last_day = (end + timedelta(days=1)).isoformat()
    allp = allp[(allp["datetime"] >= first_day) & (allp["datetime"] <= f"{last_day} 23:59")]
    allp[["turbine", "datetime", "lead_day", "power_pred"]].to_csv(directory / "submission.csv", index=False)


def write_evaluation_set(directory: Path, tamper_metric: bool = False) -> None:
    from src.backtest.evaluate import CSV_NAME, REPORT_NAME, compute_metrics
    directory.mkdir(parents=True, exist_ok=True)
    hours = pd.date_range("2025-12-02", periods=8, freq="h")
    frame = pd.DataFrame({
        "turbine": [1, 1, 1, 1, 2, 2, 2, 2], "issue_date": ["2025-12-01"] * 8,
        "datetime": hours.strftime("%Y-%m-%d %H:%M:%S"), "lead_day": [1, 1, 2, 2] * 2,
        "power_true": [.5, .3, .5, np.nan, .8, .2, .4, .6],
        "power_pred": [.4, .3, .7, .5, .6, .2, .5, .5], "power_baseline": [.5] * 8,
        "power_p10": [.1] * 8, "power_p90": [.9] * 8,
        "target_eligible": [True, True, False, False, True, True, True, True],
        "trained_through": ["2025-11-30"] * 8,
    })
    frame.to_csv(directory / CSV_NAME, index=False)
    metrics = compute_metrics(directory / CSV_NAME)
    if tamper_metric:
        metrics["clean_targets"]["turbine_1"]["lead_1"]["mae"] += 0.01
    report = {"metrics": metrics, "coverage": {"actual_rows": len(frame)}}
    (directory / REPORT_NAME).write_text(json.dumps(report))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Маленький «репозиторий»: канонические каталоги, манифест и подменённые шаги."""
    root = tmp_path / "repo"
    data_raw, cache = root / "data" / "raw", root / "data" / "weather_cache"
    artifacts, forecasts = root / "models_artifacts", root / "forecasts"
    for directory in (data_raw, cache, artifacts, forecasts):
        directory.mkdir(parents=True)
    (data_raw / "turbine_1.csv").write_text("id,power\n1,0.5\n")
    (cache / "best_match_x.json").write_text("{}")
    (artifacts / "turbine_1.pkl").write_bytes(b"model")
    write_forecast_set(forecasts)
    write_evaluation_set(artifacts / "evaluation")

    monkeypatch.setattr(rep, "ROOT", root)
    monkeypatch.setattr(rep, "DATA_RAW", data_raw)
    monkeypatch.setattr(rep, "WEATHER_CACHE", cache)
    monkeypatch.setattr(rep, "ARTIFACTS", artifacts)
    monkeypatch.setattr(rep, "FORECASTS", forecasts)
    monkeypatch.setattr(rep, "EVALUATION_DIR", artifacts / "evaluation")
    monkeypatch.setattr(rep, "MANIFEST", artifacts / "manifest.json")
    snapshot = rep.snapshot_canonical()
    manifest = {"input_sha256": {k: v for k, v in snapshot.items() if k.startswith("data/")},
                "output_sha256": {k: v for k, v in snapshot.items()
                                  if k.startswith(("forecasts/", "models_artifacts/"))}}
    (artifacts / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(rep, "install_network_guard", lambda: None)
    monkeypatch.setattr(rep, "missing_weather_cache", lambda: [])
    monkeypatch.setattr(rep, "load_weather", lambda: "weather")
    monkeypatch.setattr(rep, "_git_commit", lambda: "test")
    monkeypatch.setattr(rep, "_versions", lambda: {"python": "test"})

    def run_forecasts(output_dir, weather, start, end):
        write_forecast_set(output_dir, start, end)
        return [{"issue_date": d.isoformat(), "completed": True, "validation_ok": True,
                 "retried": False, "fallback": False, "error": None}
                for d in (start + timedelta(days=i) for i in range((end - start).days + 1))]

    monkeypatch.setattr(rep, "run_forecasts", run_forecasts)
    monkeypatch.setattr(rep, "replay_evaluation",
                        lambda evaluation_dir, weather, tolerance:
                        {"problems": [], "max_abs_deviation": {"power_pred": 0.0},
                         "rows_saved": 8, "rows_replayed": 8})
    return root


def run(sandbox, output="runs/reproduction", **kw):
    argv = ["--output-dir", str(sandbox / output), "--start", START.isoformat(),
            "--end", END.isoformat()]
    return rep.main(argv), sandbox / output


# ------------------------------------------------------------------ успешный путь

def test_successful_reproduction_writes_report_only_to_output_dir(sandbox):
    before = rep.snapshot_canonical()
    code, out = run(sandbox)
    assert code == 0
    report = json.loads((out / rep.REPORT_NAME).read_text())
    assert report["ok"] and report["failed_checks"] == []
    assert set(report["checks"]) >= {"manifest", "weather_cache", "forecast_runs", "submission",
                                     "forecast_match", "evaluation_metrics", "evaluation_replay",
                                     "canonical_unchanged"}
    assert report["checks"]["submission"]["rows"] == 2 * 2 * 24
    assert report["checks"]["forecast_match"]["worst_abs_deviation"] == 0.0
    assert report["checks"]["forecast_match"]["submission_rows_beyond_exact"] == 0
    assert report["checks"]["evaluation_metrics"]["tolerance"] == rep.METRIC_TOLERANCE
    assert rep.snapshot_canonical() == before
    assert not list((sandbox / "forecasts").glob("reproduction_report.json"))


# ------------------------------------------------------------------ защищённый каталог

@pytest.mark.parametrize("relative", ["forecasts", "forecasts/sub", "models_artifacts/evaluation",
                                      "data/raw/x", "src/out", ".git/objects", "docs", ""])
def test_protected_output_dirs_are_refused(sandbox, relative):
    code, _ = run(sandbox, relative)
    assert code == 2
    assert not (sandbox / relative / rep.REPORT_NAME).exists()


def test_symlink_into_protected_dir_is_refused(sandbox):
    link = sandbox / "runs" / "link"
    link.parent.mkdir()
    link.symlink_to(sandbox / "forecasts")
    with pytest.raises(rep.ReproductionError, match="защищённом"):
        rep.prepare_output_dir(link)
    dangling = sandbox / "runs" / "dangling"
    dangling.symlink_to(sandbox / "models_artifacts" / "new")
    with pytest.raises(rep.ReproductionError, match="защищённом"):
        rep.prepare_output_dir(dangling)


def test_non_empty_output_dir_is_refused_not_cleaned(sandbox):
    out = sandbox / "runs" / "busy"
    out.mkdir(parents=True)
    (out / "keep.txt").write_text("чужой результат")
    code, _ = run(sandbox, "runs/busy")
    assert code == 2
    assert (out / "keep.txt").read_text() == "чужой результат"
    assert not (out / rep.REPORT_NAME).exists()


def test_output_dir_outside_repo_is_allowed(sandbox, tmp_path):
    code = rep.main(["--output-dir", str(tmp_path / "elsewhere" / "run"),
                     "--start", START.isoformat(), "--end", END.isoformat()])
    assert code == 0


# ------------------------------------------------------------------ значимые отказы

def test_missing_weather_cache_fails_before_any_run(sandbox, monkeypatch):
    monkeypatch.setattr(rep, "missing_weather_cache", lambda: ["best_match_2024-03-01.json"])
    monkeypatch.setattr(rep, "load_weather", lambda: pytest.fail("погода не должна загружаться"))
    code, out = run(sandbox)
    assert code == 1
    report = json.loads((out / rep.REPORT_NAME).read_text())
    assert report["failed_checks"] == ["weather_cache"]
    assert "forecast_runs" not in report["checks"]


def test_network_guard_blocks_httpx_and_sockets(monkeypatch):
    import socket
    import httpx
    # регистрируем исходные атрибуты, чтобы после теста сеть снова была доступна
    for target, name in ((httpx, "get"), (httpx, "post"), (httpx, "request"),
                         (httpx.Client, "send"), (socket.socket, "connect"),
                         (socket.socket, "connect_ex"), (socket, "create_connection")):
        monkeypatch.setattr(target, name, getattr(target, name))
    rep.install_network_guard()
    with pytest.raises(RuntimeError, match="без сети"):
        httpx.get("https://previous-runs-api.open-meteo.com/v1/forecast")
    with pytest.raises(RuntimeError, match="без сети"):
        socket.create_connection(("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="без сети"):
        socket.socket().connect(("127.0.0.1", 9))


def test_incomplete_submission_fails(sandbox, monkeypatch):
    def broken_run(output_dir, weather, start, end):
        write_forecast_set(output_dir, start, end,
                           distort={"issue": START.isoformat(), "turbine": 1, "drop_hour": True})
        return [{"issue_date": START.isoformat(), "completed": True, "validation_ok": True,
                 "retried": False, "fallback": False, "error": None}] * 2
    monkeypatch.setattr(rep, "run_forecasts", broken_run)
    code, out = run(sandbox)
    assert code == 1
    report = json.loads((out / rep.REPORT_NAME).read_text())
    assert "submission" in report["failed_checks"]
    assert any("expected 48 hourly rows" in e for e in report["checks"]["submission"]["errors"])


def test_incomplete_day_run_fails(sandbox, monkeypatch):
    def failing_run(output_dir, weather, start, end):
        write_forecast_set(output_dir, start, end)
        return [{"issue_date": START.isoformat(), "completed": False, "validation_ok": False,
                 "retried": True, "fallback": False, "error": "валидация не прошла"}]
    monkeypatch.setattr(rep, "run_forecasts", failing_run)
    code, out = run(sandbox)
    assert code == 1
    report = json.loads((out / rep.REPORT_NAME).read_text())
    assert "forecast_runs" in report["failed_checks"]
    assert report["checks"]["forecast_runs"]["failed_issues"] == [START.isoformat()]


def test_forecast_values_beyond_tolerance_fail_with_max_deviation(sandbox, monkeypatch):
    def drifted_run(output_dir, weather, start, end):
        write_forecast_set(output_dir, start, end,
                           distort={"issue": END.isoformat(), "turbine": 2, "shift": 3e-7})
        return [{"issue_date": d.isoformat(), "completed": True, "validation_ok": True,
                 "retried": False, "fallback": False, "error": None} for d in (START, END)]
    monkeypatch.setattr(rep, "run_forecasts", drifted_run)
    code, out = run(sandbox)
    assert code == 1
    report = json.loads((out / rep.REPORT_NAME).read_text())
    assert "forecast_match" in report["failed_checks"]
    worst = report["checks"]["forecast_match"]["worst_abs_deviation"]
    assert worst == pytest.approx(3e-7, rel=1e-6)
    # тот же прогон в рамках более широкого допуска проходит: допуск явный, не молчаливый,
    # а число строк за порогом точного совпадения остаётся в отчёте
    code2 = rep.main(["--output-dir", str(sandbox / "runs" / "wide"), "--start", START.isoformat(),
                      "--end", END.isoformat(), "--tolerance", "1e-6"])
    assert code2 == 0
    wide = json.loads((sandbox / "runs" / "wide" / rep.REPORT_NAME).read_text())
    assert wide["checks"]["forecast_match"]["submission_rows_beyond_exact"] == 24
    assert wide["checks"]["forecast_match"]["submission_rows"] == 96


def test_metric_mismatch_fails_even_with_wide_forecast_tolerance(sandbox):
    write_evaluation_set(sandbox / "models_artifacts" / "evaluation", tamper_metric=True)
    code = rep.main(["--output-dir", str(sandbox / "runs" / "reproduction"), "--start",
                     START.isoformat(), "--end", END.isoformat(), "--tolerance", "0.5"])
    out = sandbox / "runs" / "reproduction"
    assert code == 1
    report = json.loads((out / rep.REPORT_NAME).read_text())
    assert "evaluation_metrics" in report["failed_checks"]
    problems = report["checks"]["evaluation_metrics"]["problems"]
    assert any("clean_targets.turbine_1.lead_1.mae" in p for p in problems)
    # манифест тоже заметил, что оценочный отчёт отличается от проверенного комплекта
    assert "manifest" in report["failed_checks"]


def test_changed_canonical_file_during_run_fails(sandbox, monkeypatch):
    original = rep.run_forecasts

    def tampering_run(output_dir, weather, start, end):
        (sandbox / "forecasts" / "submission.csv").write_text("turbine,datetime,lead_day,power_pred\n")
        return original(output_dir, weather, start, end)
    monkeypatch.setattr(rep, "run_forecasts", tampering_run)
    code, out = run(sandbox)
    assert code == 1
    report = json.loads((out / rep.REPORT_NAME).read_text())
    assert "canonical_unchanged" in report["failed_checks"]
    assert report["checks"]["canonical_unchanged"]["changed"] == ["forecasts/submission.csv"]


def test_manifest_mismatch_is_reported(sandbox):
    (sandbox / "models_artifacts" / "turbine_1.pkl").write_bytes(b"other model")
    mism = rep.manifest_mismatches(rep.snapshot_canonical(), sandbox / "models_artifacts" / "manifest.json")
    assert mism["changed"] == ["models_artifacts/turbine_1.pkl"]
    assert mism["unlisted"] == ["models_artifacts/manifest.json"]
    code, out = run(sandbox)
    assert code == 1
    assert json.loads((out / rep.REPORT_NAME).read_text())["failed_checks"] == ["manifest"]


def test_metric_tree_comparison_rules():
    saved = {"rows": {"total": 5}, "a": {"mae": 0.5, "n": 3}, "flag": True}
    same = {"rows": {"total": 5}, "a": {"mae": 0.5 + 1e-12, "n": 3}, "flag": True}
    assert rep.compare_metric_trees(saved, same, 1e-9)["problems"] == []
    off = {"rows": {"total": 6}, "a": {"mae": 0.5, "n": 3}, "flag": True}
    assert any("rows.total" in p for p in rep.compare_metric_trees(saved, off, 1e-9)["problems"])
    partial = {"rows": {"total": 5}, "a": {"mae": 0.5}, "flag": True}
    assert any("a.n" in p for p in rep.compare_metric_trees(saved, partial, 1e-9)["problems"])
    nan = {"rows": {"total": 5}, "a": {"mae": float("nan"), "n": 3}, "flag": True}
    assert rep.compare_metric_trees(saved, nan, 1e-9)["problems"]


def test_replay_mismatch_is_detected(monkeypatch, tmp_path):
    from src.backtest import evaluate
    write_evaluation_set(tmp_path)
    saved = pd.read_csv(tmp_path / evaluate.CSV_NAME)
    drifted = saved.copy()
    drifted.loc[0, "power_pred"] += 1e-6
    monkeypatch.setattr(evaluate, "_replay", lambda weather, directory: (drifted, []))
    result = rep.replay_evaluation(tmp_path, "weather", 1e-9)
    assert result["max_abs_deviation"]["power_pred"] == pytest.approx(1e-6)
    assert result["rows_beyond_exact"] == 1 and result["rows_saved"] == 8
    monkeypatch.setattr(evaluate, "_replay", lambda weather, directory: (saved.iloc[1:], [{"issue_date": "x", "reason": "пусто"}]))
    result = rep.replay_evaluation(tmp_path, "weather", 1e-9)
    assert result["problems"] and result["rows_replayed"] == 7


def test_real_cache_check_lists_every_required_file(monkeypatch, tmp_path):
    """Без стубов: пустой каталог кэша даёт полный список недостающих файлов."""
    from src.config import WEATHER_MODELS
    from src.weather import openmeteo
    assert rep.missing_weather_cache() == []          # кэш репозитория полный
    monkeypatch.setattr(openmeteo, "WEATHER_CACHE", tmp_path / "empty")
    missing = rep.missing_weather_cache()
    assert len(missing) == len(WEATHER_MODELS) * len(openmeteo._chunks(rep.TRAIN_START, rep.TEST_END))
    assert all(name.endswith(".json") for name in missing)
    assert any(name.startswith("ukmo_global_deterministic_10km_") for name in missing)
