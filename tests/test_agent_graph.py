"""Сценарии дня: реальные проверки и файлы, подменены только погода, ML и LLM."""
import json

import numpy as np
import pandas as pd
import pytest

from src.agent import graph, llm, loop, tools
from src import cli


@pytest.fixture
def day(monkeypatch, tmp_path):
    weather = pd.DataFrame({
        "best_match__wind_speed_100m": np.linspace(3, 10, 48),
        "lead_day": [1] * 24 + [2] * 24,
    }, index=pd.date_range("2026-02-11", periods=48, freq="h", name="time"))
    calls = []
    monkeypatch.setattr(graph, "FORECASTS", tmp_path)
    monkeypatch.setattr(tools, "FORECASTS", tmp_path)
    monkeypatch.setattr(tools, "get_issued_forecast", lambda issue, archive: weather.copy())
    monkeypatch.setattr(tools, "build_features", lambda frame: frame.copy())

    def predict(turbine, features):
        calls.append(turbine)
        return pd.DataFrame({
            "power_pred": np.linspace(.1, .8, 48),
            "power_baseline": np.linspace(.2, .7, 48),
            "power_lgb": np.linspace(.05, .85, 48),
            "power_p10": np.zeros(48), "power_p90": np.ones(48),
            "lead_day": features["lead_day"],
        }, index=features.index)

    monkeypatch.setattr(tools, "predict", predict)
    return weather, calls, tmp_path


def scripted_llm(system, definitions, user_msg, run_tool):
    name = "fetch_weather"
    while name:
        out = run_tool(name, {"analysis": "Проверенный прогноз"} if name == "write_outputs" else {})
        name = out.get("next_tool")
    return "ok"


@pytest.mark.parametrize("backend", ["none", "openai", "anthropic"])
def test_all_modes_persist_the_same_successful_graph(day, monkeypatch, backend):
    weather, calls, folder = day
    monkeypatch.setattr(llm, "pick_backend", lambda: backend)
    monkeypatch.setattr(llm, "openai_chat_loop", scripted_llm)
    monkeypatch.setattr(llm, "anthropic_chat_loop", scripted_llm)
    result = loop.run_day_llm("2026-02-10", weather)
    saved = json.loads((folder / "trace_2026-02-10.json").read_text())
    assert result["completed"] and result["validation_ok"]
    assert saved["completed"] and saved["mode"] == ("no-llm" if backend == "none" else backend)
    assert [s["node"] for s in saved["trace"]] == [
        "fetch_weather", "prepare_features", "run_model", "validate_forecast",
        "compare_with_previous", "write_outputs",
    ]
    assert saved["trace"] == result["trace"]
    assert calls == [1, 2]
    assert len(result["files"]) == 2


def test_llm_outage_resumes_current_state_without_repeating_model(day, monkeypatch):
    weather, calls, folder = day

    def interrupted(system, definitions, user_msg, run_tool):
        for name in ("fetch_weather", "prepare_features", "run_model"):
            run_tool(name, {})
        assert (folder / "trace_2026-02-10.json").exists()
        raise TimeoutError("тестовый таймаут")

    monkeypatch.setattr(llm, "pick_backend", lambda: "openai")
    monkeypatch.setattr(llm, "openai_chat_loop", interrupted)
    result = loop.run_day_llm("2026-02-10", weather)
    assert calls == [1, 2]
    assert result["completed"] and result["fallback"]
    fallback = next(s for s in result["trace"] if s["node"] == "__fallback__")
    assert fallback["output"]["resume_at"] == "validate_forecast"
    assert "детерминирован" in (folder / result["report"]).read_text()


def test_early_write_is_rejected_and_llm_can_continue(day, monkeypatch):
    weather, _, folder = day

    def early_write(system, definitions, user_msg, run_tool):
        rejected = run_tool("write_outputs", {"analysis": "Слишком рано"})
        assert rejected["error"] and rejected["next_tool"] == "fetch_weather"
        assert not list(folder.glob("forecast_*.csv"))
        return scripted_llm(system, definitions, user_msg, run_tool)

    monkeypatch.setattr(llm, "pick_backend", lambda: "openai")
    monkeypatch.setattr(llm, "openai_chat_loop", early_write)
    result = loop.run_day_llm("2026-02-10", weather)
    assert result["completed"]
    assert result["trace"][0]["status"] == "rejected"


@pytest.mark.parametrize("baseline_ok", [True, False])
@pytest.mark.parametrize("backend", ["none", "openai"])
def test_failed_validation_recovers_once_or_stops_without_publishing(day, monkeypatch, baseline_ok, backend):
    weather, calls, folder = day
    original_predict = tools.predict

    def invalid_prediction(turbine, features):
        frame = original_predict(turbine, features)
        frame["power_pred"] = np.nan
        if not baseline_ok:
            frame["power_baseline"] = np.nan
        return frame

    monkeypatch.setattr(tools, "predict", invalid_prediction)
    monkeypatch.setattr(llm, "pick_backend", lambda: backend)
    monkeypatch.setattr(llm, "openai_chat_loop", scripted_llm)
    if baseline_ok:
        result = loop.run_day_llm("2026-02-10", weather)
        assert result["retried"] and result["validation_ok"]
        for name in result["files"]:
            frame = pd.read_csv(folder / name)
            np.testing.assert_allclose(frame["power_pred"], frame["power_baseline"])
            assert "power_p10" not in frame and "power_p90" not in frame
        assert "baseline" in (folder / result["report"]).read_text()
    else:
        with pytest.raises(RuntimeError, match="валидац"):
            loop.run_day_llm("2026-02-10", weather)
        assert not list(folder.glob("forecast_*.csv"))
        assert not list(folder.glob("report_*.md"))
    saved = json.loads((folder / "trace_2026-02-10.json").read_text())
    nodes = [s["node"] for s in saved["trace"]]
    assert nodes.count("fetch_weather") == 1
    assert nodes.count("recover_baseline") == 1
    assert nodes.count("validate_forecast") == 2
    assert saved["completed"] == baseline_ok
    assert calls == [1, 2]


@pytest.mark.parametrize("power", [0.0, 0.97, 1.0])
@pytest.mark.parametrize("needs_recovery", [False, True])
def test_flat_forecast_publishes_with_warning(day, monkeypatch, power, needs_recovery):
    weather, _, folder = day
    original_predict = tools.predict

    def constant_prediction(turbine, features):
        frame = original_predict(turbine, features)
        frame["power_pred"] = np.nan if needs_recovery else power
        frame["power_baseline"] = power
        return frame

    monkeypatch.setattr(tools, "predict", constant_prediction)
    result = loop.run_day_no_llm("2026-02-10", weather)
    assert result["completed"] and result["validation_ok"]
    assert result["retried"] == needs_recovery
    validations = [step["output"] for step in result["trace"]
                   if step["node"] == "validate_forecast"]
    assert all(check["flatline"] for check in validations[-1]["checks"].values())
    assert len(validations[-1]["warnings"]) == 2
    assert "почти постоян" in (folder / result["report"]).read_text()
    for name in result["files"]:
        frame = pd.read_csv(folder / name)
        np.testing.assert_allclose(frame["power_pred"], power)


@pytest.mark.parametrize("power", [-0.01, 1.01])
def test_flat_forecast_outside_bounds_still_stops_without_publishing(day, monkeypatch, power):
    weather, _, folder = day
    original_predict = tools.predict

    def corrupt_prediction(turbine, features):
        frame = original_predict(turbine, features)
        frame["power_pred"] = power
        frame["power_baseline"] = power
        return frame

    monkeypatch.setattr(tools, "predict", corrupt_prediction)
    with pytest.raises(RuntimeError, match="валидац"):
        loop.run_day_no_llm("2026-02-10", weather)
    assert not list(folder.glob("forecast_*.csv"))


def test_tool_failure_saves_error_trace_without_writing_partial_output(day, monkeypatch):
    weather, _, folder = day

    def broken_features(frame):
        raise ValueError("повреждённые входные данные")

    monkeypatch.setattr(tools, "build_features", broken_features)
    with pytest.raises(RuntimeError, match="prepare_features"):
        loop.run_day_no_llm("2026-02-10", weather)
    saved = json.loads((folder / "trace_2026-02-10.json").read_text())
    assert saved["trace"][-1]["status"] == "error"
    assert not saved["completed"]
    assert not list(folder.glob("forecast_*.csv"))


def test_failed_rerun_cannot_reuse_old_csvs_in_submission(day, monkeypatch):
    weather, _, folder = day
    loop.run_day_no_llm("2026-02-10", weather)

    def broken_features(frame):
        raise ValueError("сбой повторного запуска")

    monkeypatch.setattr(tools, "build_features", broken_features)
    with pytest.raises(RuntimeError):
        loop.run_day_no_llm("2026-02-10", weather)
    monkeypatch.setattr(cli, "FORECASTS", folder)
    with pytest.raises(ValueError, match="2026-02-10"):
        cli._build_submission()
    assert not (folder / "submission.csv").exists()


def test_llm_cannot_swallow_final_trace_write_failure(day, monkeypatch):
    from pathlib import Path

    weather, _, _ = day
    original_replace = Path.replace

    def broken_replace(path, target):
        if path.name.endswith(".json.tmp") and json.loads(path.read_text())["completed"]:
            raise OSError("диск недоступен")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", broken_replace)
    monkeypatch.setattr(llm, "pick_backend", lambda: "openai")
    monkeypatch.setattr(llm, "openai_chat_loop", scripted_llm)
    with pytest.raises(RuntimeError, match="трассы"):
        loop.run_day_llm("2026-02-10", weather)


def test_failed_issue_outside_submission_period_does_not_block_build(day, monkeypatch):
    weather, _, folder = day
    loop.run_day_no_llm("2026-02-10", weather)
    (folder / "trace_2026-02-28.json").write_text(json.dumps({
        "issue_date": "2026-02-28", "completed": False,
    }))
    monkeypatch.setattr(cli, "FORECASTS", folder)
    cli._build_submission()
    assert (folder / "submission.csv").exists()
