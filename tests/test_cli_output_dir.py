"""Отдельный прогон из CLI не должен менять канонические результаты команды."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import cli
from src.agent import graph, llm, tools


@pytest.fixture
def isolated_run(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "report_2026-02-10.md").write_text("Сохранённый отчёт LLM")
    (canonical / "submission.csv").write_text("сохранённая подача")
    weather = pd.DataFrame({
        "best_match__wind_speed_100m__d1": np.linspace(3, 10, 72),
        "best_match__wind_speed_100m__d2": np.linspace(4, 11, 72),
    }, index=pd.date_range("2026-02-11", periods=72, freq="h", name="time"))
    monkeypatch.setattr(cli, "_load_weather", lambda: weather)
    for module in (cli, graph, tools):
        monkeypatch.setattr(module, "FORECASTS", canonical)
    monkeypatch.setattr(tools, "build_features", lambda frame: frame)
    monkeypatch.setattr(llm, "pick_backend", lambda: "none")

    def predict(turbine, features):
        power = features.index.hour / 30 + .05
        return pd.DataFrame({"power_pred": power, "power_baseline": power,
                             "power_lgb": power, "lead_day": features["lead_day"]},
                            index=features.index)

    monkeypatch.setattr(tools, "predict", predict)
    return canonical, tmp_path / "separate" / "run"


@pytest.mark.parametrize("backend", ["none", "openai"])
def test_cli_writes_all_outputs_and_previous_comparison_to_selected_directory(isolated_run, monkeypatch, backend):
    canonical, output = isolated_run
    before = {p.name: p.read_bytes() for p in canonical.iterdir()}
    monkeypatch.setattr(llm, "pick_backend", lambda: backend)

    def scripted_llm(system, definitions, user_msg, run_tool):
        name = "fetch_weather"
        while name:
            result = run_tool(name, {"analysis": "Отчёт LLM"} if name == "write_outputs" else {})
            name = result["next_tool"]

    monkeypatch.setattr(llm, "openai_chat_loop", scripted_llm)
    argv = ["windcast", "run-agent", "--start", "2026-02-10", "--end", "2026-02-11",
            "--output-dir", str(output)]
    if backend == "none":
        argv.append("--no-llm")
    monkeypatch.setattr("sys.argv", argv)
    cli.main()

    assert {p.name: p.read_bytes() for p in canonical.iterdir()} == before
    assert len(list(output.glob("forecast_t*.csv"))) == 4
    assert len(list(output.glob("report_*.md"))) == 2
    assert len(list(output.glob("trace_*.json"))) == 2
    assert len(pd.read_csv(output / "submission.csv")) == 144
    trace = json.loads((output / "trace_2026-02-11.json").read_text())
    assert trace["completed"]
    comparison = next(step for step in trace["trace"] if step["node"] == "compare_with_previous")
    assert comparison["output"]["turbine_1"]["hours_overlap"] == 24


def test_no_key_fallback_does_not_bypass_existing_report_protection(isolated_run, monkeypatch):
    canonical, _ = isolated_run
    monkeypatch.setattr("sys.argv", ["windcast", "run-agent", "--start", "2026-02-10",
                                     "--end", "2026-02-10"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert (canonical / "report_2026-02-10.md").read_text() == "Сохранённый отчёт LLM"


def test_overwrite_protection_applies_to_selected_directory(isolated_run):
    _, output = isolated_run
    output.mkdir(parents=True)
    (output / "report_2026-02-10.md").write_text("Отчёт LLM в другом каталоге")
    start = date(2026, 2, 10)
    assert not cli._guard_downgrade(start, start, True, False, output_dir=output)
    assert cli._guard_downgrade(start, start, True, True, output_dir=output)


def test_failed_trace_check_applies_to_selected_directory(isolated_run):
    _, output = isolated_run
    output.mkdir(parents=True)
    (output / "trace_2026-02-10.json").write_text(json.dumps({
        "issue_date": "2026-02-10", "completed": False,
    }))
    with pytest.raises(ValueError, match="2026-02-10"):
        cli._build_submission(output_dir=output)
