"""Панель оператора должна исполняться без исключений в обоих режимах.

Streamlit AppTest прогоняет скрипт целиком в том же процессе, поэтому ловит
ошибки, которые HTTP-ответ сервера не показывает: рендеринг идёт на клиенте,
и упавший скрипт всё равно вернул бы 200.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

TIMEOUT = 300
APP = Path(__file__).resolve().parent.parent / "app.py"


def _run(mode_index: int) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception, f"скрипт упал на старте: {at.exception}"
    at.radio[0].set_value(at.radio[0].options[mode_index]).run()
    return at


def test_test_period_mode_renders():
    at = _run(0)
    assert not at.exception, f"режим тестового периода упал: {at.exception}"
    assert at.metric, "нет плиток со сводкой дня"


def test_holdout_mode_renders():
    at = _run(1)
    assert not at.exception, f"режим holdout упал: {at.exception}"
    labels = [m.label for m in at.metric]
    assert "MAE этого дня" in labels, ("в режиме holdout факт известен, "
                                       f"MAE должен считаться; плитки: {labels}")


def test_turbine_switch_keeps_app_alive():
    at = _run(0)
    at.selectbox[0].select(2).run()
    assert not at.exception, f"переключение турбины упало: {at.exception}"


def test_failed_rerun_hides_previous_forecast(tmp_path, monkeypatch):
    import json
    from src import config

    monkeypatch.setattr(config, "FORECASTS", tmp_path)
    (tmp_path / "forecast_t1_2026-02-10.csv").write_text("старый файл не должен читаться")
    (tmp_path / "trace_2026-02-10.json").write_text(json.dumps({
        "issue_date": "2026-02-10", "completed": False, "trace": [],
    }))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("недоступен" in error.value for error in at.error)
    assert not at.metric


def test_dashboard_can_read_an_isolated_run(tmp_path, monkeypatch):
    import shutil
    from src import config

    for path in config.FORECASTS.glob("*2026-02-10.*"):
        shutil.copy2(path, tmp_path / path.name)
    monkeypatch.setattr(config, "FORECASTS", tmp_path / "absent-default")
    monkeypatch.setattr("sys.argv", ["app.py", "--forecast-dir", str(tmp_path)])
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.metric, "панель не прочитала прогноз из выбранного каталога"
    assert at.select_slider[0].value == "2026-02-10"
