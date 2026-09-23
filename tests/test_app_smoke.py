"""Панель оператора: оба режима исполняются и показывают то, что заявляют.

Streamlit AppTest прогоняет скрипт целиком в том же процессе, поэтому ловит
ошибки, которые HTTP-ответ сервера не показывает: рендеринг идёт на клиенте,
и упавший скрипт всё равно вернул бы 200. Режим оценки проверяется по существу:
показанная MAE обязана совпадать с расчётом по известному маленькому CSV.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("streamlit")

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

TIMEOUT = 300
APP = Path(__file__).resolve().parent.parent / "app.py"

MODE_FEB = "Тестовый период (февраль 2026)"
MODE_EVAL = "Ретроспективная оценка (есть факт)"


def _run(mode: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception, f"скрипт упал на старте: {at.exception}"
    if mode is not None:
        at.radio(key="mode").set_value(mode).run()
        assert not at.exception, f"режим «{mode}» упал: {at.exception}"
    return at


# ------------------------------------------------------------ февральский режим

def test_test_period_mode_renders():
    at = _run(MODE_FEB)
    assert at.metric, "нет плиток со сводкой дня"


def test_failed_rerun_hides_previous_forecast(tmp_path, monkeypatch):
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


# --------------------------------------------------------------- режим оценки

def test_evaluation_mode_renders_both_turbines():
    at = _run(MODE_EVAL)
    for turbine in (1, 2):
        at.selectbox(key="turbine").select(turbine).run()
        assert not at.exception, f"турбина {turbine} упала: {at.exception}"
        labels = [m.label for m in at.metric]
        assert any(l.startswith("MAE дня") for l in labels), \
            f"нет плитки MAE дня; плитки: {labels}"
        assert "Дата выпуска" in labels, "не показана дата выпуска"
        assert "Отсечка обучения" in labels, "не показана отсечка обучения"


def test_evaluation_lead_switch_and_missing_first_lead2_day():
    at = _run(MODE_EVAL)
    # lead 2 существует на обычную дату
    at.radio(key="eval_lead").set_value(2).run()
    assert not at.exception
    assert any(m.label == "Дата выпуска" for m in at.metric), "lead 2 не отрисовался"
    # первый день оценки не имеет lead-2 строк: понятное объяснение, не падение
    at.date_input(key="eval_date").set_value(pd.Timestamp("2025-12-01").date()).run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "lead 2" in warnings and "отсечк" in warnings, \
        f"нет объяснения отсутствия первого lead-2 дня: {warnings!r}"
    assert not any(m.label == "Дата выпуска" for m in at.metric)


def test_evaluation_missing_dir_shows_error_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("WINDCAST_EVALUATION_DIR", str(tmp_path / "nowhere"))
    at = _run(MODE_EVAL)
    errors = " ".join(e.value for e in at.error)
    assert "evaluation_predictions.csv" in errors, f"нет понятной ошибки: {errors!r}"
    # тихого отката к финальной модели нет: страница остановлена без метрик
    assert not at.metric


def test_evaluation_mae_matches_known_csv(tmp_path, monkeypatch):
    """Показанная MAE дня равна расчёту руками по маленькому известному CSV."""
    hours = pd.date_range("2026-01-10", periods=4, freq="h")
    df = pd.DataFrame({
        "turbine": 1, "issue_date": "2026-01-09",
        "datetime": hours.astype(str), "lead_day": 1,
        "power_true": [0.5, 0.3, 0.9, 0.4],
        "power_pred": [0.4, 0.3, 0.7, 0.9],
        "power_baseline": [0.5, 0.5, 0.5, 0.5],
        "power_p10": [0.1] * 4, "power_p90": [0.95] * 4,
        # последняя строка не «чистая цель» и в MAE чистых целей не входит
        "target_eligible": [True, True, True, False],
        "trained_through": "2025-11-30",
    })
    df.to_csv(tmp_path / "evaluation_predictions.csv", index=False)
    (tmp_path / "evaluation_report.json").write_text(json.dumps({
        "schema_version": 1, "protocol": "retrospective-issue-replay-v1",
        "trained_through": "2025-11-30",
        "periods": {"evaluate": ["2025-12-01", "2026-01-31"]},
    }))
    monkeypatch.setenv("WINDCAST_EVALUATION_DIR", str(tmp_path))

    at = _run(MODE_EVAL)
    at.date_input(key="eval_date").set_value(pd.Timestamp("2026-01-10").date()).run()
    assert not at.exception

    shown = {m.label: m.value for m in at.metric}
    mae_label = next(l for l in shown if l.startswith("MAE дня"))
    # чистые цели: |0.4-0.5|, |0.3-0.3|, |0.7-0.9| -> (0.1 + 0.0 + 0.2) / 3
    assert shown[mae_label] == f"{(0.1 + 0.0 + 0.2) / 3:.3f}"
    assert shown["Дата выпуска"] == "2026-01-09"
    assert shown["Часов с фактом"] == "3 из 4"

    # переключение на все наблюдаемые часы меняет и маску, и число
    at.radio(key="eval_scope").set_value("Все наблюдаемые часы").run()
    shown = {m.label: m.value for m in at.metric}
    mae_label = next(l for l in shown if l.startswith("MAE дня"))
    assert shown[mae_label] == f"{(0.1 + 0.0 + 0.2 + 0.5) / 4:.3f}"
