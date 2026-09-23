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
