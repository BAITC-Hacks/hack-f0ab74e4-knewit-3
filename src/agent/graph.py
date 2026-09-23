"""Явный граф состояний агента: узлы, переходы, условный повтор, трасса выполнения.

Граф объявлен данными, а не зашит в последовательность вызовов: исполнитель идёт по
рёбрам и записывает каждый переход. Отсюда три свойства, ради которых обычно берут
фреймворки вроде LangGraph, — но без внешней зависимости:

- цикл: провал валидации возвращает управление на получение погоды (одна попытка),
- наблюдаемость: трасса с временем, статусом и выходом каждого узла пишется в JSON,
- воспроизводимость: тот же граф исполняется и LLM-агентом, и режимом --no-llm.

Трасса `forecasts/trace_{issue_date}.json` — источник данных для панели наблюдения.
"""
from __future__ import annotations

import json
import time
from typing import Callable

from src.agent import tools as T
from src.config import FORECASTS

# Узел: имя -> (человекочитаемое описание, функция над контекстом)
NODES: dict[str, tuple[str, Callable]] = {
    "fetch_weather": ("Получение архивного прогноза погоды", T.fetch_weather),
    "prepare_features": ("Подготовка признаков", T.prepare_features),
    "run_model": ("Запуск моделей обеих турбин", T.run_model),
    "validate_forecast": ("Физическая валидация прогноза", T.validate_forecast),
    "compare_with_previous": ("Сравнение с прошлым запуском", T.compare_with_previous),
}

# Безусловные рёбра
EDGES: dict[str, str] = {
    "fetch_weather": "prepare_features",
    "prepare_features": "run_model",
    "run_model": "validate_forecast",
    "compare_with_previous": "write_outputs",
}

ENTRY = "fetch_weather"
TERMINAL = "write_outputs"


def _route_after_validation(result: dict, retried: bool) -> str:
    """Условное ребро: неудачная валидация один раз отправляет цикл на перезабор входа."""
    if result.get("ok"):
        return "compare_with_previous"
    return "compare_with_previous" if retried else "fetch_weather"


def run_graph(issue_date: str, weather, analysis_fn: Callable[[dict, T.DayContext], str]) -> dict:
    """Исполнить граф за один прогнозный день.

    analysis_fn(outputs, ctx) -> текст отчёта; именно сюда подключается LLM либо
    детерминированный шаблон. Возвращает результат дня с трассой.
    """
    ctx = T.DayContext(issue_date, weather)
    trace: list[dict] = []
    outputs: dict[str, dict] = {}
    node, retried, guard = ENTRY, False, 0

    while node != TERMINAL and guard < 16:
        guard += 1
        label, fn = NODES[node]
        started = time.perf_counter()
        try:
            result = fn(ctx)
            status = "ok"
        except Exception as exc:
            result = {"error": str(exc)}
            status = "error"
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        outputs[node] = result
        trace.append({"node": node, "label": label, "status": status,
                      "ms": elapsed_ms, "output": result, "attempt": 2 if retried else 1})

        if status == "error":
            break
        if node == "validate_forecast":
            nxt = _route_after_validation(result, retried)
            if nxt == ENTRY:
                retried = True
                trace.append({"node": "__retry__", "label": "Валидация не прошла — повтор цикла",
                              "status": "retry", "ms": 0.0, "output": {}, "attempt": 1})
            node = nxt
        else:
            node = EDGES[node]

    analysis = analysis_fn(outputs, ctx)
    written = T.write_outputs(ctx, analysis)
    trace.append({"node": "write_outputs", "label": "Запись прогноза и отчёта",
                  "status": "ok", "ms": 0.0, "output": written, "attempt": 1})

    FORECASTS.mkdir(exist_ok=True)
    (FORECASTS / f"trace_{issue_date}.json").write_text(
        json.dumps({"issue_date": issue_date, "retried": retried, "trace": trace},
                   ensure_ascii=False, indent=1))

    validation = outputs.get("validate_forecast", {})
    return {"issue_date": issue_date, "validation_ok": bool(validation.get("ok")),
            "retried": retried, "trace": trace, **written}


def graph_topology() -> dict:
    """Описание графа для отрисовки в панели наблюдения."""
    nodes = [{"id": k, "label": v[0]} for k, v in NODES.items()]
    nodes.append({"id": TERMINAL, "label": "Запись прогноза и отчёта"})
    edges = [{"from": a, "to": b, "kind": "always"} for a, b in EDGES.items()]
    edges += [{"from": "validate_forecast", "to": "compare_with_previous", "kind": "успех"},
              {"from": "validate_forecast", "to": "fetch_weather", "kind": "повтор при сбое"}]
    return {"nodes": nodes, "edges": edges, "entry": ENTRY, "terminal": TERMINAL}
