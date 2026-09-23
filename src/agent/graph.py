"""Общее состояние, переходы и журнал для LLM и детерминированного запуска дня."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from src.agent import tools as T
from src.config import FORECASTS

NODES: dict[str, tuple[str, Callable]] = {
    "fetch_weather": ("Получение архивного прогноза погоды", T.fetch_weather),
    "prepare_features": ("Подготовка признаков", T.prepare_features),
    "run_model": ("Запуск моделей обеих турбин", T.run_model),
    "validate_forecast": ("Физическая валидация прогноза", T.validate_forecast),
    "recover_baseline": ("Резервный прогноз по кривой мощности", T.recover_baseline),
    "compare_with_previous": ("Сравнение с прошлым запуском", T.compare_with_previous),
}
EDGES = {
    "fetch_weather": "prepare_features",
    "prepare_features": "run_model",
    "run_model": "validate_forecast",
    "recover_baseline": "validate_forecast",
    "compare_with_previous": "write_outputs",
}
ENTRY = "fetch_weather"
TERMINAL = "write_outputs"


class DayGraph:
    """Один контекст на весь день, включая достройку после сбоя LLM.

    Модель получает next_tool после каждого вызова; недопустимый переход не меняет
    контекст. Ошибка расчёта завершает день с трассой и исключением для CLI.
    """

    def __init__(self, issue_date: str, weather, mode: str = "no-llm", *, output_dir: Path | None = None):
        self.ctx = T.DayContext(issue_date, weather,
                               output_dir=output_dir if output_dir is not None else FORECASTS)
        self.mode = mode
        self.node: str | None = ENTRY
        self.outputs: dict[str, dict] = {}
        self.trace: list[dict] = []
        self.retried = False
        self.fallback = False
        self.completed = False
        self.error: str | None = None

    def result(self) -> dict:
        return {
            "issue_date": self.ctx.issue_date, "mode": self.mode,
            "completed": self.completed, "next_tool": self.node,
            "validation_ok": bool((self.ctx.validation or {}).get("ok")),
            "retried": self.retried, "fallback": self.fallback,
            "error": self.error, "trace": self.trace,
            **self.outputs.get(TERMINAL, {}),
        }

    def _record(self, node: str, label: str, status: str, output: dict, ms: float = 0) -> None:
        self.trace.append({"node": node, "label": label, "status": status,
                           "ms": ms, "output": output, "attempt": 2 if self.retried else 1,
                           "next_tool": self.node})
        try:
            self.ctx.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.ctx.output_dir / f"trace_{self.ctx.issue_date}.json"
            # Замена файла не даёт панели прочитать недописанный JSON между шагами.
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(self.result(), ensure_ascii=False, indent=1), encoding="utf-8")
            temp.replace(path)
        except Exception as exc:
            self.error = f"Сбой записи трассы: {exc}"
            self.completed, self.node = False, None
            raise RuntimeError(self.error) from exc

    def execute(self, name: str, args: dict) -> dict:
        if self.error:
            raise RuntimeError(self.error)
        if self.completed or name != self.node:
            out = {"error": f"Недопустимый переход: {name}", "next_tool": self.node}
            self._record("__rejected__", "Отклонён вызов вне порядка графа", "rejected",
                         {"requested_tool": name, **out})
            return out
        if not isinstance(args, dict) or (name == TERMINAL and
                (not isinstance(args.get("analysis"), str) or not args["analysis"].strip())):
            out = {"error": "write_outputs требует непустой analysis; аргументы — объект",
                   "next_tool": self.node}
            self._record("__rejected__", "Отклонены аргументы инструмента", "rejected", out)
            return out

        label = "Запись прогноза и отчёта" if name == TERMINAL else NODES[name][0]
        started = time.perf_counter()
        try:
            if name == TERMINAL:
                analysis = args["analysis"]
                if self.retried:
                    analysis += ("\n\nРезервный режим: после неудачной валидации использована "
                                 "физическая кривая мощности (baseline), повторная валидация пройдена. "
                                 "Интервалы ансамбля к резервному прогнозу не применяются.")
                if self.fallback:
                    analysis += "\n\nПосле сбоя LLM граф завершён детерминированно с сохранённого шага."
                result = T.write_outputs(self.ctx, analysis)
                self.completed, self.node = True, None
            else:
                result = NODES[name][1](self.ctx)
                if name == "validate_forecast":
                    if result["ok"]:
                        self.node = "compare_with_previous"
                    elif not self.retried:
                        self.node = "recover_baseline"
                    else:
                        self.error = "Повторная валидация не пройдена: прогноз не опубликован"
                        self.node = None
                else:
                    if name == "recover_baseline":
                        self.retried = True
                    self.node = EDGES[name]
            self.outputs[name] = result
        except Exception as exc:
            self.error = f"Сбой узла {name}: {exc}"
            self.node = None
            self._record(name, label, "error", {"error": self.error},
                         round((time.perf_counter() - started) * 1000, 1))
            raise RuntimeError(self.error) from exc

        status = "invalid" if name == "validate_forecast" and not result["ok"] else "ok"
        self._record(name, label, status, result, round((time.perf_counter() - started) * 1000, 1))
        if self.error:
            raise RuntimeError(self.error)
        return {**result, "next_tool": self.node}

    def use_fallback(self, reason: str) -> None:
        self.fallback = True
        self._record("__fallback__", "Продолжение без LLM", "fallback",
                     {"reason": reason, "resume_at": self.node})

    def finish(self, analysis_fn: Callable[[dict, T.DayContext], str]) -> dict:
        """Продолжить с текущего узла; уже выполненные инструменты не вызываются снова."""
        if self.error:
            raise RuntimeError(self.error)
        while not self.completed:
            args = {"analysis": analysis_fn(self.outputs, self.ctx)} if self.node == TERMINAL else {}
            self.execute(self.node, args)
        return self.result()


def run_graph(issue_date: str, weather, analysis_fn: Callable[[dict, T.DayContext], str],
              *, output_dir: Path | None = None) -> dict:
    return DayGraph(issue_date, weather, output_dir=output_dir).finish(analysis_fn)


def graph_topology() -> dict:
    """Узлы и переходы для панели; неуспешная повторная валидация завершает день."""
    nodes = [{"id": k, "label": v[0]} for k, v in NODES.items()]
    nodes.append({"id": TERMINAL, "label": "Запись прогноза и отчёта"})
    edges = [{"from": a, "to": b, "kind": "always"} for a, b in EDGES.items()]
    edges += [{"from": "validate_forecast", "to": "compare_with_previous", "kind": "успех"},
              {"from": "validate_forecast", "to": "recover_baseline", "kind": "одна попытка восстановления"}]
    return {"nodes": nodes, "edges": edges, "entry": ENTRY, "terminal": TERMINAL}
