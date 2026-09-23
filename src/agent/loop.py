"""LLM вызывает инструменты общего графа и пишет отчёт; без ключа граф идёт по шаблону."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.agent import tools as T

SYSTEM = """Ты — операционный агент прогнозирования выработки ветроэлектростанции (2 турбины,
Шелекский коридор, Казахстан). Твоя задача на каждый день D: выполнить полный цикл прогноза
на следующие 48 часов из архивного прогноза Previous Runs: lead 1 для D+1, lead 2 для D+2.
Входы содержат прогнозную погоду, а не наблюдения. Время публикации в архиве отсутствует:
не утверждай, что все значения были доступны к конкретному часу дня D.

Порядок: fetch_weather -> prepare_features -> run_model -> validate_forecast ->
compare_with_previous -> write_outputs. Каждый ответ инструмента содержит next_tool:
вызывай именно его, строго один инструмент за ответ. Дождись результата перед следующим
вызовом и составляй отчёт только по полученным данным.
При неудачной валидации граф разрешает один recover_baseline, затем
повторную validate_forecast. Это замена ансамбля физической кривой мощности на той же
архивной погоде, а не получение свежего выпуска. Объясни причину в отчёте. Если резерв
также не проходит проверки, граф останавливается без публикации прогноза.
significant_update=true означает отличие от вчерашнего прогноза на общие часы;
оно само по себе не доказывает изменение погодных данных.

В финале вызови write_outputs, передав analysis — краткий отчёт по-русски: погодная ситуация,
ожидаемая выработка обеих турбин, качество входных данных, отличия от вчерашнего запуска,
замечания. Пиши как оператор ВЭС, конкретно и без воды."""

TOOL_DEFS = [
    {"name": "fetch_weather", "description": "Получить срез архивного прогноза Previous Runs для даты запуска",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "prepare_features", "description": "Построить матрицу фич из полученного прогноза",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_model", "description": "Прогнать модели обеих турбин (LightGBM + физический baseline)",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "validate_forecast", "description": "Проверить прогноз: границы 0..1, полнота 48ч, аномалии",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "recover_baseline", "description": "После провала валидации один раз применить физический baseline; затем нужна повторная проверка",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "compare_with_previous", "description": "Сравнить прогноз мощности с предыдущим выпуском на общие часы",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "write_outputs", "description": "Записать CSV прогнозов и markdown-отчёт. Завершает день.",
     "input_schema": {"type": "object", "properties": {
         "analysis": {"type": "string", "description": "Отчёт оператора по-русски"}},
         "required": ["analysis"]}},
]


def _template_analysis(outputs: dict, _ctx: T.DayContext) -> str:
    """Отчёт детерминированного режима — из выходов узлов графа."""
    w = outputs.get("fetch_weather", {})
    f = outputs.get("prepare_features", {})
    m = outputs.get("recover_baseline", {}).get("forecast", outputs.get("run_model", {}))
    v = outputs.get("validate_forecast", {})
    c = outputs.get("compare_with_previous", {})
    return (
        f"Режим без LLM (детерминированный обход графа).\n\n"
        f"- Погода: средний ветер 100м {w.get('wind100_mean_ms')} м/с, "
        f"макс {w.get('wind100_max_ms')} м/с, часов без данных: "
        f"{w.get('hours_without_any_model')}.\n"
        f"- Признаки: {f.get('rows')} часов x {f.get('n_features')}.\n"
        f"- Итоговый прогноз: {json.dumps(m, ensure_ascii=False)}.\n"
        f"- Валидация: {'OK' if v.get('ok') else 'ПРОБЛЕМЫ: ' + json.dumps(v.get('checks', {}), ensure_ascii=False)}.\n"
        f"- Сравнение с прошлым запуском: {json.dumps(c, ensure_ascii=False)}.\n"
    )


def run_day_no_llm(issue_date: str, weather: pd.DataFrame, *, output_dir: Path | None = None) -> dict:
    """Детерминированный обход графа состояний (для судей без ключей)."""
    from src.agent.graph import run_graph
    return run_graph(issue_date, weather, _template_analysis, output_dir=output_dir)


def run_day_llm(issue_date: str, weather: pd.DataFrame, *, output_dir: Path | None = None) -> dict:
    """Полноценный агент: OpenAI, NVIDIA NIM или Claude через типизированные тулы.

    Бэкенд выбирается по ключам окружения (src/agent/llm.py). Если агент не довёл цикл
    до write_outputs — день достраивается детерминированно, прогноз не теряется."""
    from src.agent.llm import anthropic_chat_loop, openai_chat_loop, pick_backend
    from src.agent.graph import DayGraph

    backend = pick_backend()
    if backend == "none":
        return run_day_no_llm(issue_date, weather, output_dir=output_dir)
    model = os.environ.get("LLM_MODEL", "по умолчанию")
    print(f"  [{issue_date}] LLM: {backend} / {model}")

    graph = DayGraph(issue_date, weather, mode=backend, output_dir=output_dir)
    user_msg = f"День запуска: {issue_date}. Выполни полный цикл прогноза на 48 часов."
    loop_fn = anthropic_chat_loop if backend == "anthropic" else openai_chat_loop
    try:
        loop_fn(SYSTEM, TOOL_DEFS, user_msg, graph.execute)
    except Exception as exc:
        if graph.error:
            raise  # Ошибка расчёта не исправляется сменой LLM на шаблон.
        if not graph.completed:
            graph.use_fallback(f"{type(exc).__name__}: {exc}")
    if not graph.completed and not graph.fallback:
        graph.use_fallback("LLM завершил ответ или исчерпал шаги до write_outputs")
    if graph.fallback:
        print(f"  [{issue_date}] Продолжаю без LLM с узла {graph.node}")
    return graph.finish(_template_analysis)
