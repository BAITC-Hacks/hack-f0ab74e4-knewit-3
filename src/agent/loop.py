"""Агентный цикл: Claude оркестрирует прогнозный день через тулы.

Режимы:
- LLM-агент (нужен ANTHROPIC_API_KEY): Claude сам вызывает тулы, анализирует их выход,
  решает о пересчёте и пишет содержательный отчёт.
- --no-llm: тот же конвейер в штатном порядке с шаблонным отчётом — воспроизводимость
  без ключей (ADR-003).
"""
from __future__ import annotations

import json
import os

import pandas as pd

from src.agent import tools as T

SYSTEM = """Ты — операционный агент прогнозирования выработки ветроэлектростанции (2 турбины,
Шелекский коридор, Казахстан). Твоя задача на каждый день D: выполнить полный цикл прогноза
на следующие 48 часов, используя ТОЛЬКО архивный прогноз погоды, доступный в день D.

Порядок: fetch_weather -> prepare_features -> run_model -> validate_forecast ->
compare_with_previous -> write_outputs. Если валидация провалилась — попробуй понять причину
по выходам тулов и запусти цепочку повторно (не более одного повтора), затем честно опиши
проблему в отчёте. Если compare_with_previous показывает significant_update=true — отметь
в отчёте, что вход существенно обновился и прогноз пересчитан свежими данными (это штатно).

В финале вызови write_outputs, передав analysis — краткий отчёт по-русски: погодная ситуация,
ожидаемая выработка обеих турбин, качество входных данных, отличия от вчерашнего запуска,
замечания. Пиши как оператор ВЭС, конкретно и без воды."""

TOOL_DEFS = [
    {"name": "fetch_weather", "description": "Получить архивный прогноз погоды на 48ч, доступный в день запуска",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "prepare_features", "description": "Построить матрицу фич из полученного прогноза",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_model", "description": "Прогнать модели обеих турбин (LightGBM + физический baseline)",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "validate_forecast", "description": "Проверить прогноз: границы 0..1, полнота 48ч, аномалии",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "compare_with_previous", "description": "Сравнить с прогнозом предыдущего дня (дрейф входных данных)",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "write_outputs", "description": "Записать CSV прогнозов и markdown-отчёт. Завершает день.",
     "input_schema": {"type": "object", "properties": {
         "analysis": {"type": "string", "description": "Отчёт оператора по-русски"}},
         "required": ["analysis"]}},
]


def _run_tool(name: str, args: dict, ctx: T.DayContext) -> dict:
    fn = {"fetch_weather": T.fetch_weather, "prepare_features": T.prepare_features,
          "run_model": T.run_model, "validate_forecast": T.validate_forecast,
          "compare_with_previous": T.compare_with_previous}.get(name)
    if fn:
        return fn(ctx)
    if name == "write_outputs":
        return T.write_outputs(ctx, args.get("analysis", ""))
    return {"error": f"неизвестный тул {name}"}


def run_day_no_llm(issue_date: str, weather: pd.DataFrame) -> dict:
    """Детерминированный цикл без LLM (для судей без ключей)."""
    ctx = T.DayContext(issue_date, weather)
    w = T.fetch_weather(ctx)
    f = T.prepare_features(ctx)
    m = T.run_model(ctx)
    v = T.validate_forecast(ctx)
    c = T.compare_with_previous(ctx)
    analysis = (
        f"Режим без LLM (детерминированный конвейер).\n\n"
        f"- Погода: средний ветер 100м {w['wind100_mean_ms']} м/с, макс {w['wind100_max_ms']} м/с, "
        f"часов без данных: {w['hours_without_any_model']}.\n"
        f"- Фичи: {f['rows']} часов x {f['n_features']} признаков.\n"
        f"- Прогноз: {json.dumps(m, ensure_ascii=False)}.\n"
        f"- Валидация: {'OK' if v['ok'] else 'ПРОБЛЕМЫ: ' + json.dumps(v['checks'], ensure_ascii=False)}.\n"
        f"- Сравнение с прошлым запуском: {json.dumps(c, ensure_ascii=False)}.\n"
    )
    out = T.write_outputs(ctx, analysis)
    return {"issue_date": issue_date, "validation_ok": v["ok"], **out}


def run_day_llm(issue_date: str, weather: pd.DataFrame) -> dict:
    """Полноценный агент: Claude (Anthropic API) или OpenAI-совместимый LLM (NVIDIA NIM/OpenAI).

    Бэкенд выбирается по ключам окружения (src/agent/llm.py). Если агент не довёл цикл
    до write_outputs — день достраивается детерминированно, прогноз не теряется."""
    from src.agent.llm import anthropic_chat_loop, openai_chat_loop, pick_backend

    backend = pick_backend()
    if backend == "none":
        return run_day_no_llm(issue_date, weather)
    model = os.environ.get("LLM_MODEL", "по умолчанию")
    print(f"  [{issue_date}] LLM: {backend} / {model}")

    ctx = T.DayContext(issue_date, weather)
    result: dict | None = None

    def run_tool(name: str, args: dict) -> dict:
        nonlocal result
        out = _run_tool(name, args, ctx)
        if name == "write_outputs":
            result = {"issue_date": issue_date,
                      "validation_ok": bool(ctx.validation and ctx.validation["ok"]), **out}
        return out

    user_msg = f"День запуска: {issue_date}. Выполни полный цикл прогноза на 48 часов."
    loop_fn = anthropic_chat_loop if backend == "anthropic" else openai_chat_loop
    try:
        loop_fn(SYSTEM, TOOL_DEFS, user_msg, run_tool)
    except Exception as e:  # сеть/лимиты LLM не должны ронять прогон
        print(f"  [{issue_date}] LLM-бэкенд {backend} упал ({e}), достраиваю без LLM")
    if result is None:
        return run_day_no_llm(issue_date, weather)
    return result
