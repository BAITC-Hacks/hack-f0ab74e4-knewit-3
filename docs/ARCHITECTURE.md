# Архитектура: Agentic AI-система прогноза выработки ВЭС

## Общая схема

```
                        ┌──────────────────────────────────────┐
                        │  АГЕНТ-ОРКЕСТРАТОР (OpenAI API)       │
                        │  планирует цикл, вызывает тулы,       │
                        │  анализирует результат, решает о      │
                        │  повторном расчёте, пишет отчёт       │
                        └──────┬───────────────────────────────┘
                               │ tool calls
   ┌───────────────┬───────────┼──────────────┬────────────────┐
   ▼               ▼           ▼              ▼                ▼
fetch_weather  prepare_    run_model    validate_forecast  compare/write
(Open-Meteo    features    (LightGBM    (физические        (сравнение с
 Previous      (матрица    per turbine   границы 0..1,      предыдущим
 Runs API,     фич на      + baseline    полнота 48 ч,      запуском, дрейф
 кэш JSON)     48 ч)       power curve)  аномалии)          погоды, вывод)
```

Агент — цикл вызова типизированных тулов OpenAI GPT-6 Luna (или Claude/NVIDIA NIM) над
состоянием `DayContext`. Каждый тул — обычная Python-функция. Режим `--no-llm`
исполняет явный граф в `src/agent/graph.py` и пишет трассу переходов; LLM-режим
вызывает те же тулы через отдельный цикл в `src/agent/loop.py`. При ошибке
валидации граф повторяет расчёт один раз, а LLM-промпт разрешает один повтор.
Автоматической замены погодного источника нет: повтор читает тот же кэш.

## Модули и владельцы

| Путь | Что делает | Владелец |
| --- | --- | --- |
| `src/weather/` | клиенты Open-Meteo (previous-runs, historical-forecast, forecast), кэш, таймзоны | Dev A |
| `src/features/` | агрегация датасета к часу, чистка простоев, матрица фич | Dev A |
| `src/models/` | baseline power curve, LightGBM, обучение, сериализация, метрики | Dev B |
| `src/backtest/` | rolling-валидация на holdout, отчёт по метрикам | Dev B |
| `src/agent/` | оркестратор, определения тулов, промпты, цикл анализа | Dev C |
| `src/cli.py` | точки входа: `train`, `forecast --date`, `run-agent --start --end` | Dev C |
| `forecasts/` | выход: CSV на каждый день + submission.csv | генерируется |

**Границы жёсткие:** менять чужой модуль без согласования нельзя — трое работают параллельно
в одном репо. Интерфейсы между модулями зафиксированы ниже; менять интерфейс — только сообщив
команде в чат.

## Контракты между модулями (согласовано, менять — только всем вместе)

```python
# src/weather/openmeteo.py -> src/features
def get_weather(start: str, end: str, models=None) -> pd.DataFrame:
    """Архивные прогнозы previous-runs API за период, все источники и оба lead time.
    Индекс: datetime (Asia/Almaty). Колонки: {model}__{var}__d{1|2}, ветер в м/с.
    Фактическая погода сюда попасть НЕ может по построению. Кэш data/weather_cache/."""

def get_issued_forecast(issue_date: str, weather: pd.DataFrame) -> pd.DataFrame:
    """Срез «что было доступно в день issue_date»: часы D+1 (lead 1) + D+2 (lead 2).
    Колонки {model}__{var} + lead_day; на краю архива допустимы только 24 часа."""

# src/features -> src/models
def build_features(weather_slice: pd.DataFrame) -> pd.DataFrame:
    """108 фич из прогнозной погоды (RESEARCH.md §3). Детерминированно, без сети."""

def load_hourly(turbine: int) -> pd.DataFrame:
    """Почасовая агрегация data/raw: колонка target с NaN на неполных/простойных часах."""

# src/models -> src/agent
def predict(turbine: int, features: pd.DataFrame) -> pd.DataFrame:
    """Колонки: power_pred (клип 0..1), power_baseline, power_lgb, power_p10, power_p90.
    Артефакты (бэггинг 5 LightGBM + изотоническая кривая + квантили) из models_artifacts/."""

# src/agent — выход
# forecasts/forecast_t{N}_{issue_date}.csv: turbine, datetime, horizon_h, lead_day,
#   power_pred, power_baseline, power_lgb, power_p10, power_p90
# forecasts/report_{issue_date}.md: отчёт агента; forecasts/submission.csv — сводный
```

## Rolling-цикл теста (сценарий, который увидят судьи)

```
python -m src.cli run-agent --start 2026-01-31 --end 2026-02-28
```

Для каждого дня D агент: получает архивный прогноз на D+1..D+2 → строит фичи → прогоняет обе
модели → валидирует → сравнивает с прогнозом от D-1 и, если вход существенно обновился,
помечает пересчёт → пишет прогноз и отчёт. В конце склеивает `submission.csv`:
на каждый час берётся прогноз минимального lead time.

## Стек

Python 3.11+ · venv+pip · pandas · LightGBM · scikit-learn · httpx · anthropic.
Обучение на CPU за секунды — GPU/NVIDIA-кредиты не требуются (ADR-004).
Ключ `OPENAI_API_KEY` в `.env` (не коммитить); без ключа работает режим
`--no-llm` — тот же пайплайн детерминированным циклом, судьи могут проверить без ключа.
