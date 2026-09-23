# Архитектура: Agentic AI-система прогноза выработки ВЭС

## Общая схема

```
                        ┌──────────────────────────────────────┐
                        │   АГЕНТ-ОРКЕСТРАТОР (Claude API)      │
                        │  планирует цикл, вызывает тулы,       │
                        │  анализирует результат, решает о      │
                        │  повторном расчёте, пишет отчёт       │
                        └──────┬───────────────────────────────┘
                               │ tool calls
   ┌───────────────┬───────────┼──────────────┬────────────────┐
   ▼               ▼           ▼              ▼                ▼
fetch_weather  prepareـ    run_model    validate_forecast  analyze_report
(Open-Meteo    features    (LightGBM    (физические        (сравнение с
 Previous      (матрица    per turbine   границы 0..1,      предыдущим
 Runs API,     фич на      + baseline    полнота 48 ч,      запуском, дрейф
 кэш JSON)     48 ч)       power curve)  аномалии)          погоды, вывод)
```

Агент — это цикл ReAct на Anthropic API (Claude) с типизированными тулами. Каждый тул —
обычная Python-функция в `src/`, работающая и без агента (для тестов и отладки). Агент даёт
autonomy: сам находит и чинит проблемы (дыра в погоде → повторный запрос/фолбэк-модель;
прогноз вне границ → клип и предупреждение; свежий прогон погодной модели → пересчёт).

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
# src/weather -> src/features
def get_archived_forecast(lat: float, lon: float, issue_date: date,
                          horizon_h: int = 48) -> pd.DataFrame:
    """Прогноз, доступный утром issue_date, на следующие horizon_h часов.
    Индекс: datetime (Asia/Almaty, hourly). Колонки: wind_speed_10m/80m/100m/120m,
    wind_direction_80m, wind_gusts_10m, temperature_2m, surface_pressure, lead_time_h.
    Всё из previous-runs API — фактическая погода сюда попасть НЕ может."""

def get_training_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """То же самое поколонно, но за исторический период (historical-forecast-api)."""

# src/features -> src/models
def build_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Фичи из погоды (см. RESEARCH.md §3). Детерминированно, без обращений к сети."""

def load_hourly_target(turbine: int) -> pd.Series:
    """Почасовая нормализованная мощность из data/raw, с NaN на неполных/простойных часах."""

# src/models -> src/agent
def predict(turbine: int, features: pd.DataFrame) -> pd.DataFrame:
    """Колонки: power_pred (клип 0..1), power_baseline. Модель из models/artifacts/."""

# src/agent — выход
# forecasts/forecast_t{N}_{issue_date}.csv: turbine, datetime, horizon_h, power_pred
# forecasts/report_{issue_date}.md: анализ агента (что получил, что заметил, решения)
```

## Rolling-цикл теста (сценарий, который увидят судьи)

```
python -m src.cli run-agent --start 2026-01-31 --end 2026-02-28
```

Для каждого дня D агент: получает архивный прогноз на D+1..D+2 → строит фичи → прогоняет обе
модели → валидирует → пишет прогноз и отчёт → сравнивает с прогнозом от D-1 и, если вход
существенно обновился, помечает пересчёт. В конце склеивает `submission.csv` (на каждый час —
прогноз минимального lead time).

## Стек

Python 3.11+ · pandas · LightGBM · scikit-learn · httpx · anthropic · uv (окружение).
Обучение на CPU за секунды — GPU/NVIDIA-кредиты не требуются (ADR-004).
Ключ `ANTHROPIC_API_KEY` в `.env` (не коммитить); без ключа работает деградированный режим
`--no-llm` — тот же пайплайн детерминированным циклом, судьи могут проверить без ключа.
