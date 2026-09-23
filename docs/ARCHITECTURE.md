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

Агент вызывает типизированные инструменты OpenAI GPT-6 Luna (или Claude/NVIDIA NIM)
через общий `DayGraph` в `src/agent/graph.py`. Режим `--no-llm` использует тот же
исполнитель и `DayContext`, выбирая следующий узел без API. Все переходы, отклонённые
вызовы и достройка без LLM сохраняются в `trace_{дата}.json` выбранного каталога после каждого шага.
LLM получает `next_tool`; недопустимый порядок не меняет состояние и не пишет CSV.

После провала `validate_forecast` разрешён один переход `recover_baseline`:
`power_pred` заменяется уже рассчитанным `power_baseline`, интервалы ансамбля удаляются,
погода остаётся прежней. Затем выполняется повторная валидация. Если она не прошла,
граф останавливает день с ошибкой и трассой, без публикации нового прогноза.
Сбой API или исчерпание шагов LLM включает достройку с текущего узла того же графа;
ошибку вычислений нельзя скрыть повторным запуском всего дня.

Трасса содержит `mode`, `completed`, `validation_ok`, `retried`, `fallback`, `error`,
`next_tool` и события `trace` (`node`, `status`, `ms`, `output`, `attempt`, `next_tool`).
`retried` означает применение baseline, `fallback` — продолжение без LLM. Для старых
закоммиченных трасс часть полей отсутствует. Важны статусы `invalid` и `error`: они
не равны успешному исполнению. Старые CSV не удаляются при неудаче нового запуска,
но панель скрывает их, а сборка submission проверяет завершение нужных февральских выпусков.

## Каталог одного прогона

CLI принимает `--output-dir` (по умолчанию `forecasts/`). Значение передаётся явным
необязательным аргументом `output_dir` через `run_day_llm`/`run_day_no_llm`, `DayGraph`
и `DayContext`. Инструменты читают предыдущий выпуск и пишут CSV/отчёт в `ctx.output_dir`;
исполнитель сохраняет там трассу, CLI собирает там submission. Глобальные пути при
запуске не переназначаются. Старые вызовы без аргумента продолжают работать.

Панель выбирает тот же каталог через `streamlit run app.py -- --forecast-dir <каталог>`.
Модели и погодный кэш остаются входами из стандартных каталогов проекта. Флаг выбора
результатов не меняет ни конфигурацию обучения, ни используемые модели.

## Модули и владельцы

| Путь | Что делает | Владелец |
| --- | --- | --- |
| `src/weather/` | клиент Open-Meteo Previous Runs, кэш, таймзоны | Dev A |
| `src/features/` | агрегация датасета к часу, чистка простоев, матрица фич | Dev A |
| `src/models/` | baseline power curve, LightGBM, обучение, сериализация, метрики | Dev B |
| `src/backtest/` | rolling-валидация на holdout, отчёт по метрикам | Dev B |
| `src/agent/` | оркестратор, определения тулов, промпты, цикл анализа | Dev C |
| `src/cli.py` | точки входа: `train`, `run-agent --start --end`, `check-tz`, `list-models` | Dev C |
| `forecasts/` | выход: CSV на каждый день + submission.csv | генерируется |

Точные границы текущей параллельной работы — в [TASKS](TASKS.md). Изменения погодных
и модельных контрактов согласуются через handoff интегратору; текущие сигнатуры ниже
сохраняются для совместимости веток.

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
python -m src.cli run-agent --start 2026-01-31 --end 2026-02-27 --no-llm --output-dir runs/check
```

Для каждого дня D агент: получает архивный прогноз на D+1..D+2 → строит фичи → прогоняет обе
модели → валидирует (при необходимости один резервный расчёт) → сравнивает
с прогнозом от D-1 → пишет прогноз, отчёт и трассу. Всего 28 запусков, 56 дневных CSV. В конце склеивает `submission.csv`:
на каждый час берётся прогноз минимального lead time.

## Стек

Python 3.11+ · venv+pip · pandas · LightGBM · scikit-learn · httpx · anthropic.
Обучение реализовано на CPU (ADR-004); время зависит от среды и выбранного эксперимента.
Ключ `OPENAI_API_KEY` в `.env` (не коммитить); без ключа работает режим
`--no-llm` — тот же пайплайн детерминированным циклом, судьи могут проверить без ключа.
