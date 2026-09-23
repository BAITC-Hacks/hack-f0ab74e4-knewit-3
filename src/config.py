"""Константы проекта: координаты, периоды, источники погоды."""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Загрузка локального .env без замены уже заданных переменных окружения."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
WEATHER_CACHE = ROOT / "data" / "weather_cache"
ARTIFACTS = ROOT / "models_artifacts"
FORECASTS = ROOT / "forecasts"

# Координаты турбин (из ссылок организаторов). Турбины в ~300 м — точка погоды одна.
TURBINES = {1: (43.645150, 78.535604), 2: (43.643198, 78.538828)}
WEATHER_POINT = (43.6452, 78.5356)

TIMEZONE = "Asia/Almaty"  # таймзона датасета, ADR-006

# Архив previous-runs покрыт с марта 2024 (проверено живыми запросами 23.09.2026)
TRAIN_START = "2024-03-01"
TRAIN_END = "2025-11-30"      # обучение
HOLDOUT_START = "2025-12-01"  # период настройки; имя сохранено для совместимости
HOLDOUT_END = "2026-01-31"
TEST_START = "2026-02-01"     # тестовый период организаторов
TEST_END = "2026-02-28"

# Погодные модели Open-Meteo: мини-ансамбль источников.
# Исторический отбор по связи прогнозного и измеренного ветра на периоде настройки.
# Сравнение источников и ограничения оценки — docs/RESEARCH.md.
WEATHER_MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless",
                  "ukmo_global_deterministic_10km"]

# Пространственные точки вокруг станции (~±0.5°): градиенты давления через хребты —
# движущая сила ветра в Шелекском коридоре (gap wind)
SPATIAL_POINTS = {"N": (44.15, 78.5356), "S": (43.15, 78.5356),
                  "E": (43.6452, 79.15), "W": (43.6452, 77.95)}
SPATIAL_MODELS = ["best_match", "icon_seamless", "ecmwf_ifs025"]
SPATIAL_VARS = ["surface_pressure", "wind_speed_100m", "temperature_2m"]

# Базовые почасовые переменные (previous-runs добавляет суффикс _previous_dayN)
WEATHER_VARS = [
    "wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m",
    "wind_direction_100m", "wind_gusts_10m", "temperature_2m", "surface_pressure",
]
LEAD_DAYS = [1, 2]  # прогноз, выпущенный за 1 и за 2 дня до валидного часа
