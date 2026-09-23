"""Агрегация 10-минутного датасета организаторов к почасовому таргету.

Правила из docs/DATA.md: час засчитывается при >=4 из 6 десятиминуток; длительные
простои (мощность ~0 при рабочем ветре) исключаются из обучения — это ремонт/ограничение,
а не физика ветра, предсказать их по погоде нельзя.
"""
from __future__ import annotations

import pandas as pd

from src.config import DATA_RAW

COLS = {"Статистическое время": "time",
        "Средняя скорость ветра(m/s)": "wind_meas",
        "Нормализованная активная мощность": "power",
        "Средняя температура окружающей среды(°C)": "temp_meas"}

# Признак предполагаемого простоя для одного отсчёта; доля за час проверяется ниже.
DOWNTIME_POWER = 0.02
DOWNTIME_WIND = 4.0


def load_10min(turbine: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / f"turbine_{turbine}.csv")
    df = df.rename(columns=COLS)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()[["wind_meas", "power", "temp_meas"]]


def load_hourly(turbine: int) -> pd.DataFrame:
    """Почасовые: power (таргет), wind_meas/temp_meas (только контроль), флаг downtime."""
    raw = load_10min(turbine)
    grp = raw.resample("1h")
    hourly = grp.mean()
    hourly["n_obs"] = grp["power"].count()
    # простои на 10-минутках -> доля простоя в часе
    dt10 = (raw["power"] < DOWNTIME_POWER) & (raw["wind_meas"] > DOWNTIME_WIND)
    hourly["downtime_frac"] = dt10.resample("1h").mean()
    # таргет валиден: достаточно наблюдений и час не в простое
    valid = (hourly["n_obs"] >= 4) & (hourly["downtime_frac"].fillna(1) < 0.5)
    hourly["target"] = hourly["power"].where(valid)
    return hourly
