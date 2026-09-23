"""Агрегация 10-минутного датасета организаторов к почасовому таргету.

Правила из docs/DATA.md: для цели нужны >=4 наблюдений мощности за час. Если >=50%
отсчётов имеют мощность <0.02 при ветре >4 м/с, час исключается из цели как предполагаемый
простой. Причина и непрерывная длительность простоя по этим порогам не определяются.
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
    """Почасовой факт power, отфильтрованная цель target и доля downtime_frac.

    wind_meas/temp_meas используются для диагностики и исследовательских экспериментов.
    """
    raw = load_10min(turbine)
    grp = raw.resample("1h")
    hourly = grp.mean()
    hourly["n_obs"] = grp["power"].count()
    # Доля отсчётов с признаками предполагаемого простоя за час.
    dt10 = (raw["power"] < DOWNTIME_POWER) & (raw["wind_meas"] > DOWNTIME_WIND)
    hourly["downtime_frac"] = dt10.resample("1h").mean()
    # Цель сохраняется при достаточном числе наблюдений и прохождении фильтра.
    valid = (hourly["n_obs"] >= 4) & (hourly["downtime_frac"].fillna(1) < 0.5)
    hourly["target"] = hourly["power"].where(valid)
    return hourly
