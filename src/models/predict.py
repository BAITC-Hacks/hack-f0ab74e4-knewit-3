"""Инференс: прогноз нормализованной мощности по фичам архивного прогноза погоды."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from src.config import ARTIFACTS


def load_model(turbine: int) -> dict:
    with open(ARTIFACTS / f"turbine_{turbine}.pkl", "rb") as f:
        return pickle.load(f)


def predict(turbine: int, features: pd.DataFrame) -> pd.DataFrame:
    art = load_model(turbine)
    X = features.reindex(columns=art["features"])
    lgb_p = np.clip(np.mean([m.predict(X) for m in art["lgbs"]], axis=0), 0, 1)
    iso_p = art["iso"].predict(X["ens_ws_mean"])
    w = art["w_lgb"]
    out = pd.DataFrame(index=features.index)
    out["power_baseline"] = iso_p
    out["power_lgb"] = lgb_p
    out["power_pred"] = np.clip(w * lgb_p + (1 - w) * iso_p, 0, 1)
    if "lead_day" in features.columns:
        out["lead_day"] = features["lead_day"]
    return out
