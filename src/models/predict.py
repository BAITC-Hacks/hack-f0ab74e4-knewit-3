"""Инференс: прогноз нормализованной мощности по фичам архивного прогноза погоды.

artifacts_dir позволяет читать альтернативный набор артефактов (например, модель
ретроспективной оценки с более ранней отсечкой обучения) тем же продовым путём;
по умолчанию — канонические артефакты сдачи.
"""
from __future__ import annotations

from pathlib import Path

import pickle

import numpy as np
import pandas as pd

from src.config import ARTIFACTS


def load_model(turbine: int, artifacts_dir: Path | str | None = None) -> dict:
    directory = Path(artifacts_dir or ARTIFACTS)
    with open(directory / f"turbine_{turbine}.pkl", "rb") as f:
        return pickle.load(f)


def predict(turbine: int, features: pd.DataFrame,
            artifacts_dir: Path | str | None = None) -> pd.DataFrame:
    art = load_model(turbine, artifacts_dir)
    X = features.reindex(columns=art["features"])
    # Без ветра всех источников нет и физического baseline. Сохраняем пропуск,
    # чтобы граф выполнил валидацию и остановил публикацию после резервной попытки.
    available = np.isfinite(X["ens_ws_mean"].to_numpy(dtype=float))
    columns = ["power_baseline", "power_lgb", "power_pred"]
    if "quantiles" in art:
        columns += ["power_p10", "power_p90"]
    out = pd.DataFrame(np.nan, index=features.index, columns=columns)
    if available.any():
        valid = X.loc[available]
        lgb_p = np.clip(np.mean([m.predict(valid) for m in art["lgbs"]], axis=0), 0, 1)
        iso_p = art["iso"].predict(valid["ens_ws_mean"])
        w = art["w_lgb"]
        point = np.clip(w * lgb_p + (1 - w) * iso_p, 0, 1)
        out.loc[available, "power_baseline"] = iso_p
        out.loc[available, "power_lgb"] = lgb_p
        out.loc[available, "power_pred"] = point
        if "quantiles" in art:
            p10 = np.clip(art["quantiles"][0.1].predict(valid), 0, 1)
            p90 = np.clip(art["quantiles"][0.9].predict(valid), 0, 1)
            out.loc[available, "power_p10"] = np.minimum(p10, point)
            out.loc[available, "power_p90"] = np.maximum(p90, point)
    if "lead_day" in features.columns:
        out["lead_day"] = features["lead_day"]
    return out
