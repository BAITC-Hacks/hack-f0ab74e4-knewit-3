"""Новые фактические цели не должны незаметно расширять период обучения подачи."""
import numpy as np
import pandas as pd

from src.features import build
from src.models import train


def _fixture(monkeypatch):
    index = pd.to_datetime([
        "2024-03-01 00:00", "2025-12-01 00:00", "2026-01-31 23:00", "2026-02-01 00:00",
    ])
    X = pd.DataFrame({"ens_ws_mean": [4., 5., 6., 7.], "lead_day": [1, 1, 2, 1]}, index=index)
    hourly = pd.DataFrame({"target": [.1, .2, .3, .9]}, index=index)
    meta = pd.DataFrame({"datetime": index, "lead_day": X["lead_day"].values})
    calls = []

    def stack(_weather):
        calls.append(1)
        return X, meta

    monkeypatch.setattr(build, "issued_feature_stack", stack)
    monkeypatch.setattr(train, "issued_feature_stack", stack, raising=False)
    monkeypatch.setattr(train, "load_hourly", lambda _turbine: hourly)

    def fit(features, target, fit_mask, tune_mask, final_mask):
        assert features[final_mask].index.max() == pd.Timestamp("2026-01-31 23:00")
        n = int(tune_mask.sum())
        return {}, {"tune_pred_baseline": np.zeros(n), "tune_pred_lgb": np.zeros(n),
                    "w_lgb": 1., "best_iteration": 200}

    monkeypatch.setattr(train, "select_and_fit", fit)
    return calls


def test_canonical_training_excludes_future_observed_targets(tmp_path, monkeypatch):
    _fixture(monkeypatch)
    report = train.train_turbine(1, pd.DataFrame(), artifacts_dir=tmp_path)
    assert report["rows_holdout"] == 2
    assert report["protocol"]["final_fit_through"] == "2026-01-31"


def test_both_turbines_share_one_issued_feature_stack(tmp_path, monkeypatch):
    calls = _fixture(monkeypatch)
    train.train_all(pd.DataFrame(), artifacts_dir=tmp_path)
    assert len(calls) == 1
