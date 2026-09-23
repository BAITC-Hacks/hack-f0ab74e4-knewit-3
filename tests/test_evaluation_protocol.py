"""Протокол ретроспективной оценки: хронология, независимость от оценочных целей,
пересчёт метрик из сохранённого CSV и продовый путь инференса.

Тесты на синтетике: контракт training_matrix (X с lead_day и ens_ws_mean,
выровненный DatetimeIndex целевых часов) воспроизводится вручную, поэтому тесты
не зависят от погодного кэша и укладываются в секунды.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.backtest.evaluate import (EVAL_PERIOD, FIT_PERIOD, TRAINED_THROUGH,
                                   TUNE_PERIOD, compute_metrics, split_masks)
from src.models.predict import predict
from src.models.train import LGB_PARAMS, save_artifact, select_and_fit

# Быстрые параметры: те же поля, меньше деревьев — тесты не должны тянуться минутами
FAST_PARAMS = {**LGB_PARAMS, "n_estimators": 30, "num_leaves": 15,
               "learning_rate": 0.15, "min_child_samples": 5}


@pytest.fixture(scope="module")
def synthetic():
    idx = pd.date_range("2024-03-01", "2026-01-31 23:00", freq="h")
    rng = np.random.default_rng(7)
    X = pd.DataFrame({
        "ens_ws_mean": rng.uniform(0, 15, len(idx)),
        "f_aux1": rng.normal(size=len(idx)),
        "f_aux2": rng.normal(size=len(idx)),
        "lead_day": rng.integers(1, 3, len(idx)),
    }, index=idx)
    y = (X["ens_ws_mean"] / 12 + rng.normal(0, 0.05, len(idx))).clip(0, 1)
    return X, pd.Series(y, index=idx, name="target")


# ---------------------------------------------------------------- хронология

def test_split_masks_disjoint_and_ordered(synthetic):
    X, _ = synthetic
    fit, tune, ev = split_masks(X.index)
    assert not (fit & tune).any(), "цель попала и в fit, и в tune"
    assert not (fit & ev).any(), "цель попала и в fit, и в оценку"
    assert not (tune & ev).any(), "цель попала и в tune, и в оценку"
    # каждый интервал строго внутри своих границ
    assert X.index[fit].max() <= pd.Timestamp(f"{FIT_PERIOD[1]} 23:59")
    assert X.index[tune].min() >= pd.Timestamp(TUNE_PERIOD[0])
    assert X.index[tune].max() <= pd.Timestamp(f"{TUNE_PERIOD[1]} 23:59")
    assert X.index[ev].min() >= pd.Timestamp(EVAL_PERIOD[0])
    assert X.index[ev].max() <= pd.Timestamp(f"{EVAL_PERIOD[1]} 23:59")
    # границы согласованы: обучение оцениваемой модели заканчивается до оценки
    assert pd.Timestamp(TRAINED_THROUGH) < pd.Timestamp(EVAL_PERIOD[0])
    # покрытие: в границах периодов не теряется ни один целевой час
    inside = (X.index >= FIT_PERIOD[0]) & (X.index <= f"{EVAL_PERIOD[1]} 23:59")
    assert ((fit | tune | ev) == inside).all()


def test_duplicate_target_hours_stay_separated(synthetic):
    """Контракт: один целевой час встречается с lead 1 и lead 2 — маски по времени
    обязаны отправлять оба примера в один и тот же интервал."""
    X, _ = synthetic
    dup = pd.DatetimeIndex(list(X.index[:4]) * 2)
    fit, tune, ev = split_masks(dup)
    assert fit[:4].tolist() == fit[4:].tolist()
    assert ev[:4].tolist() == ev[4:].tolist()


# ------------------------------------------- независимость от оценочных целей

def test_eval_targets_cannot_affect_fitted_model(synthetic):
    """Меняем целевые значения в периоде оценки — подобранная конфигурация
    (деревья, вес бленда) и прогнозы сохранённого ансамбля не должны шевелиться."""
    X, y = synthetic
    fit, tune, ev = split_masks(X.index)

    art_a, sel_a = select_and_fit(X, y, fit, tune, fit | tune, params=FAST_PARAMS)

    y_b = y.copy()
    rng = np.random.default_rng(99)
    y_b[ev] = rng.uniform(0, 1, int(ev.sum()))  # оценочные цели искажены
    art_b, sel_b = select_and_fit(X, y_b, fit, tune, fit | tune, params=FAST_PARAMS)

    assert sel_a["n_estimators"] == sel_b["n_estimators"]
    assert sel_a["w_lgb"] == sel_b["w_lgb"]
    probe = X.iloc[:200]
    pa = np.mean([m.predict(probe) for m in art_a["lgbs"]], axis=0)
    pb = np.mean([m.predict(probe) for m in art_b["lgbs"]], axis=0)
    np.testing.assert_allclose(pa, pb, atol=1e-9)
    np.testing.assert_allclose(art_a["iso"].predict(probe["ens_ws_mean"]),
                               art_b["iso"].predict(probe["ens_ws_mean"]), atol=1e-9)


def test_final_mask_never_contains_eval_targets(synthetic):
    X, _ = synthetic
    fit, tune, ev = split_masks(X.index)
    final = fit | tune
    assert not (final & ev).any()
    assert X.index[final].max() <= pd.Timestamp(f"{TRAINED_THROUGH} 23:59")


# ------------------------------------------------- продовый путь инференса

def test_predict_reads_artifact_from_alternate_dir(tmp_path, synthetic):
    X, y = synthetic
    fit, tune, _ = split_masks(X.index)
    artifact, _ = select_and_fit(X, y, fit, tune, fit | tune, params=FAST_PARAMS)
    save_artifact(artifact, 1, tmp_path)

    features = X.iloc[:48]
    out = predict(1, features, artifacts_dir=tmp_path)
    assert list(out.index) == list(features.index)
    for col in ("power_pred", "power_baseline", "power_lgb", "power_p10", "power_p90"):
        assert col in out.columns, f"нет колонки {col}"
        assert out[col].between(0, 1).all(), f"{col} вне [0, 1]"
    # сохранённый ансамбль и есть тот, что предсказывает: перезагрузка даёт то же
    out2 = predict(1, features, artifacts_dir=tmp_path)
    np.testing.assert_allclose(out["power_pred"], out2["power_pred"], atol=0)


# ------------------------------------------------- метрики из сохранённого CSV

def test_metrics_recompute_from_saved_csv(tmp_path):
    df = pd.DataFrame({
        "turbine": [1, 1, 1, 1, 2, 2],
        "issue_date": ["2025-12-01"] * 6,
        "datetime": pd.date_range("2025-12-02", periods=6, freq="h").astype(str),
        "lead_day": [1, 1, 2, 2, 1, 2],
        "power_true": [0.5, 0.3, 0.5, np.nan, 0.8, 0.2],
        "power_pred": [0.4, 0.3, 0.7, 0.5, 0.6, 0.2],
        "power_baseline": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "power_p10": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "power_p90": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        "target_eligible": [True, True, False, False, True, True],
        "trained_through": [TRAINED_THROUGH] * 6,
    })
    path = tmp_path / "evaluation_predictions.csv"
    df.to_csv(path, index=False)

    m = compute_metrics(path)
    # чистое подмножество, турбина 1, lead 1: ошибки 0.1 и 0.0
    g = m["clean_targets"]["turbine_1"]["lead_1"]
    assert g["n"] == 2
    assert abs(g["mae"] - 0.05) < 1e-12
    # все наблюдаемые часы включают строку с target_eligible=False, но не NaN
    a = m["all_observed"]["turbine_1"]["lead_2"]
    assert a["n"] == 1 and abs(a["mae"] - 0.2) < 1e-12
    # строки без факта посчитаны, а не выброшены молча
    assert m["rows"]["missing_truth"] == 1
    assert m["rows"]["total"] == 6
    # покрытие интервала измерено по факту, а не заявлено
    assert 0.0 <= m["clean_targets"]["quantile_coverage_p10_p90"] <= 1.0


def test_metrics_json_roundtrip(tmp_path):
    """Отчёт обязан сериализоваться в JSON без numpy-типов."""
    df = pd.DataFrame({
        "turbine": [1], "issue_date": ["2025-12-01"],
        "datetime": ["2025-12-02 00:00:00"], "lead_day": [1],
        "power_true": [0.5], "power_pred": [0.4], "power_baseline": [0.5],
        "power_p10": [0.1], "power_p90": [0.9],
        "target_eligible": [True], "trained_through": [TRAINED_THROUGH],
    })
    path = tmp_path / "evaluation_predictions.csv"
    df.to_csv(path, index=False)
    json.dumps(compute_metrics(path))


@pytest.mark.parametrize("column", ["power_pred", "power_baseline"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_metrics_reject_nonfinite_predictions(tmp_path, column, invalid):
    """Пропуск прогноза не должен превращаться в MAE=0 с завышенным n."""
    rows = pd.DataFrame({
        "turbine": [1, 1], "lead_day": [1, 1],
        "power_true": [0.2, 0.8], "power_pred": [0.2, 0.8],
        "power_baseline": [0.2, 0.8], "target_eligible": [True, True],
        "power_p10": [0.1, 0.7], "power_p90": [0.3, 0.9],
    })
    rows.loc[1, column] = invalid
    path = tmp_path / "invalid.csv"
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match=column):
        compute_metrics(path)


def test_replay_never_precedes_model_training_cutoff(monkeypatch, tmp_path):
    from src.backtest.evaluate import _replay
    from src.features import dataset
    from src.models import predict as inference

    index = pd.date_range(EVAL_PERIOD[0], f"{EVAL_PERIOD[1]} 23:00", freq="h")
    weather = pd.DataFrame({
        "test__wind_speed_100m__d1": 5.0,
        "test__wind_speed_100m__d2": 6.0,
    }, index=index)
    monkeypatch.setattr(dataset, "load_hourly", lambda turbine: pd.DataFrame(
        {"power": 0.5, "target": 0.5}, index=index))

    def predict_without_fit(turbine, features, artifacts_dir=None):
        return pd.DataFrame({"power_pred": 0.5, "power_baseline": 0.5,
                             "lead_day": features["lead_day"]}, index=features.index)

    monkeypatch.setattr(inference, "predict", predict_without_fit)
    rows, skipped = _replay(weather, tmp_path)
    assert not skipped
    assert rows["issue_date"].min() == TRAINED_THROUGH
    assert (rows["issue_date"] >= rows["trained_through"]).all()
    assert len(rows) == 5904
    first_day = rows[rows["datetime"].str.startswith(EVAL_PERIOD[0])]
    assert set(first_day["lead_day"]) == {1}


def test_evaluation_fit_reuses_stack_and_cuts_targets(monkeypatch, tmp_path, synthetic):
    from src.backtest.evaluate import _fit_evaluation_models
    from src.features import build, dataset
    from src.models import train

    X, y = synthetic
    calls = []
    stack = (X, pd.DataFrame({"datetime": X.index}))

    def build_once(weather):
        calls.append(weather)
        return stack

    monkeypatch.setattr(build, "issued_feature_stack", build_once)
    monkeypatch.setattr(dataset, "load_hourly", lambda turbine: y.to_frame("target"))

    def fit_without_trees(features, targets, fit, tune, final):
        assert features.index.max() <= pd.Timestamp(f"{TRAINED_THROUGH} 23:00")
        assert features.index.equals(targets.index)
        return {"features": list(features.columns)}, {
            "tune_pred_baseline": [], "tune_pred_lgb": [], "n_members": 5,
        }

    monkeypatch.setattr(train, "select_and_fit", fit_without_trees)
    monkeypatch.setattr(train, "save_artifact", lambda *args: None)
    selections = _fit_evaluation_models(pd.DataFrame(), tmp_path)
    assert len(calls) == 1
    assert selections["1"]["features"] == list(X.columns)
    assert selections["2"]["features"] == list(X.columns)


def test_report_records_configuration_and_missing_replay_rows(monkeypatch, tmp_path):
    from src.backtest import evaluate
    from src.models.train import EARLY_STOPPING_ROUNDS, QUANTILE_ALPHAS, SEEDS

    rows = pd.DataFrame({
        "turbine": [1], "issue_date": [TRAINED_THROUGH],
        "datetime": [f"{EVAL_PERIOD[0]} 00:00:00"], "lead_day": [1],
        "power_true": [0.5], "power_pred": [0.4], "power_baseline": [0.5],
        "power_p10": [0.1], "power_p90": [0.9], "target_eligible": [True],
        "trained_through": [TRAINED_THROUGH],
    })
    monkeypatch.setattr(evaluate, "_fit_evaluation_models", lambda *args: {
        "1": {"features": ["ens_ws_mean"], "n_members": 5},
    })
    monkeypatch.setattr(evaluate, "_replay", lambda *args: (rows, []))
    report = evaluate.run_evaluation(tmp_path, weather=pd.DataFrame())
    assert report["issue_range"][0] == TRAINED_THROUGH
    assert report["config"]["lgb_params"] == LGB_PARAMS
    assert report["config"]["ensemble_seeds"] == list(SEEDS)
    assert report["config"]["quantile_alphas"] == list(QUANTILE_ALPHAS)
    assert report["config"]["early_stopping_rounds"] == EARLY_STOPPING_ROUNDS
    assert report["coverage"]["expected_rows"] == 5904
    assert report["coverage"]["actual_rows"] == 1
    assert report["coverage"]["missing_rows"] == 5903
    assert report["coverage"]["excluded_before_model_cutoff"] == 48
