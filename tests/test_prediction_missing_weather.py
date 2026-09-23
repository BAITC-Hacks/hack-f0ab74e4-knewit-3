"""Потерянный час погоды доходит до валидации и не превращается в выдуманную мощность."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.isotonic import IsotonicRegression

from src.agent.graph import DayGraph
from src.models import predict as inference


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    directory = tmp_path / "models"
    directory.mkdir()
    X = pd.DataFrame({"ens_ws_mean": [0.0, 10.0], "lead_day": [1, 2]})
    y = np.array([0.0, 1.0])
    artifact = {
        "features": list(X.columns),
        "lgbs": [DummyRegressor(strategy="constant", constant=0.6).fit(X, y)],
        "iso": IsotonicRegression(out_of_bounds="clip").fit(X["ens_ws_mean"], y),
        "w_lgb": 0.5,
        "quantiles": {
            q: DummyRegressor(strategy="constant", constant=q).fit(X, y)
            for q in (0.1, 0.9)
        },
    }
    for turbine in (1, 2):
        with (directory / f"turbine_{turbine}.pkl").open("wb") as stream:
            pickle.dump(artifact, stream)
    monkeypatch.setattr(inference, "ARTIFACTS", directory)
    return directory


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_unavailable_wind_stays_missing_and_valid_predictions_are_preserved(artifacts, missing):
    features = pd.DataFrame({
        "ens_ws_mean": [1.0, missing, 8.0], "lead_day": [1, 1, 2],
    }, index=pd.date_range("2026-02-11", periods=3, freq="h"))
    result = inference.predict(1, features)
    power = [column for column in result if column.startswith("power_")]
    assert result.iloc[1][power].isna().all()
    np.testing.assert_allclose(result["power_baseline"].iloc[[0, 2]], [0.1, 0.8])
    np.testing.assert_allclose(result["power_pred"].iloc[[0, 2]], [0.35, 0.7])
    np.testing.assert_allclose(result["power_lgb"].iloc[[0, 2]], [0.6, 0.6])
    np.testing.assert_allclose(result["power_p10"].iloc[[0, 2]], [0.1, 0.1])
    np.testing.assert_allclose(result["power_p90"].iloc[[0, 2]], [0.9, 0.9])
    pd.testing.assert_series_equal(result["lead_day"], features["lead_day"])


def test_all_missing_wind_returns_missing_predictions_without_estimator_exception(artifacts):
    features = pd.DataFrame({"ens_ws_mean": [np.nan, np.inf], "lead_day": [1, 2]})
    result = inference.predict(1, features)
    assert result.filter(like="power_").isna().all().all()
    assert result["lead_day"].tolist() == [1, 2]


def test_missing_weather_hour_retries_baseline_then_stops_without_publishing(artifacts, tmp_path):
    index = pd.date_range("2026-02-11", periods=48, freq="h", name="time")
    weather = pd.DataFrame({
        f"best_match__wind_speed_100m__d{lead}": np.linspace(3.0, 9.0, 48)
        for lead in (1, 2)
    }, index=index).drop(index=index[8])
    output = tmp_path / "forecast"
    runner = DayGraph("2026-02-10", weather, output_dir=output)
    with pytest.raises(RuntimeError, match="Повторная валидация"):
        runner.finish(lambda outputs, context: "Прогноз не должен публиковаться")

    trace = json.loads((output / "trace_2026-02-10.json").read_text())
    assert not trace["completed"] and trace["retried"]
    assert [step["node"] for step in trace["trace"]] == [
        "fetch_weather", "prepare_features", "run_model", "validate_forecast",
        "recover_baseline", "validate_forecast",
    ]
    assert trace["trace"][-1]["status"] == "invalid"
    for prediction in runner.ctx.preds.values():
        assert pd.isna(prediction.loc[index[8], "power_pred"])
    assert not list(output.glob("forecast_*.csv"))
    assert not list(output.glob("report_*.md"))
