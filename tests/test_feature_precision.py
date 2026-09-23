"""Математически равные признаки не должны разделяться из-за младших битов libm."""
import numpy as np
import pandas as pd

from src.features import build


def weather_slice():
    index = pd.date_range("2026-02-01", periods=48, freq="h")
    frame = pd.DataFrame({
        "best_match__wind_speed_100m": np.linspace(3.1, 24.9, 48),
        "best_match__wind_direction_100m": np.tile(np.arange(0, 360, 15), 2),
        "best_match__temperature_2m": -5.2,
        "best_match__surface_pressure": 950.1,
        "lead_day": np.repeat([1, 2], 24),
    }, index=index)
    frame.loc[index[8], "best_match__wind_speed_100m"] = np.nan
    return frame


def test_symmetric_hours_have_identical_cosine_for_tree_splits():
    features = build.build_features(weather_slice())
    for hour in range(1, 12):
        assert features.iloc[hour]["hour_cos"] == features.iloc[24 - hour]["hour_cos"]
    assert features.iloc[6]["hour_cos"] == features.iloc[18]["hour_cos"] == 0


def test_one_ulp_trigonometric_variation_preserves_features(monkeypatch):
    weather = weather_slice()
    expected = build.build_features(weather)
    for name in ("sin", "cos"):
        original = getattr(np, name)
        monkeypatch.setattr(np, name, lambda values, fn=original: np.nextafter(fn(values), np.inf))
    actual = build.build_features(weather)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual["lead_day"].tolist() == weather["lead_day"].tolist()
    assert pd.isna(actual.loc[weather.index[8], "ens_ws_mean"])
