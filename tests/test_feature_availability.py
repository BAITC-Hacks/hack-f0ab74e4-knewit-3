"""Одинаковые входы обучения и выпуска прогноза (docs/FEATURE_AVAILABILITY.md).

Синтетический архив без сети: значение каждой погодной колонки кодирует день, lead и час
(`day_idx*1000 + lead*100 + hour`), поэтому по числу видно, из какого выпуска оно пришло.
Выпуск D владеет ровно двумя блоками архива: `__d1` за день D+1 и `__d2` за день D+2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build
from src.weather import openmeteo

DAY0 = pd.Timestamp("2026-01-01")
N_DAYS = 8
MODELS = ("best_match", "ecmwf_ifs025")
VARS = ("wind_speed_10m", "wind_speed_100m", "wind_speed_120m", "wind_direction_100m",
        "wind_gusts_10m", "temperature_2m", "surface_pressure")


def make_archive(n_days: int = N_DAYS) -> pd.DataFrame:
    idx = pd.date_range(DAY0, periods=24 * n_days, freq="h", name="time")
    day_idx = (idx.normalize() - DAY0).days.values
    cols = {}
    for model_i, model in enumerate(MODELS):
        for var_i, var in enumerate(VARS):
            for lead in (1, 2):
                code = day_idx * 1000 + lead * 100 + idx.hour.values
                # разные переменные/модели смещены, чтобы фичи не совпадали случайно
                cols[f"{model}__{var}__d{lead}"] = code + model_i * 0.1 + var_i * 0.01
    return pd.DataFrame(cols, index=idx)


def make_target(archive: pd.DataFrame) -> pd.Series:
    return pd.Series(np.linspace(0.0, 1.0, len(archive)), index=archive.index, name="target")


def day(n: int) -> pd.Timestamp:
    return DAY0 + pd.Timedelta(days=n)


def issue(n: int) -> str:
    return day(n).date().isoformat()


def rows_of_issue(archive, target, issue_date, **kw):
    """Строки обучающей матрицы, происходящие из выпуска issue_date."""
    X, y = build.training_matrix(archive, target, **kw)
    meta = build.training_provenance(archive, target, **kw)
    assert len(meta) == len(X)
    mask = (meta["issue_date"] == issue_date).values
    return X[mask], y[mask], meta[mask]


def inference_features(archive, issue_date):
    return build.build_features(openmeteo.get_issued_forecast(issue_date, archive))


# ---------------------------------------------------------------- равенство train == inference

def test_training_rows_equal_inference_features_for_same_issue():
    archive, target = make_archive(), make_target(make_archive())
    for n in (0, 3, N_DAYS - 3):
        X_tr, y_tr, meta = rows_of_issue(archive, target, issue(n))
        X_inf = inference_features(archive, issue(n))
        assert len(X_tr) == len(X_inf) == 48
        pd.testing.assert_frame_equal(X_tr, X_inf, check_like=True)
        assert list(meta["lead_day"]) == [1] * 24 + [2] * 24
        assert X_tr.index.equals(y_tr.index)


def test_midnight_lag_comes_from_same_issue_not_neighbouring_day():
    """00:00 дня D+2 (lead 2): lag1 — это 23:00 дня D+1 из lead-1 колонки того же выпуска."""
    archive, target = make_archive(), make_target(make_archive())
    n = 2
    X_tr, _, _ = rows_of_issue(archive, target, issue(n))
    row = X_tr.loc[day(n + 2)]                       # 00:00 D+2
    expected = archive.loc[day(n + 1) + pd.Timedelta(hours=23), "best_match__wind_speed_100m__d1"]
    assert row["best_match_ws_lag1"] == pytest.approx(expected)
    # а не 23:00 D+1 из lead-2 колонки (так делала старая реализация)
    wrong = archive.loc[day(n + 1) + pd.Timedelta(hours=23), "best_match__wind_speed_100m__d2"]
    assert row["best_match_ws_lag1"] != pytest.approx(wrong)


def test_slice_edges_do_not_see_other_issues():
    """Первый час среза без предыстории, последний — без опережения: за краями другой выпуск."""
    archive, target = make_archive(), make_target(make_archive())
    n = 2
    X_tr, _, _ = rows_of_issue(archive, target, issue(n))
    first, last = X_tr.iloc[0], X_tr.iloc[-1]
    assert np.isnan(first["best_match_ws_lag1"]) and np.isnan(first["best_match_ws_lag3"])
    assert np.isnan(last["best_match_ws_lead1"]) and np.isnan(last["best_match_ws_lead3"])
    # центрированное окно на краю усредняет только доступные часы своего выпуска
    ws = archive.loc[day(n + 2) + pd.Timedelta(hours=22):day(n + 2) + pd.Timedelta(hours=23),
                     "best_match__wind_speed_100m__d2"]
    assert last["best_match_ws_roll3"] == pytest.approx(ws.mean())


def test_changing_other_issues_cannot_change_checked_issue_inputs():
    archive, target = make_archive(), make_target(make_archive())
    n = 2
    before, y_before, _ = rows_of_issue(archive, target, issue(n))

    mutated = archive.copy()
    own = {(1, n + 1), (2, n + 2)}                    # (lead, день), которыми владеет выпуск D
    for col in mutated.columns:
        lead = int(col[-1])
        for d in range(N_DAYS):
            if (lead, d) not in own:
                sel = mutated.index.normalize() == day(d)
                mutated.loc[sel, col] = mutated.loc[sel, col] * 7 + 5000
    after, y_after, _ = rows_of_issue(mutated, target, issue(n))
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_series_equal(y_before, y_after)
    # контроль: у соседних выпусков строки действительно изменились
    other_before, _, _ = rows_of_issue(archive, target, issue(n + 1))
    other_after, _, _ = rows_of_issue(mutated, target, issue(n + 1))
    assert not other_before.equals(other_after)


def test_adjacent_hours_across_midnight_and_leads():
    archive, target = make_archive(), make_target(make_archive())
    n = 3
    X_tr, _, meta = rows_of_issue(archive, target, issue(n))
    h23 = day(n + 1) + pd.Timedelta(hours=23)
    h00 = day(n + 2)
    assert meta.set_index("datetime").loc[h23, "lead_day"] == 1
    assert meta.set_index("datetime").loc[h00, "lead_day"] == 2
    # опережение из 23:00 D+1 смотрит на 00:00 D+2 того же выпуска (lead 2)
    assert X_tr.loc[h23, "best_match_ws_lead1"] == pytest.approx(
        archive.loc[h00, "best_match__wind_speed_100m__d2"])
    assert X_tr.loc[h00, "lead_day"] == 2 and X_tr.loc[h23, "lead_day"] == 1


# ---------------------------------------------------------------- границы архива

def test_each_issue_target_pair_appears_once_and_every_hour_twice():
    archive, target = make_archive(), make_target(make_archive())
    meta = build.training_provenance(archive, target)
    assert not meta.duplicated(["issue_date", "datetime"]).any()
    counts = meta.groupby("datetime").size()
    interior = counts[(counts.index >= day(2)) & (counts.index < day(N_DAYS))]
    assert (interior == 2).all()                      # lead 1 от D-1 и lead 2 от D-2
    assert (counts[counts.index < day(1)] == 1).all()  # день 0: только lead 1 от выпуска «-1»


def test_archive_start_first_issue_is_day_before_archive():
    archive, target = make_archive(), make_target(make_archive())
    issues = openmeteo.issue_dates(archive)
    assert issues[0] == (DAY0 - pd.Timedelta(days=1)).date().isoformat()
    first_slice = openmeteo.get_issued_forecast(issues[0], archive)
    assert list(first_slice["lead_day"].unique()) == [1, 2]
    # выпуск за два дня до начала архива имел бы только lead 2 — инференс такой формы не делает
    two_before = (DAY0 - pd.Timedelta(days=2)).date().isoformat()
    assert two_before not in issues
    meta = build.training_provenance(archive, target)
    assert two_before not in set(meta["issue_date"])


def test_final_boundary_gives_24_hours_lead_1_only():
    archive, target = make_archive(), make_target(make_archive())
    issues = openmeteo.issue_dates(archive)
    last = issues[-1]
    assert last == day(N_DAYS - 2).date().isoformat()
    sl = openmeteo.get_issued_forecast(last, archive)
    assert len(sl) == 24 and sl["lead_day"].unique().tolist() == [1]
    X_tr, _, meta = rows_of_issue(archive, target, last)
    assert len(X_tr) == 24 and set(meta["lead_day"]) == {1}
    pd.testing.assert_frame_equal(X_tr, inference_features(archive, last), check_like=True)
    with pytest.raises(ValueError):
        openmeteo.get_issued_forecast(day(N_DAYS - 1).date().isoformat(), archive)


def test_missing_interior_hour_is_nan_row_not_glued_neighbour():
    archive, target = make_archive(), make_target(make_archive())
    hole = day(3) + pd.Timedelta(hours=10)
    archive = archive.drop(index=hole)
    sl = openmeteo.get_issued_forecast(issue(2), archive)   # день 3 — lead 1 этого выпуска
    assert len(sl) == 48 and hole in sl.index
    assert sl.loc[hole].drop("lead_day").isna().all()
    X = build.build_features(sl)
    # lag1 в 11:00 — это пропавший час (NaN), а не 09:00
    assert np.isnan(X.loc[hole + pd.Timedelta(hours=1), "best_match_ws_lag1"])
    X_tr, y_tr, _ = rows_of_issue(archive, target, issue(2))
    assert hole not in X_tr.index                            # строка без ветра не обучает
    assert len(X_tr) == 47


def test_missing_source_values_drop_only_rows_without_ensemble_wind():
    archive, target = make_archive(), make_target(make_archive())
    bad_hours = archive.index[(archive.index.normalize() == day(3)) & (archive.index.hour < 3)]
    for m in MODELS:
        archive.loc[bad_hours, f"{m}__wind_speed_100m__d1"] = np.nan
        archive.loc[bad_hours, f"{m}__wind_speed_120m__d1"] = np.nan
        archive.loc[bad_hours, f"{m}__wind_speed_10m__d1"] = np.nan
    X_tr, y_tr, meta = rows_of_issue(archive, target, issue(2))
    assert len(X_tr) == 45 and not set(bad_hours) & set(X_tr.index)
    assert X_tr["ens_ws_mean"].notna().all()
    # частичный NaN (одна модель) строку не удаляет — LightGBM переживает пропуски
    archive2 = make_archive()
    archive2.loc[bad_hours, "ecmwf_ifs025__wind_speed_100m__d1"] = np.nan
    X2, _, _ = rows_of_issue(archive2, target, issue(2))
    assert len(X2) == 48


def test_missing_targets_are_dropped_not_filled_and_xy_aligned():
    archive = make_archive()
    target = make_target(archive)
    target.loc[day(3) + pd.Timedelta(hours=5):day(3) + pd.Timedelta(hours=7)] = np.nan
    target = target.drop(index=day(4) + pd.Timedelta(hours=1))   # часа вовсе нет в таргете
    X, y = build.training_matrix(archive, target)
    assert len(X) == len(y) and X.index.equals(y.index)
    assert y.notna().all()
    gone = [day(3) + pd.Timedelta(hours=h) for h in (5, 6, 7)] + [day(4) + pd.Timedelta(hours=1)]
    assert not set(gone) & set(X.index)


def test_target_end_cutoff_keeps_later_hours_out_of_training():
    archive, target = make_archive(), make_target(make_archive())
    cutoff = day(4) + pd.Timedelta(hours=23)
    X, y = build.training_matrix(archive, target, target_end=cutoff)
    assert X.index.max() <= cutoff and y.index.max() <= cutoff
    meta = build.training_provenance(archive, target, target_end=cutoff)
    assert meta["datetime"].max() <= cutoff and len(meta) == len(X)
    X_all, _ = build.training_matrix(archive, target)
    assert len(X_all) > len(X)


def test_index_must_be_naive_hourly_datetime():
    archive = make_archive()
    aware = archive.tz_localize("Asia/Almaty")
    with pytest.raises(ValueError, match="naive"):
        openmeteo.get_issued_forecast(issue(1), aware)
    with pytest.raises(ValueError):
        openmeteo.get_issued_forecast(issue(1), archive.reset_index(drop=True))


# ---------------------------------------------------------------- реальный кэш (офлайн)

def _cache_complete() -> bool:
    from src.config import TEST_END, TRAIN_START, WEATHER_MODELS
    return all(openmeteo._cache_path(m, s, e).exists()
               for m in WEATHER_MODELS for s, e in openmeteo._chunks(TRAIN_START, TEST_END))


@pytest.mark.skipif(not _cache_complete(), reason="погодный кэш не полный — сеть запрещена")
def test_real_cache_training_rows_match_daily_inference(monkeypatch):
    from src.config import TEST_END, TRAIN_START

    def no_network(*a, **k):
        raise AssertionError("тест не должен ходить в сеть")
    monkeypatch.setattr(openmeteo.httpx, "get", no_network)

    weather = openmeteo.get_weather(TRAIN_START, TEST_END)
    target = pd.Series(0.5, index=weather.index)      # таргет здесь не важен, важны фичи
    checks = {"2024-03-01": 48, "2024-08-05": 48, "2025-06-15": 48, "2026-02-27": 24}
    for issue_date, n_rows in checks.items():
        X_inf = inference_features(weather, issue_date)
        assert len(X_inf) == n_rows
        sl = openmeteo.get_issued_forecast(issue_date, weather)
        X_one, _ = build.training_matrix(weather.loc[sl.index.min() - pd.Timedelta(days=3):
                                                     sl.index.max() + pd.Timedelta(days=3)],
                                         target)
        meta_one = build.training_provenance(weather.loc[sl.index.min() - pd.Timedelta(days=3):
                                                         sl.index.max() + pd.Timedelta(days=3)],
                                             target)
        X_tr = X_one[(meta_one["issue_date"] == issue_date).values]
        # колонки источников, отсутствующих в этом выпуске (UKMO до 06.08.2024, ECMWF в первые
        # дни), в стеке есть и целиком NaN; по остальным значения совпадают с инференсом
        extra = [c for c in X_tr.columns if c not in X_inf.columns]
        assert X_tr[extra].isna().all().all()
        pd.testing.assert_frame_equal(X_tr[X_inf.columns], X_inf.loc[X_tr.index])
        # основная скорость каждого источника берётся с одной и той же высоты во всех выпусках
        for model, col in (("best_match", "wind_speed_100m"),
                           ("ukmo_global_deterministic_10km", "wind_speed_10m")):
            if f"{model}_ws_main" in X_inf.columns:
                pd.testing.assert_series_equal(
                    X_inf[f"{model}_ws_main"], sl[f"{model}__{col}"], check_names=False)
