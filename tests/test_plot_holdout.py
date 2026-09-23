"""Построение графика из неполной пользовательской оценки."""

import pandas as pd
import pytest

from scripts import plot_holdout


@pytest.fixture
def evaluation_dir(tmp_path, monkeypatch):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "plot_holdout", "--evaluation-dir", str(directory),
        "--start", "2026-01-12", "--days", "1",
    ])
    yield directory
    plot_holdout.plt.close("all")


def prediction(turbine, **overrides):
    return {
        "turbine": turbine,
        "datetime": "2026-01-12 00:00:00",
        "issue_date": "2026-01-11",
        "lead_day": 1,
        "power_true": 0.4,
        "power_pred": 0.5,
        "power_baseline": 0.3,
        "power_p10": 0.2,
        "power_p90": 0.7,
        **overrides,
    }


@pytest.mark.parametrize("missing_turbine", [1, 2])
@pytest.mark.parametrize("excluded_row", [
    None, {"lead_day": 2}, {"datetime": "2026-01-13 00:00:00"},
])
@pytest.mark.parametrize("existing_image", [False, True])
def test_missing_turbine_in_selected_window_exits_without_image(
    evaluation_dir, missing_turbine, excluded_row, existing_image,
):
    rows = [prediction(3 - missing_turbine)]
    if excluded_row is not None:
        rows.append(prediction(missing_turbine, **excluded_row))
    pd.DataFrame(rows).to_csv(evaluation_dir / "evaluation_predictions.csv", index=False)
    if existing_image:
        plot_holdout.OUT.parent.mkdir(parents=True)
        plot_holdout.OUT.write_bytes(b"saved image")

    with pytest.raises(SystemExit, match=rf"турбин.*\b{missing_turbine}\b"):
        plot_holdout.main()

    if existing_image:
        assert plot_holdout.OUT.read_bytes() == b"saved image"
    else:
        assert not plot_holdout.OUT.exists()


def test_both_turbines_produce_image(evaluation_dir):
    pd.DataFrame([prediction(1), prediction(2)]).to_csv(
        evaluation_dir / "evaluation_predictions.csv", index=False,
    )

    plot_holdout.main()

    assert plot_holdout.OUT.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
