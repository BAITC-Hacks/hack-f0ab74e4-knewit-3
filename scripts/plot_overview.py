"""Обновить три обзорных SVG в README из сохранённых JSON/CSV.

    python -m scripts.plot_overview

Исторический график источников nwp_sources.svg относится к исследованию и не меняется.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, FORECASTS, ROOT

OUT = ROOT / "docs/img"
BLUE, GREY = "#2a78d6", "#b8b6ad"


def _save(fig, name, source):
    path = OUT / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None, "Description": source})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    plt.rcParams.update({
        "svg.fonttype": "none", "svg.hashsalt": "windcast", "font.family": "DejaVu Sans",
        "font.size": 10, "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.edgecolor": "#e6e5e1", "text.color": "#202522",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
    })
    train = json.loads((ARTIFACTS / "train_report.json").read_text())
    report = json.loads((ARTIFACTS / "evaluation/evaluation_report.json").read_text())
    metrics = report["metrics"]["clean_targets"]
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ys = np.array([1, 0])
    for key, label, color, offset in (
        ("baseline", "Isotonic power curve", GREY, .16),
        ("lightgbm", "LightGBM tuning model", BLUE, -.16),
    ):
        bars = ax.barh(ys + offset, [train[str(t)][key]["mae"] for t in (1, 2)],
                       height=.27, color=color, label=label)
        ax.bar_label(bars, fmt="%.4f", padding=6)
    ax.set_yticks(ys, ["Turbine 1", "Turbine 2"])
    ax.set_xlim(0, .225)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e6e5e1")
    ax.set_xlabel("MAE, fraction of rated capacity")
    ax.tick_params(length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2)
    fig.suptitle("Model tuning: December 2025 – January 2026", x=.09,
                 ha="left", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.09, right=.96, top=.77, bottom=.21)
    _save(fig, "model_ladder", "models_artifacts/train_report.json; tuning metrics.")

    fig, ax = plt.subplots(figsize=(12, 4.2))
    pairs = [(1, 1), (1, 2), (2, 1), (2, 2)]
    groups = [metrics[f"turbine_{t}"][f"lead_{lead}"] for t, lead in pairs]
    for field, label, color, offset in (
        ("mae_baseline", "Power-curve baseline", GREY, -.18),
        ("mae", "LightGBM ×5", BLUE, .18),
    ):
        bars = ax.bar(np.arange(4) + offset, [g[field] for g in groups],
                      width=.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.set_xticks(np.arange(4), [f"T{t} · lead {lead}\nn={g['n']}"
                               for (t, lead), g in zip(pairs, groups)])
    ax.set_ylim(0, .25)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e5e1")
    ax.set_ylabel("MAE, fraction of rated capacity")
    ax.tick_params(length=0)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Retrospective ensemble evaluation", x=.075, ha="left",
                 fontsize=15, fontweight="bold")
    period = report["periods"]["evaluate"]
    fig.text(.075, .865, f"{period[0]} – {period[1]} · trained through {report['trained_through']}"
             f" · {report['metrics']['rows']['clean_targets']} eligible targets",
             fontsize=10, color="#52514e")
    fig.subplots_adjust(left=.075, right=.97, top=.81, bottom=.18)
    _save(fig, "evaluation_mae", "models_artifacts/evaluation/evaluation_report.json; clean_targets.")

    frames = [pd.read_csv(path, parse_dates=["datetime"])
              for path in sorted(FORECASTS.glob("forecast_t*.csv"))]
    forecasts = (pd.concat(frames).sort_values(["turbine", "datetime", "lead_day"])
                 .drop_duplicates(["turbine", "datetime"]))
    forecasts = forecasts[(forecasts.datetime >= "2026-02-01")
                          & (forecasts.datetime < "2026-03-01")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
    for turbine, ax in zip((1, 2), axes):
        rows = forecasts[forecasts.turbine == turbine]
        ax.fill_between(rows.datetime, rows.power_p10, rows.power_p90,
                        color=BLUE, alpha=.16, label="P10–P90")
        ax.plot(rows.datetime, rows.power_pred, color=BLUE, lw=1, label="Forecast")
        ax.set_ylabel(f"T{turbine} · rated fraction")
        ax.set_ylim(-.02, 1.04)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e5e1")
        ax.tick_params(length=0)
    axes[0].legend(frameon=False, loc="upper right", ncol=2)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    axes[1].set_xlim(forecasts.datetime.min(), forecasts.datetime.max())
    fig.suptitle(f"Submission: February 2026 · {len(forecasts)} turbine-hours",
                 x=.075, ha="left", fontsize=15, fontweight="bold")
    fig.text(.075, .9, "Latest issue per hour (lead 1) · February actuals unavailable",
             fontsize=10, color="#52514e")
    fig.subplots_adjust(left=.075, right=.98, top=.84, bottom=.09, hspace=.18)
    _save(fig, "february_submission", "forecasts/forecast_t*.csv; latest lead per turbine-hour.")


if __name__ == "__main__":
    main()
