"""График для README из сохранённой ретроспективной оценки.

    python -m scripts.plot_holdout [--evaluation-dir models_artifacts/evaluation]
                                   [--lead 1] [--start 2026-01-12] [--days 7]

Читает evaluation_predictions.csv — прогнозы модели, обученной строго до оценочного
периода (см. docs/EVALUATION.md). Финальная модель здесь не пересчитывается.
Каждый показанный день — отдельный выпуск: сегменты рисуются с разрывом на границе
выпусков, чтобы разные выпуски не выглядели одной непрерывной серией.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import TURBINES

INK, BLUE, GREY, BAND = "#3f3f46", "#2563eb", "#9ca3af", "#2563eb22"
OUT = Path("docs/img/holdout_week.png")


def _with_issue_gaps(rows: pd.DataFrame) -> pd.DataFrame:
    """NaN-строка между выпусками: линия рвётся на границе, а не мостится через неё."""
    parts = []
    for _, grp in rows.groupby("issue_date", sort=True):
        parts.append(grp.sort_values("datetime"))
        gap = grp.iloc[-1:].copy()
        gap["datetime"] = gap["datetime"] + pd.Timedelta(minutes=30)
        for col in ("power_true", "power_pred", "power_baseline", "power_p10", "power_p90"):
            gap[col] = np.nan
        parts.append(gap)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluation-dir", type=Path, default=Path("models_artifacts/evaluation"))
    ap.add_argument("--lead", type=int, choices=(1, 2), default=1)
    ap.add_argument("--start", default="2026-01-12")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    csv_path = args.evaluation_dir / "evaluation_predictions.csv"
    report_path = args.evaluation_dir / "evaluation_report.json"
    if not csv_path.is_file():
        raise SystemExit(f"Нет {csv_path}. Сначала: python -m src.backtest.evaluate "
                         f"--output-dir {args.evaluation_dir}")
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    trained_through = "?"
    if report_path.is_file():
        trained_through = json.loads(report_path.read_text()).get("trained_through", "?")

    lo = pd.Timestamp(args.start)
    hi = lo + pd.Timedelta(days=args.days)
    window = df[(df["lead_day"] == args.lead)
                & (df["datetime"] >= lo) & (df["datetime"] < hi)]
    if window.empty:
        raise SystemExit(f"В оценке нет строк lead {args.lead} за {args.start} +{args.days}д.")
    missing_turbines = [t for t in TURBINES if t not in window["turbine"].unique()]
    if missing_turbines:
        missing = ", ".join(map(str, missing_turbines))
        raise SystemExit(f"В выбранном окне оценки нет строк для турбин: {missing}. "
                         "Для графика нужны обе турбины.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.patch.set_facecolor("white")
    for ax, t in zip(axes, TURBINES):
        rows = _with_issue_gaps(window[window["turbine"] == t])
        x = rows["datetime"]
        if rows["power_p10"].notna().any():
            ax.fill_between(x, rows["power_p10"], rows["power_p90"],
                            color=BAND, linewidth=0, label="P10–P90")
        ax.plot(x, rows["power_true"], color=INK, lw=1.6, label="Факт")
        ax.plot(x, rows["power_pred"], color=BLUE, lw=1.6,
                label="Прогноз (сохранённая оценка)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Турбина {t}\nнорм. мощность", fontsize=9)
        ax.grid(True, color="#e5e7eb", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=GREY, labelsize=8)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    end = (hi - pd.Timedelta(days=1)).date()
    axes[0].set_title(
        f"Ретроспективная оценка, lead {args.lead} (выпуск за {args.lead} дн. до цели): "
        f"{lo.date()} – {end}\nМодель обучена по {trained_through}; "
        f"разрывы линий — границы отдельных выпусков",
        fontsize=10.5, color=INK, loc="left")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"сохранено: {OUT}")


if __name__ == "__main__":
    main()
