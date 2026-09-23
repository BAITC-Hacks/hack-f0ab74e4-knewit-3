"""График для README: прогноз vs факт на holdout (запуск: python scripts/plot_holdout.py).

Одна панель на турбину: часовой ряд за показательную неделю января 2026 (не тестовый
период — февраль модель не видела и факта у нас нет)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import TEST_END, TRAIN_START, TURBINES
from src.features.build import build_features
from src.features.dataset import load_hourly
from src.models.predict import predict
from src.weather.openmeteo import get_weather

WEEK = ("2026-01-12", "2026-01-19")  # ветреная неделя holdout
INK, BLUE, MUTED = "#3f3f46", "#2563eb", "#9ca3af"

weather = get_weather(TRAIN_START, TEST_END)
lead = 1
cols = {c: c.replace(f"__d{lead}", "") for c in weather.columns if c.endswith(f"__d{lead}")}
sl = weather[list(cols)].rename(columns=cols)
sl["lead_day"] = lead
X = build_features(sl)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fig.patch.set_facecolor("white")
for ax, t in zip(axes, TURBINES):
    actual = load_hourly(t)["power"].loc[WEEK[0]:WEEK[1]]
    pred = predict(t, X.loc[WEEK[0]:WEEK[1]])["power_pred"]
    ax.plot(actual.index, actual.values, color=INK, lw=1.6, label="Факт")
    ax.plot(pred.index, pred.values, color=BLUE, lw=1.6, label="Прогноз (lead 24 ч)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Турбина {t}\nнорм. мощность", fontsize=9)
    ax.grid(True, color="#e5e7eb", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
axes[0].legend(loc="upper right", frameon=False, fontsize=9)
axes[0].set_title("Прогноз выработки за сутки вперёд vs факт — неделя из holdout (12–19 января 2026)",
                  fontsize=11, color=INK, loc="left")
plt.tight_layout()
out = Path("docs/img/holdout_week.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150)
print(f"сохранено: {out}")
