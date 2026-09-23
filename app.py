"""Панель оператора ВЭС: граф выполнения агента, прогноз с доверительным интервалом,
погодный ансамбль и журнал решений.

Запуск: streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (FORECASTS, HOLDOUT_END, HOLDOUT_START, TEST_END,
                        TEST_START, TRAIN_START, TURBINES)

INK, ACCENT, BAND, MUTED, GREEN, AMBER = (
    "#111821", "#1D5FD6", "rgba(29,95,214,.14)", "#8894A2", "#18734A", "#9E5A06")

st.set_page_config(page_title="WindCast — панель оператора", page_icon="🌬️", layout="wide")


# ---------------------------------------------------------------- данные

@st.cache_resource(show_spinner="Загрузка архивных прогнозов погоды…")
def _weather() -> pd.DataFrame:
    from src.weather.openmeteo import get_weather
    return get_weather(TRAIN_START, TEST_END)


@st.cache_data(show_spinner=False)
def _actual(turbine: int) -> pd.Series:
    from src.features.dataset import load_hourly
    return load_hourly(turbine)["power"]


@st.cache_data(show_spinner=False)
def _daily_forecast(turbine: int, issue: str) -> pd.DataFrame | None:
    path = FORECASTS / f"forecast_t{turbine}_{issue}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")


@st.cache_data(show_spinner=False)
def _trace(issue: str) -> dict | None:
    path = FORECASTS / f"trace_{issue}.json"
    return json.loads(path.read_text()) if path.is_file() else None


@st.cache_data(show_spinner=False)
def _report(issue: str) -> str | None:
    path = FORECASTS / f"report_{issue}.md"
    return path.read_text() if path.is_file() else None


@st.cache_data(show_spinner="Расчёт прогноза на дату holdout…")
def _holdout_forecast(turbine: int, issue: str) -> pd.DataFrame:
    """Прогноз на дату из holdout — считается на лету, чтобы сверить с фактом."""
    from src.features.build import build_features
    from src.models.predict import predict
    from src.weather.openmeteo import get_issued_forecast
    sl = get_issued_forecast(issue, _weather())
    return predict(turbine, build_features(sl))


@st.cache_data(show_spinner=False)
def _train_report() -> dict:
    from src.config import ARTIFACTS
    path = ARTIFACTS / "train_report.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _issue_dates() -> list[str]:
    return sorted(p.name[len("forecast_t1_"):-len(".csv")]
                  for p in FORECASTS.glob("forecast_t1_*.csv"))


# ---------------------------------------------------------------- граф

STATUS_STYLE = {"ok": (GREEN, "✓"), "error": ("#B3261E", "✕"), "retry": (AMBER, "↻")}


def render_graph(trace: dict | None) -> None:
    from src.agent.graph import graph_topology

    topo = graph_topology()
    done = {s["node"]: s for s in (trace or {}).get("trace", [])}

    cells = []
    for node in topo["nodes"]:
        step = done.get(node["id"])
        if step:
            color, mark = STATUS_STYLE.get(step["status"], (MUTED, "•"))
            timing = f"{step['ms']:.0f} мс"
        else:
            color, mark, timing = MUTED, "○", "не выполнялся"
        cells.append(f"""
        <div style="flex:1 1 150px;min-width:150px;border:1px solid #DEE5EC;
                    border-left:3px solid {color};border-radius:8px;padding:10px 12px;
                    background:#FFF">
          <div style="font:500 10px ui-monospace,monospace;letter-spacing:.07em;
                      text-transform:uppercase;color:{color}">{mark} {step['status'] if step else 'ожидание'}</div>
          <div style="font:600 13px system-ui;color:#111821;margin:4px 0 2px">{node['label']}</div>
          <div style="font:400 11px ui-monospace,monospace;color:#8894A2">{node['id']} · {timing}</div>
        </div>""")

    retried = (trace or {}).get("retried")
    banner = ("" if not retried else
              f"<div style='font:500 12px system-ui;color:{AMBER};margin-bottom:8px'>"
              "↻ Валидация не прошла с первой попытки — граф выполнил повторный цикл</div>")
    st.markdown(banner + "<div style='display:flex;gap:8px;flex-wrap:wrap'>"
                + "".join(cells) + "</div>", unsafe_allow_html=True)
    st.caption("Условное ребро: validate_forecast → fetch_weather при неудачной валидации "
               "(одна попытка), иначе → compare_with_previous. Трасса пишется в "
               "forecasts/trace_<дата>.json.")


# ---------------------------------------------------------------- графики

def power_chart(pred: pd.DataFrame, actual: pd.Series | None, title: str) -> go.Figure:
    fig = go.Figure()
    if {"power_p10", "power_p90"} <= set(pred.columns):
        fig.add_trace(go.Scatter(x=pred.index, y=pred["power_p90"], mode="lines",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=pred.index, y=pred["power_p10"], mode="lines",
                                 line=dict(width=0), fill="tonexty", fillcolor=BAND,
                                 name="Интервал P10–P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pred.index, y=pred["power_pred"], mode="lines",
                             line=dict(color=ACCENT, width=2.4), name="Прогноз",
                             hovertemplate="%{x|%d.%m %H:%M}<br>прогноз %{y:.3f}<extra></extra>"))
    if actual is not None and actual.notna().any():
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines",
                                 line=dict(color=INK, width=2), name="Факт",
                                 hovertemplate="%{x|%d.%m %H:%M}<br>факт %{y:.3f}<extra></extra>"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), height=340,
        margin=dict(l=8, r=8, t=44, b=8), hovermode="x unified",
        yaxis=dict(title="норм. мощность", range=[0, 1.04], gridcolor="#EEF2F7"),
        xaxis=dict(gridcolor="#EEF2F7"), plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
        font=dict(family="system-ui", color=INK))
    return fig


def wind_chart(slice_: pd.DataFrame) -> go.Figure:
    cols = [c for c in slice_.columns if c.endswith("wind_speed_100m")
            or c.endswith("wind_speed_10m")]
    fig = go.Figure()
    for c in cols:
        src = c.split("__")[0]
        fig.add_trace(go.Scatter(x=slice_.index, y=slice_[c], mode="lines", name=src,
                                 line=dict(width=1.3, color=MUTED), opacity=.55,
                                 hovertemplate=f"{src}: %{{y:.1f}} м/с<extra></extra>"))
    if cols:
        fig.add_trace(go.Scatter(x=slice_.index, y=slice_[cols].mean(axis=1), mode="lines",
                                 name="среднее ансамбля", line=dict(color=ACCENT, width=2.4),
                                 hovertemplate="ансамбль: %{y:.1f} м/с<extra></extra>"))
    fig.update_layout(
        title=dict(text="Прогноз ветра: ансамбль NWP-источников", font=dict(size=15)),
        height=300, margin=dict(l=8, r=8, t=44, b=8), hovermode="x unified",
        yaxis=dict(title="скорость ветра, м/с", gridcolor="#EEF2F7"),
        xaxis=dict(gridcolor="#EEF2F7"), plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.16, x=0, font=dict(size=10)),
        font=dict(family="system-ui", color=INK))
    return fig


# ---------------------------------------------------------------- страница

st.markdown("## 🌬️ WindCast — панель оператора ВЭС")
st.caption("Шелекский коридор, Алматинская область · две турбины · горизонт 24–48 часов · "
           "вход — архивные прогнозы погоды, доступные на момент выпуска")

with st.sidebar:
    st.markdown("### Режим")
    mode = st.radio("Режим", ["Тестовый период (февраль 2026)",
                              "Проверка на holdout (есть факт)"],
                    label_visibility="collapsed")
    turbine = st.selectbox("Турбина", list(TURBINES), format_func=lambda t: f"Турбина {t}")

    if mode.startswith("Тестовый"):
        dates = _issue_dates()
        issue = st.select_slider("Дата выпуска прогноза", dates,
                                 value=dates[len(dates) // 2]) if dates else None
    else:
        issue = st.date_input("Дата выпуска прогноза",
                              value=date(2026, 1, 14),
                              min_value=date.fromisoformat(HOLDOUT_START),
                              max_value=date.fromisoformat(HOLDOUT_END) - timedelta(days=2)
                              ).isoformat()

    st.divider()
    lat, lon = TURBINES[turbine]
    st.markdown(f"**Координаты**  \n`{lat}, {lon}`")
    rep = _train_report().get(str(turbine), {})
    if rep:
        st.markdown(f"**MAE на holdout**  \n`{rep['lightgbm']['mae']:.4f}`  \n"
                    f"0–24 ч `{rep['by_lead']['1']['mae']:.4f}` · "
                    f"24–48 ч `{rep['by_lead']['2']['mae']:.4f}`")

if not issue:
    st.warning("Прогнозы не найдены. Сначала выполните "
               "`python -m src.cli run-agent --start 2026-01-31 --end 2026-02-27 --no-llm`.")
    st.stop()

holdout_mode = not mode.startswith("Тестовый")

# --- прогноз и факт
if holdout_mode:
    pred = _holdout_forecast(turbine, issue)
    actual = _actual(turbine).reindex(pred.index)
    subtitle = "прогноз рассчитан на лету, факт известен — видно качество модели"
else:
    pred = _daily_forecast(turbine, issue)
    actual = _actual(turbine).reindex(pred.index) if pred is not None else None
    subtitle = "тестовый период: факта нет, его знают только организаторы"

if pred is None or pred.empty:
    st.error(f"Нет прогноза на {issue}.")
    st.stop()

# --- сводка дня
c1, c2, c3, c4 = st.columns(4)
day1 = pred[pred["lead_day"] == 1]["power_pred"] if "lead_day" in pred else pred["power_pred"]
c1.metric("Средняя мощность, 48 ч", f"{pred['power_pred'].mean():.3f}")
c2.metric("Пик прогноза", f"{pred['power_pred'].max():.3f}")
c3.metric("Выработка за 1-е сутки", f"{day1.sum():.1f}", help="сумма нормированной мощности")
if actual is not None and actual.notna().any():
    err = (pred["power_pred"] - actual).abs().mean()
    c4.metric("MAE этого дня", f"{err:.3f}")
else:
    c4.metric("Горизонт", f"{len(pred)} ч")

st.plotly_chart(power_chart(pred, actual, f"Турбина {turbine}: прогноз выработки — {subtitle}"),
                use_container_width=True)

# --- погодный ансамбль
from src.weather.openmeteo import get_issued_forecast  # noqa: E402

try:
    st.plotly_chart(wind_chart(get_issued_forecast(issue, _weather())),
                    use_container_width=True)
except Exception as exc:
    st.info(f"Погодный срез недоступен: {exc}")

# --- граф и журнал
left, right = st.columns([3, 2])

with left:
    st.markdown("#### Граф выполнения агента")
    tr = _trace(issue)
    if tr:
        render_graph(tr)
    elif holdout_mode:
        st.info("Для дат holdout агент не запускался — прогноз посчитан напрямую моделью. "
                "Граф наполняется при `run-agent`.")
    else:
        st.info("Трасса не найдена: этот день считался до появления графа. "
                "Перезапустите `run-agent` на эту дату.")

with right:
    st.markdown("#### Журнал решений агента")
    rp = _report(issue)
    if rp:
        st.markdown(f"<div style='font:13px/1.6 system-ui;max-height:330px;overflow:auto;"
                    f"border:1px solid #DEE5EC;border-radius:8px;padding:12px 14px'>"
                    f"{rp.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
    else:
        st.info("Отчёт за эту дату не найден.")

st.divider()
st.caption(f"Данные: датасет организаторов (10-мин SCADA, 03.2023–01.2026) и Open-Meteo "
           f"Previous Runs API. Тестовый период {TEST_START} – {TEST_END}. "
           f"Подробности в README и docs/.")
