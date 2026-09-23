"""Панель оператора ВЭС: граф выполнения агента, прогноз с доверительным интервалом,
погодный ансамбль и журнал решений.

Запуск: streamlit run app.py
"""
from __future__ import annotations

import json
import argparse
import os
import shlex
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (FORECASTS, HOLDOUT_END, HOLDOUT_START, TEST_END,
                        TEST_START, TRAIN_START, TURBINES)

_args = argparse.ArgumentParser(add_help=False)
_args.add_argument("--forecast-dir", type=Path)
_args.add_argument("--evaluation-dir", type=Path)
_options, _ = _args.parse_known_args()
if _options.forecast_dir is not None:
    FORECASTS = _options.forecast_dir
# Каталог сохранённой ретроспективной оценки; env — для тестов без CLI-аргументов
EVALUATION_DIR = (_options.evaluation_dir
                  or Path(os.environ.get("WINDCAST_EVALUATION_DIR",
                                         "models_artifacts/evaluation")))

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


def _daily_forecast(turbine: int, issue: str) -> pd.DataFrame | None:
    path = FORECASTS / f"forecast_t{turbine}_{issue}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")


def _trace(issue: str) -> dict | None:
    path = FORECASTS / f"trace_{issue}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _report(issue: str) -> str | None:
    path = FORECASTS / f"report_{issue}.md"
    return path.read_text() if path.is_file() else None


EVAL_CSV_COLUMNS = {"turbine", "issue_date", "datetime", "lead_day", "power_true",
                    "power_pred", "power_baseline", "power_p10", "power_p90",
                    "target_eligible", "trained_through"}


@st.cache_data(show_spinner="Чтение сохранённой оценки…")
def _evaluation_data(directory: str) -> tuple[pd.DataFrame | None, dict | None, str | None]:
    """Сохранённые прогнозы ретроспективной оценки. Финальная модель не вызывается:
    режим оценки показывает ровно то, что записано протоколом, или честно объясняет,
    почему показать нечего — без тихого отката к пересчёту."""
    base = Path(directory)
    csv_path, json_path = base / "evaluation_predictions.csv", base / "evaluation_report.json"
    if not csv_path.is_file() or not json_path.is_file():
        return None, None, (
            f"В каталоге `{base}` нет сохранённой оценки "
            f"(нужны `evaluation_predictions.csv` и `evaluation_report.json`). "
            f"Создать: `python -m src.backtest.evaluate --output-dir {shlex.quote(str(base))}` "
            f"или укажите другой каталог через `--evaluation-dir`.")
    try:
        df = pd.read_csv(csv_path)
        report = json.loads(json_path.read_text())
    except Exception as exc:
        return None, None, f"Файлы оценки в `{base}` не читаются: {exc}"
    missing = EVAL_CSV_COLUMNS - set(df.columns)
    if missing:
        return None, None, (f"Схема `{csv_path.name}` не совпадает с docs/EVALUATION.md: "
                            f"нет колонок {sorted(missing)}.")
    if df.empty:
        return None, None, f"`{csv_path.name}` пуст — оценивать нечего."

    try:
        if not isinstance(report, dict) or not isinstance(report.get("periods", {}), dict):
            raise ValueError("отчёт и periods должны быть JSON-объектами")
        period = report.get("periods", {}).get("evaluate", ["2025-12-01", "2026-01-31"])
        if not isinstance(period, list) or len(period) != 2:
            raise ValueError("periods.evaluate должен содержать две даты YYYY-MM-DD")
        ev_lo, ev_hi = (date.fromisoformat(value) for value in period)
        if ev_lo > ev_hi:
            raise ValueError("начало periods.evaluate позже окончания")
    except (TypeError, ValueError) as exc:
        return None, None, f"Некорректный `{json_path.name}`: {exc}"

    try:
        df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", errors="raise")
        if df["datetime"].isna().any() or df["datetime"].dt.tz is not None:
            raise ValueError("нужны целевые часы без пропусков в локальном времени")
    except (TypeError, ValueError) as exc:
        return None, None, f"Некорректная колонка `datetime` в `{csv_path.name}`: {exc}"
    if not pd.api.types.is_bool_dtype(df["target_eligible"].dtype):
        return None, None, (f"Некорректная колонка `target_eligible` в `{csv_path.name}`: "
                            "ожидаются только True или False без пропусков.")
    for column in ("power_pred", "power_baseline", "power_true", "power_p10", "power_p90"):
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
            values = df[column].to_numpy(dtype=float)
            invalid = ~np.isfinite(values)
            if column not in ("power_pred", "power_baseline"):
                invalid &= ~np.isnan(values)  # отсутствие факта/интервала разрешено
            if invalid.any():
                raise ValueError(f"{int(invalid.sum())} значений NaN/∞ недопустимы")
        except (TypeError, ValueError) as exc:
            return None, None, f"Некорректная колонка `{column}` в `{csv_path.name}`: {exc}"
    return df, report, None


def _recompute_metrics(rows: pd.DataFrame) -> dict:
    """Метрики строго из выбранных CSV-строк: только наблюдаемый факт, без заполнения."""
    observed = rows[rows["power_true"].notna()]
    err = observed["power_pred"] - observed["power_true"]
    base = observed["power_baseline"] - observed["power_true"]
    out = {"n": int(len(observed)), "n_missing_truth": int(len(rows) - len(observed))}
    if len(observed):
        out["mae"] = float(err.abs().mean())
        out["rmse"] = float((err ** 2).mean() ** 0.5)
        out["mae_baseline"] = float(base.abs().mean())
    band = observed[observed["power_p10"].notna() & observed["power_p90"].notna()]
    out["n_band"] = int(len(band))
    if len(band):
        out["coverage"] = float(((band["power_true"] >= band["power_p10"])
                                 & (band["power_true"] <= band["power_p90"])).mean())
    return out


@st.cache_data(show_spinner=False)
def _train_report() -> dict:
    from src.config import ARTIFACTS
    path = ARTIFACTS / "train_report.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _issue_dates() -> list[str]:
    return sorted(p.name[len("forecast_t1_"):-len(".csv")]
                  for p in FORECASTS.glob("forecast_t1_*.csv"))


# ---------------------------------------------------------------- граф

STATUS_STYLE = {"ok": (GREEN, "✓"), "error": ("#B3261E", "✕"), "retry": (AMBER, "↻"),
                "invalid": ("#B3261E", "✕"), "fallback": (AMBER, "↻"),
                "rejected": (AMBER, "!")}


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
              "↻ Валидация не прошла с первой попытки — см. попытку восстановления в журнале</div>")
    st.markdown(banner + "<div style='display:flex;gap:8px;flex-wrap:wrap'>"
                + "".join(cells) + "</div>", unsafe_allow_html=True)
    st.caption("При неудачной валидации: recover_baseline → validate_forecast (одна попытка). "
               "Публикация разрешена только после успешной проверки. "
               "Ниже — сохранённые события выбранного запуска.")
    if trace and trace.get("completed") is False:
        st.warning("Текущий запуск не завершён. Ранее сохранённые CSV этой даты "
                   "не считаются результатом текущего запуска.")
    if trace and trace.get("fallback"):
        st.info("LLM не завершил день: граф продолжен детерминированно с сохранённого шага.")
    with st.expander("Структурированный журнал шагов"):
        st.json(trace)


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
        title=dict(text=title, font=dict(size=15), x=0, xanchor="left",
                   y=0.97, yanchor="top", yref="container"),
        height=370, margin=dict(l=8, r=8, t=86, b=8), hovermode="x unified",
        yaxis=dict(title="норм. мощность", range=[0, 1.04], gridcolor="#EEF2F7"),
        xaxis=dict(gridcolor="#EEF2F7", tickformat="%d.%m %H:%M"), plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=11)),
        font=dict(family="system-ui", color=INK))
    return fig


def evaluation_chart(rows: pd.DataFrame, title: str) -> go.Figure:
    """Один выпуск из сохранённой оценки: факт, прогноз, baseline и интервал P10–P90."""
    rows = rows.sort_values("datetime")
    x = rows["datetime"]
    fig = go.Figure()
    if rows["power_p10"].notna().any() and rows["power_p90"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=rows["power_p90"], mode="lines",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=rows["power_p10"], mode="lines",
                                 line=dict(width=0), fill="tonexty", fillcolor=BAND,
                                 name="Интервал P10–P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=rows["power_baseline"], mode="lines",
                             line=dict(color=MUTED, width=1.6, dash="dot"),
                             name="Физический baseline",
                             hovertemplate="%{x|%d.%m %H:%M}<br>baseline %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=rows["power_pred"], mode="lines",
                             line=dict(color=ACCENT, width=2.4), name="Прогноз ансамбля",
                             hovertemplate="%{x|%d.%m %H:%M}<br>прогноз %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=rows["power_true"], mode="lines",
                             line=dict(color=INK, width=2), name="Факт",
                             connectgaps=False,
                             hovertemplate="%{x|%d.%m %H:%M}<br>факт %{y:.3f}<extra></extra>"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0, xanchor="left",
                   y=0.97, yanchor="top", yref="container"),
        height=370, margin=dict(l=8, r=8, t=86, b=8), hovermode="x unified",
        yaxis=dict(title="норм. мощность", range=[0, 1.04], gridcolor="#EEF2F7"),
        xaxis=dict(gridcolor="#EEF2F7", tickformat="%d.%m %H:%M"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=11)),
        font=dict(family="system-ui", color=INK))
    return fig


def wind_chart(slice_: pd.DataFrame) -> go.Figure:
    """По одной кривой на источник: высота ступицы (100 м), иначе лучшее доступное.

    Часть моделей отдаёт и 100 м, и 10 м — без выбора источник попадал бы в легенду дважды.
    """
    sources = sorted({c.split("__")[0] for c in slice_.columns if "__" in c})
    cols: dict[str, str] = {}
    for src in sources:
        for var in ("wind_speed_100m", "wind_speed_120m", "wind_speed_80m", "wind_speed_10m"):
            col = f"{src}__{var}"
            if col in slice_.columns and slice_[col].notna().any():
                cols[f"{src} ({var.rsplit('_', 1)[1]})"] = col
                break

    fig = go.Figure()
    for label, col in cols.items():
        fig.add_trace(go.Scatter(x=slice_.index, y=slice_[col], mode="lines", name=label,
                                 line=dict(width=1.3, color=MUTED), opacity=.55,
                                 hovertemplate=f"{label}: %{{y:.1f}} м/с<extra></extra>"))
    if cols:
        fig.add_trace(go.Scatter(x=slice_.index, y=slice_[list(cols.values())].mean(axis=1),
                                 mode="lines", name="среднее ансамбля",
                                 line=dict(color=ACCENT, width=2.4),
                                 hovertemplate="ансамбль: %{y:.1f} м/с<extra></extra>"))
    fig.update_layout(
        title=dict(text="Прогноз ветра: ансамбль NWP-источников", font=dict(size=15),
                   x=0, xanchor="left", y=0.97, yanchor="top", yref="container"),
        height=330, margin=dict(l=8, r=8, t=86, b=8), hovermode="x unified",
        yaxis=dict(title="скорость ветра, м/с", gridcolor="#EEF2F7"),
        xaxis=dict(gridcolor="#EEF2F7", tickformat="%d.%m %H:%M"), plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=10)),
        font=dict(family="system-ui", color=INK))
    return fig


# ---------------------------------------------------------------- страница

st.markdown("## 🌬️ WindCast — панель оператора ВЭС")
st.caption("Шелекский коридор, Алматинская область · две турбины · горизонт 24–48 часов · "
           "вход — архивные прогнозы погоды, доступные на момент выпуска")

with st.sidebar:
    st.markdown("### Режим")
    st.caption(f"Каталог результатов: {FORECASTS}")
    mode = st.radio("Режим", ["Тестовый период (февраль 2026)",
                              "Ретроспективная оценка (есть факт)"],
                    label_visibility="collapsed", key="mode")
    turbine = st.selectbox("Турбина", list(TURBINES),
                           format_func=lambda t: f"Турбина {t}", key="turbine")
    eval_mode = not mode.startswith("Тестовый")

    if not eval_mode:
        dates = _issue_dates()
        issue = st.select_slider("Дата выпуска прогноза", dates,
                                 value=dates[len(dates) // 2]) if dates else None
    else:
        st.caption(f"Каталог оценки: {EVALUATION_DIR}")
        issue = None

    st.divider()
    lat, lon = TURBINES[turbine]
    st.markdown(f"**Координаты**  \n`{lat}, {lon}`")
    rep = _train_report().get(str(turbine), {})
    if rep:
        st.markdown(f"**MAE при подборе модели**  \n`{rep['lightgbm']['mae']:.4f}`  \n"
                    f"0–24 ч `{rep['by_lead']['1']['mae']:.4f}` · "
                    f"24–48 ч `{rep['by_lead']['2']['mae']:.4f}`")
        st.caption("Метрики настройки (early stopping, бленд), не независимая оценка.")
    st.button("Обновить результаты", on_click=st.cache_data.clear)

# ================================================================ режим оценки
if eval_mode:
    eval_df, eval_report, eval_error = _evaluation_data(str(EVALUATION_DIR))
    if eval_error:
        st.error(eval_error)
        st.stop()

    periods = eval_report.get("periods", {})
    trained_through = eval_report.get("trained_through", "?")
    ev_lo, ev_hi = periods.get("evaluate", ["2025-12-01", "2026-01-31"])
    st.info(f"**Ретроспективная оценка, не нетронутый тест.** Модель обучена на целях "
            f"по **{trained_through}** и предсказывала период {ev_lo} — {ev_hi} по датам "
            f"выпуска, тем же путём, что и февральская подача. Декабрь–январь ранее "
            f"влияли на выбор погодных источников. Февральского факта у команды нет. "
            f"Ниже — сохранённые прогнозы протокола, финальная модель не пересчитывается.")

    scope = st.radio("Какие целевые часы", ["Чистые цели (без простоев)", "Все наблюдаемые часы"],
                     horizontal=True, key="eval_scope")
    clean_only = scope.startswith("Чистые")

    # сводка за весь период: пересчёт из строк CSV, а не из чисел в JSON
    scoped = eval_df[eval_df["target_eligible"]] if clean_only else eval_df
    rows_summary = []
    for (t, lead), grp in scoped.groupby(["turbine", "lead_day"]):
        m = _recompute_metrics(grp)
        rows_summary.append({"Турбина": t, "Горизонт": f"{(lead - 1) * 24}–{lead * 24} ч",
                             "MAE": round(m.get("mae", float("nan")), 4),
                             "RMSE": round(m.get("rmse", float("nan")), 4),
                             "MAE baseline": round(m.get("mae_baseline", float("nan")), 4),
                             "Часов-выпусков": m["n"]})
    st.markdown("#### Сводка за период оценки")
    st.dataframe(pd.DataFrame(rows_summary), hide_index=True, use_container_width=True)
    total = _recompute_metrics(scoped)
    cov_note = (f"Измеренное покрытие P10–P90: **{total['coverage']:.1%}** "
                f"на {total['n_band']} строках при номинальной ширине 80% — интервал "
                f"недокалиброван." if total.get("coverage") is not None
                else "Интервал P10–P90 в выбранных строках отсутствует.")
    st.caption(f"{total['n']} строк с фактом, {total['n_missing_truth']} без наблюдений "
               f"(не заполняются). {cov_note} Один целевой час встречается в двух строках "
               f"(lead 1 и lead 2) — они не усредняются.")

    # дневной просмотр: календарная дата + явный выбор горизонта => один выпуск
    st.markdown("#### Один выпуск подробно")
    csel1, csel2 = st.columns([1, 2])
    with csel1:
        lead = st.radio("Горизонт", [1, 2], key="eval_lead", horizontal=True,
                        format_func=lambda l: f"{(l - 1) * 24}–{l * 24} ч (lead {l})")
    with csel2:
        min_day, max_day = date.fromisoformat(ev_lo), date.fromisoformat(ev_hi)
        default_day = min(max(date(2026, 1, 14), min_day), max_day)
        day = st.date_input("Целевая дата", value=default_day, key="eval_date",
                            min_value=min_day, max_value=max_day)

    day_rows = eval_df[(eval_df["turbine"] == turbine)
                       & (eval_df["lead_day"] == lead)
                       & (eval_df["datetime"].dt.date == day)]
    if day_rows.empty:
        first_l2 = eval_df[eval_df["lead_day"] == 2]["datetime"].min()
        if lead == 2 and pd.Timestamp(day) < first_l2:
            st.warning(f"Для {day} нет строк lead 2 — их выпуск датировался бы раньше "
                       f"отсечки обучения модели ({trained_through}) и исключён из "
                       f"протокола. Первая дата с lead 2: {first_l2:%Y-%m-%d}. "
                       f"Выберите lead 1 или другую дату.")
        else:
            st.warning(f"На {day} (lead {lead}, турбина {turbine}) строк в сохранённой "
                       f"оценке нет.")
        st.stop()

    issue_of_day = day_rows["issue_date"].iloc[0]
    eligible = day_rows[day_rows["target_eligible"]]
    day_m = _recompute_metrics(eligible if clean_only else day_rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Дата выпуска", issue_of_day)
    c2.metric("MAE дня" + (" (чистые цели)" if clean_only else " (все часы)"),
              f"{day_m['mae']:.3f}" if day_m.get("mae") is not None else "—")
    c3.metric("Часов с фактом", f"{day_m['n']} из {len(day_rows)}")
    c4.metric("Отсечка обучения", str(day_rows["trained_through"].iloc[0]))

    st.plotly_chart(evaluation_chart(
        day_rows, f"Турбина {turbine}, {day}: выпуск {issue_of_day}, lead {lead} — "
                  f"сохранённая ретроспективная оценка"), use_container_width=True)
    if day_rows["power_p10"].isna().all():
        st.caption("Интервал P10–P90 для этих строк не записан — полоса не показана.")

    # погодный overlay строго из того же выпуска
    try:
        from src.weather.openmeteo import get_issued_forecast
        sl = get_issued_forecast(str(issue_of_day), _weather())
        sl = sl[sl.index.normalize() == pd.Timestamp(day)]
        st.plotly_chart(wind_chart(sl), use_container_width=True)
        st.caption(f"Прогноз ветра из выпуска {issue_of_day} — тот же вход, что видела "
                   f"модель; измеренная на турбине погода сюда не подмешивается.")
    except Exception as exc:
        st.info(f"Погодный срез выпуска недоступен: {exc}")

    st.divider()
    st.caption(f"Источник: {EVALUATION_DIR}/evaluation_predictions.csv и "
               f"evaluation_report.json (протокол {eval_report.get('protocol', '?')}). "
               f"Метрики пересчитаны из строк CSV. Полное описание: docs/EVALUATION.md "
               f"и docs/DASHBOARD.md.")
    st.stop()

# ============================================================ февральский режим
if not issue:
    st.warning("Прогнозы не найдены. Сначала выполните "
               "`python -m src.cli run-agent --start 2026-01-31 --end 2026-02-27 "
               f"--no-llm --output-dir {shlex.quote(str(FORECASTS))}`.")
    st.stop()

current_trace = _trace(issue)
if current_trace and current_trace.get("completed") is False:
    st.error("Прогноз этого запуска недоступен: выполнение не завершилось успешно.")
    render_graph(current_trace)
    st.stop()

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
