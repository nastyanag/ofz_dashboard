import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

MASTER = Path(__file__).parent.parent / "data" / "raw" / "master_daily.csv"
FW = Path(__file__).parent.parent / "data" / "processed" / "features_weekly.csv"

RED = "#FF0508"
CHARCOAL = "#33373B"
LAVENDER = "#E8E8F4"
SKY_BLUE = "#5FD2FF"
WHITE = "#FFFFFF"


def _is_dark() -> bool:
    return st.session_state.get("theme", "light") == "dark"


def _theme_line_color() -> str:
    return WHITE if _is_dark() else CHARCOAL


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_fw() -> pd.DataFrame:
    df = pd.read_csv(FW, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def render():
    st.header("Данные", divider="gray")

    if not MASTER.exists():
        st.error("Файл data/raw/master_daily.csv не найден.")
        return

    df = load_master()

    # ── Фильтр периода ────────────────────────────────────────────────────────
    PERIOD_OPTIONS = {
        "Весь период": ("2013-01-01", "2026-12-31"),
        "2016–2019":   ("2016-01-01", "2019-12-31"),
        "2019–2022":   ("2019-01-01", "2022-12-31"),
        "2022–2026":   ("2022-01-01", "2026-12-31"),
    }
    period = st.radio(
        "Период",
        list(PERIOD_OPTIONS.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="period_selector_v2",
    )
    date_from = pd.Timestamp(PERIOD_OPTIONS[period][0])
    date_to   = pd.Timestamp(PERIOD_OPTIONS[period][1])

    # ── Все датафреймы фильтруются по date_from И date_to ────────────────────
    def filt(d, col=None):
        if col:
            d = d.dropna(subset=[col] if isinstance(col, str) else col)
        return d[(d["date"] >= date_from) & (d["date"] <= date_to)].reset_index(drop=True)

    df_ks    = filt(df, "key_rate")
    df_rgbi  = filt(df, ["rgbi", "rgbitr"])
    df_ruonia= filt(df, "ruonia")
    df_yld   = filt(df, ["y_1y", "y_2y", "y_5y", "y_10y"])

    if len(df_ks) < 2:
        st.warning("Недостаточно данных для выбранного периода.")
        return

    last      = df_ks.iloc[-1]
    prev      = df_ks.iloc[-2]
    last_rgbi = df_rgbi.iloc[-1] if len(df_rgbi) > 0 else None

    # ── KPI ───────────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("КС (на конец периода)",
                f"{last['key_rate']:.2f}%",
                delta=f"{last['key_rate'] - prev['key_rate']:.2f}%")
    col2.metric("КС (пик периода)",
                f"{df_ks['key_rate'].max():.2f}%")
    col3.metric("Δ КС 3 мес",
                f"{last['delta_ks_3m']:+.2f} пп"
                if pd.notna(last['delta_ks_3m']) else "—")
    col4.metric("Реальная ставка",
                f"{last['real_rate']:.1f}%"
                if pd.notna(last['real_rate']) else "—")
    col5.metric("RGBI",   f"{last_rgbi['rgbi']:.1f}" if last_rgbi is not None else "—")
    col6.metric("RGBITR", f"{last_rgbi['rgbitr']:.1f}" if last_rgbi is not None else "—")

    st.divider()

    # ── График 1: КС + RUONIA + фазы цикла ───────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Ключевая ставка ЦБ РФ и RUONIA, %",
            "Индекс RGBI и RGBITR ",
        ),
        vertical_spacing=0.10,
        row_heights=[0.45, 0.55],
    )

    # Фазы цикла — фильтруем fw по обеим датам
    if FW.exists():
        fw = load_fw()
        fw = fw[(fw["date"] >= date_from) & (fw["date"] <= date_to)].copy()
        fw["cycle_phase"] = fw["cycle_phase"].fillna(0)

        phase_colors = {
            -1: "rgba(232,232,244,0.35)",
             0: "rgba(95,210,255,0.10)",
             1: "rgba(95,210,255,0.22)",
        }
        for phase_val in [-1, 0, 1]:
            prev_in = False
            start   = None
            for _, row in fw.iterrows():
                in_phase = row["cycle_phase"] == phase_val
                if in_phase and not prev_in:
                    start = row["date"]
                elif not in_phase and prev_in and start is not None:
                    fig.add_vrect(x0=start, x1=row["date"],
                                  fillcolor=phase_colors[phase_val],
                                  layer="below", line_width=0, row=1, col=1)
                    start = None
                prev_in = in_phase
            if prev_in and start is not None:
                fig.add_vrect(x0=start, x1=fw["date"].iloc[-1],
                              fillcolor=phase_colors[phase_val],
                              layer="below", line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_ks["date"], y=df_ks["key_rate"],
        mode="lines", line=dict(color=RED, width=2, shape="hv"),
        name="Ключевая ставка",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_ruonia["date"], y=df_ruonia["ruonia"],
        mode="lines", line=dict(color=_theme_line_color(), width=2.5, dash="dot"),
        name="RUONIA",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_rgbi["date"], y=df_rgbi["rgbi"],
        mode="lines", line=dict(color=_theme_line_color(), width=1.5),
        name="RGBI",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_rgbi["date"], y=df_rgbi["rgbitr"],
        mode="lines", line=dict(color=SKY_BLUE, width=2),
        name="RGBITR",
    ), row=2, col=1)

    # Вертикальная метка пика только если он попадает в период
    peak_date = pd.Timestamp("2024-10-25")
    if date_from <= peak_date <= date_to:
        fig.add_vline(x="2024-10-25", line_dash="dot",
                      line_color=_theme_line_color(),
                      annotation_text="Пик КС 21%",
                      annotation_position="top right")

    fig.update_layout(
        height=520,
        template="plotly_dark" if _is_dark() else "plotly_white",
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=40, b=20),
    )
    fig.update_xaxes(dtick="M12", tickformat="%Y", tickangle=0)
    fig.update_yaxes(title_text="%",      row=1, col=1)
    fig.update_yaxes(title_text="пункты", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── График 2: Кривая доходности ───────────────────────────────────────────
    st.subheader("Кривая доходности ОФЗ")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        fig2 = go.Figure()
        for col_name, label, color in [
            ("y_1y",  "1 год",  "rgba(232,232,244,0.9)"),
            ("y_2y",  "2 года", "rgba(95,210,255,0.8)"),
            ("y_5y",  "5 лет",  "rgba(255,5,8,0.9)"),
            ("y_10y", "10 лет", _theme_line_color()),
        ]:
            fig2.add_trace(go.Scatter(
                x=df_yld["date"], y=df_yld[col_name],
                mode="lines", name=label,
                line=dict(color=color, width=1.5),
            ))
        fig2.update_layout(height=280, margin=dict(t=20, b=20),
                           template="plotly_dark" if _is_dark() else "plotly_white",
                           yaxis_title="%", legend=dict(orientation="h", y=-0.20))
        fig2.update_xaxes(dtick="M12", tickformat="%Y", tickangle=0)
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        if len(df_yld) > 0:
            last_yld   = df_yld.iloc[-1]
            date_label = last_yld["date"].date() if hasattr(last_yld["date"], "date") else last_yld["date"]
            fig3 = go.Figure(go.Scatter(
                x=[1, 2, 5, 10],
                y=[last_yld["y_1y"], last_yld["y_2y"], last_yld["y_5y"], last_yld["y_10y"]],
                mode="lines+markers",
                line=dict(color=RED, width=2),
                marker=dict(size=8, color=RED),
            ))
            fig3.update_layout(
                title=f"Кривая на {date_label}",
                height=280,
                template="plotly_dark" if _is_dark() else "plotly_white",
                xaxis_title="Срок (лет)", yaxis_title="%",
                margin=dict(t=40, b=20),
            )
            fig3.add_hline(y=last["key_rate"], line_dash="dot",
                           line_color=_theme_line_color(),
                           annotation_text=f"КС={last['key_rate']:.1f}%")
            st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── График 3: Риторика ЦБ ─────────────────────────────────────────────────
    if FW.exists():
        fw_full = load_fw()
        fw_full = fw_full[(fw_full["date"] >= date_from) & (fw_full["date"] <= date_to)].copy()

        st.subheader("Риторика ЦБ РФ")
        st.caption("−1 = жёсткая риторика, 0 = нейтральная, +1 = мягкая")

        fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Риторика ЦБ РФ", "Ключевая ставка, %"),
                             vertical_spacing=0.10, row_heights=[0.5, 0.5])

        fig4.add_trace(go.Scatter(
            x=fw_full["date"], y=fw_full["cbr_rhetoric_score"],
            mode="lines", line=dict(color=SKY_BLUE, width=1.5),
            fill="tozeroy", fillcolor="rgba(95,210,255,0.12)",
            name="Риторика ЦБ",
        ), row=1, col=1)
        fig4.add_hline(y=0, row=1, col=1, line_dash="dot",
                       line_color=_theme_line_color())

        fw_ks = fw_full.dropna(subset=["key_rate"])
        fig4.add_trace(go.Scatter(
            x=fw_ks["date"], y=fw_ks["key_rate"],
            mode="lines", line=dict(color=RED, width=2, shape="hv"),
            name="КС",
        ), row=2, col=1)

        fig4.update_layout(height=380,
                           template="plotly_dark" if _is_dark() else "plotly_white",
                           legend=dict(orientation="h", y=-0.05),
                           margin=dict(t=40, b=20))
        fig4.update_xaxes(dtick="M12", tickformat="%Y", tickangle=0)
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    def formula_signal(f: dict) -> float:
        ks = f.get("key_rate", 10.0)
        d3 = f.get("delta_ks_3m", 0.0)
        d6 = f.get("delta_ks_6m", 0.0)
        rr = f.get("real_rate", 0.0)
        cr = f.get("is_crisis", 0)
        phase = f.get("cycle_phase", 0.0)
        rhet = f.get("cbr_rhetoric_score", 0.0)

        s = np.clip((ks - 8) / 13, 0, 1) * 0.30
        s += np.clip(-d3 / 6, 0, 1) * 0.25
        s += np.clip(-d6 / 9, 0, 1) * 0.15
        s += np.clip(rr / 10, 0, 1) * 0.15
        s += np.clip((phase + 1) / 2, 0, 1) * 0.10
        s += np.clip((rhet + 1) / 2, 0, 1) * 0.05
        s *= 0.6 if cr else 1.0
        return float(np.clip(s, 0.0, 1.0))



    st.subheader("КС и Сигнал")

    src_df = pd.read_csv(FW if FW.exists() else MASTER, parse_dates=["date"])
    src_df = src_df.dropna(subset=["key_rate"]).copy()
    src_df["signal_prob"] = src_df.apply(
        lambda row: formula_signal({
            "key_rate":           row.get("key_rate")           or 10,
            "delta_ks_3m":        row.get("delta_ks_3m")        or 0,
            "delta_ks_6m":        row.get("delta_ks_6m")        or 0,
            "real_rate":          row.get("real_rate")          or 0,
            "is_crisis":          row.get("is_crisis")          or 0,
            "cycle_phase":        row.get("cycle_phase")        or 0,
            "cbr_rhetoric_score": row.get("cbr_rhetoric_score") or 0,
        }), axis=1,
    )

    fig_hist = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Сигнал P(BUY)", "Ключевая ставка, %"),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
    )
    fig_hist.add_trace(go.Scatter(
        x=src_df["date"], y=src_df["signal_prob"],
        mode="lines", line=dict(color=SKY_BLUE, width=1.5),
        name="P(BUY)", fill="tozeroy",
        fillcolor="rgba(95,210,255,0.12)",
    ), row=1, col=1)
    fig_hist.add_hline(y=0.65, row=1, col=1, line_dash="dot", line_color=SKY_BLUE,
                       annotation_text="BUY ≥ 65%", annotation_position="bottom right")
    fig_hist.add_hline(y=0.45, row=1, col=1, line_dash="dot", line_color=CHARCOAL,
                       annotation_text="WEAK BUY ≥ 45%", annotation_position="bottom right")
    fig_hist.add_trace(go.Scatter(
        x=src_df["date"], y=src_df["key_rate"],
        mode="lines", line=dict(color=RED, width=2, shape="hv"),
        name="КС",
    ), row=2, col=1)
    for event_date, event_label in [
        ("2015-01-30", "КС=17%"),
        ("2022-02-28", "Фев.2022"),
        ("2024-10-25", "Пик 21%"),
    ]:
        for row_n in [1, 2]:
            fig_hist.add_vline(
                x=event_date, line_dash="dot", line_color=CHARCOAL,
                annotation_text=event_label if row_n == 1 else "",
                annotation_position="top right",
                row=row_n, col=1,
            )
    fig_hist.update_layout(
        height=450,
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=40, b=20),
    )
    fig_hist.update_yaxes(title_text="P(BUY)", tickformat=".0%",
                          range=[0, 1.05], row=1, col=1)
    fig_hist.update_yaxes(title_text="%", row=2, col=1)
    st.plotly_chart(fig_hist, use_container_width=True)

