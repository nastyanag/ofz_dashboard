

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

MASTER = Path("data/raw/master_daily.csv")
FW = Path("data/processed/features_weekly.csv")

# ── Палитра ПУЛЬС MOEX ────────────────────────────────────────────────────────
RED = "#FF0508"
CHARCOAL = "#33373B"
LAVENDER = "#E8E8F4"
SKY_BLUE = "#5FD2FF"
WHITE = "#FFFFFF"


def _current_defaults() -> dict:
    if FW.exists():
        df = pd.read_csv(FW, parse_dates=["date"])
        row = df.dropna(subset=["key_rate"]).iloc[-1]
    elif MASTER.exists():
        df = pd.read_csv(MASTER, parse_dates=["date"])
        row = df.dropna(subset=["key_rate"]).iloc[-1]
    else:
        return {"ks": 14.5, "d3m": -1.5, "d6m": -3.0, "inflation": 8.0,
                "rgbi_trend": 4.0, "phase": 1.0, "rhetoric": 0.0}
    return {
        "ks":        float(row.get("key_rate",          14.5)),
        "d3m":       float(row.get("delta_ks_3m",       -1.5) or -1.5),
        "d6m":       float(row.get("delta_ks_6m",       -3.0) or -3.0),
        "inflation": float(row.get("inflation_yoy",      8.0) or  8.0),
        "rgbi_trend":float(row.get("rgbi_log_ret",       0.0) or  0.0) * 100,
        "phase":     float(row.get("cycle_phase",        0.0) or  0.0),
        "rhetoric":  float(row.get("cbr_rhetoric_score", 0.0) or  0.0),
    }


def formula_signal(ks, d3m, d6m, inflation, rgbi_trend, phase=0.0, rhetoric=0.0) -> float:
    real_rate = ks - inflation
    s = np.clip((ks - 8) / 13,      0, 1) * 0.28
    s += np.clip(-d3m / 6,           0, 1) * 0.22
    s += np.clip(-d6m / 9,           0, 1) * 0.15
    s += np.clip(real_rate / 10,     0, 1) * 0.15
    s += np.clip(rgbi_trend / 10,    0, 1) * 0.05
    s += np.clip((phase + 1) / 2,    0, 1) * 0.10
    s += np.clip((rhetoric + 1) / 2, 0, 1) * 0.05
    return float(np.clip(s, 0.0, 1.0))


def render():
    st.header("Определить момент покупки ", divider="gray")


    cur = _current_defaults()

    st.subheader("Параметры")
    col1, col2 = st.columns(2)

    with col1:
        ks = st.slider("Ключевая ставка ЦБ РФ (%)", 4.0, 22.0,
                       float(round(cur["ks"] * 4) / 4), 0.25,
                       help=f"Текущее: {cur['ks']:.2f}%")
        delta_3m = st.slider("Изменение КС за 3 мес (пп)", -6.0, 4.0,
                             float(max(-6.0, min(4.0, round(cur["d3m"] * 4) / 4))), 0.25,
                             help="Отрицательное = снижение ставки")
        delta_6m = st.slider("Изменение КС за 6 мес (пп)", -10.0, 6.0,
                             float(max(-10.0, min(6.0, round(cur["d6m"] * 4) / 4))), 0.25)

    with col2:
        inflation = st.slider("Инфляция (%)", 3.0, 15.0,
                              float(max(3.0, min(15.0, round(cur["inflation"] * 2) / 2))), 0.5)
        rgbi_trend = st.slider("Динамика RGBI за 3 мес (%)", -10.0, 10.0,
                               float(max(-10.0, min(10.0, round(cur["rgbi_trend"] * 2) / 2))), 0.5,
                               help="Положительная = цены ОФЗ растут")
        phase_labels = {-1.0: "Ужесточение (-1)", 0.0: "Пауза (0)", 1.0: "Смягчение (+1)"}
        phase_sel = st.select_slider("Фаза цикла КС",
                                     options=[-1.0, 0.0, 1.0],
                                     value=float(round(cur["phase"])),
                                     format_func=lambda x: phase_labels[x])
        rhetoric = st.slider("Риторика ЦБ", -1.0, 1.0,
                             float(round(cur["rhetoric"] * 10) / 10), 0.1,
                             help="−1 жёсткая, 0 нейтральная, +1 мягкая")

    prob = formula_signal(ks, delta_3m, delta_6m, inflation, rgbi_trend,
                               float(phase_sel), rhetoric)
    real_rate = ks - inflation

    # Цвета и метки строго из шаблона
    if prob >= 0.65:
        color, border, bg = SKY_BLUE, SKY_BLUE, "rgba(95,210,255,0.12)"
        label = "BUY \U00002705"
        rec = f"Рекомендация: **{"долгосрочные ОФЗ-ПД (дюрация > 7 лет)"}**"
    elif prob >= 0.5:
        color, border, bg = CHARCOAL, CHARCOAL, LAVENDER
        label = "WEAK BUY  \N{thumbs up sign}"
        rec = f"Рекомендация: **{"среднесрочные ОФЗ-ПД (дюрация 3–7 лет)"}**"
    else:
        color, border, bg = RED, RED, "rgba(255,5,8,0.08)"
        label = "WAIT \u274C"
        rec = f"Рекомендация: **{"ОФЗ - ПД неоптимальны к покупке"}**"

    st.divider()

    # ── Результат ─────────────────────────────────────────────────────────────
    r1, r2, r3 = st.columns(3)
    r1.metric("",          label)
    r2.metric("Вероятность",   f"{prob:.0%}")
    r3.metric("Реальная % ставка",   f"{real_rate:.1f}%")

    # st.markdown(
    #     f'<div style="background:{bg};border-left:5px solid {border};'
    #     f'padding:14px 20px;border-radius:6px;margin-top:12px;">'
    #     f'<b style="color:{color}">{label}</b> — {rec}</div>',
    #     unsafe_allow_html=True,
    # )
    st.write(rec)

    st.divider()
