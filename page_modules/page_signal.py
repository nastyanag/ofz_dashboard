"""
pages/page_signal.py — Экран 2 «Текущий сигнал»
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

BASE   = Path(__file__).parent.parent
MASTER = BASE / "data" / "raw" / "master_daily.csv"
FW     = BASE / "data" / "processed" / "features_weekly.csv"

RED      = "#FF0508"
CHARCOAL = "#33373B"
LAVENDER = "#E8E8F4"
SKY_BLUE = "#5FD2FF"
WHITE    = "#FFFFFF"


def get_current_features() -> dict:
    result = {}
    if MASTER.exists():
        df  = pd.read_csv(MASTER, parse_dates=["date"])
        row = df.dropna(subset=["key_rate"]).iloc[-1]
        result.update({
            "date":          row["date"],
            "key_rate":      float(row["key_rate"]),
            "delta_ks_1m":   float(row.get("delta_ks_1m") or 0),
            "delta_ks_3m":   float(row.get("delta_ks_3m") or 0),
            "delta_ks_6m":   float(row.get("delta_ks_6m") or 0),
            "real_rate":     float(row.get("real_rate")   or 0),
            "slope_10y_2y":  float(row.get("slope_10y_2y") or 0),
            "inflation_yoy": float(row.get("inflation_yoy") or 0),
            "is_crisis":     int(row.get("is_crisis") or 0),
        })
    if FW.exists():
        fw  = pd.read_csv(FW, parse_dates=["date"])
        row = fw.dropna(subset=["key_rate"]).iloc[-1]
        result.update({
            "cycle_phase":        float(row.get("cycle_phase") or 0),
            "cbr_rhetoric_score": float(row.get("cbr_rhetoric_score") or 0),
            "weeks_since_change": int(row.get("weeks_since_change") or 0),
            "n_hikes_3m":         int(row.get("n_hikes_3m") or 0),
            "n_cuts_3m":          int(row.get("n_cuts_3m")  or 0),
            "ruonia_spread":      float(row.get("ruonia_spread") or 0),
        })
    return result


def formula_signal(f: dict) -> float:
    ks    = f.get("key_rate",          10.0)
    d3    = f.get("delta_ks_3m",        0.0)
    d6    = f.get("delta_ks_6m",        0.0)
    rr    = f.get("real_rate",          0.0)
    cr    = f.get("is_crisis",          0)
    phase = f.get("cycle_phase",        0.0)
    rhet  = f.get("cbr_rhetoric_score", 0.0)

    s  = np.clip((ks - 8) / 13,   0, 1) * 0.30
    s += np.clip(-d3 / 6,         0, 1) * 0.25
    s += np.clip(-d6 / 9,         0, 1) * 0.15
    s += np.clip(rr / 10,         0, 1) * 0.15
    s += np.clip((phase + 1) / 2, 0, 1) * 0.10
    s += np.clip((rhet + 1) / 2,  0, 1) * 0.05
    s *= 0.6 if cr else 1.0
    return float(np.clip(s, 0.0, 1.0))


def get_signal(f: dict) -> dict:
    model_path = Path(__file__).parent.parent / "models" / "main.pkl"
    model_live = False

    if model_path.exists():
        try:
            import joblib
            bundle = joblib.load(model_path)

            if isinstance(bundle, dict):
                mdl       = bundle["model"]
                scaler    = bundle["scaler"]
                feat_cols = bundle["feature_cols"]
                X_raw = pd.DataFrame(
                    [[f.get(c, 0.0) for c in feat_cols]],
                    columns=feat_cols,
                ).fillna(0)
                X_scaled = scaler.transform(X_raw)
                raw_pred = float(mdl.predict(X_scaled)[0])
                prob = float(1 / (1 + np.exp(-raw_pred * 20)))
            else:
                if hasattr(bundle, "feature_name_"):
                    feat_cols = bundle.feature_name_
                elif hasattr(bundle, "feature_names_in_"):
                    feat_cols = bundle.feature_names_in_.tolist()
                else:
                    feat_cols = [
                        "key_rate","delta_ks_1m","delta_ks_3m","delta_ks_6m",
                        "weeks_since_change","n_hikes_3m","n_cuts_3m",
                        "real_rate","real_rate_ma3","real_rate_change_3m",
                        "slope_10y_2y","slope_10y_1y","curvature_5y",
                        "y_10y","y_5y","y_1y",
                        "ruonia","ruonia_spread","ruonia_spread_ma3",
                        "brent_ret_1m","inflation_yoy","is_crisis",
                        "cycle_phase","cbr_rhetoric_score",
                    ]
                X = pd.DataFrame(
                    [[f.get(c, 0.0) for c in feat_cols]],
                    columns=feat_cols,
                ).fillna(0)
                raw_pred = float(bundle.predict(X)[0])
                prob = float(1 / (1 + np.exp(-raw_pred * 20)))

            model_live = True
        except Exception as e:
            st.caption(f"⚠️ Ошибка загрузки модели: {e}")
            prob = formula_signal(f)
    else:
        prob = formula_signal(f)

    if prob >= 0.65:
        label = "BUY✅"
        color = SKY_BLUE
        rec   = "Долгосрочные ОФЗ-ПД (дюрация >7 лет)"
        exp   = ("КС прошла пик и находится в цикле снижения. "
                 "Стратегия: зафиксировать высокую доходность")
    elif prob >= 0.5:
        label = "WEAK BUY👍"
        color = CHARCOAL
        rec   = "Среднесрочные ОФЗ-ПД (3–5 лет)"
        exp   = "Сигнал неоднозначный. "
    else:
        label = "WAIT❌"
        color = RED
        rec   = "Неоптимально покупать ОФЗ-ПД "
        exp   = "КС ещё не достигла пика или сигнал разворота слабый."

    return {"prob": prob, "label": label, "color": color,
            "rec": rec, "exp": exp, "model_live": model_live}


def render():
    st.header("Текущая вероятность", divider="gray")

    f   = get_current_features()
    sig = get_signal(f)

    if not sig["model_live"]:
        st.info("models/main.pkl не найден — используется эвристика.", icon="ℹ️")

    # ── Сигнал (без колонок — без серого фона) ────────────────────────────────
    date_str = f.get("date", "")
    if hasattr(date_str, "date"):
        date_str = date_str.date()

    st.markdown(f"### {sig['label']}")
    st.markdown(f"**Рекомендация:** {sig['rec']}")
    st.markdown(sig['exp'])
    st.caption(f"Данные на: {date_str}")

    # ── Gauge ─────────────────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sig["prob"] * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Уверенность в приобретении ОФЗ-ПД"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": sig["color"]},
            "steps": [
                {"range": [0,  45], "color": LAVENDER},
                {"range": [45, 65], "color": "rgba(95,210,255,0.25)"},
                {"range": [65,100], "color": "rgba(95,210,255,0.50)"},
            ],
            "threshold": {
                "line": {"color": CHARCOAL, "width": 3},
                "value": 65,
            },
        },
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── Таблица признаков ─────────────────────────────────────────────────────
    st.subheader("Текущие значения признаков")

    rows_base = [
        ("КС текущая (%)",         f.get("key_rate"),        "высокая → сиигнал"),
        ("Δ КС 1 мес (пп)",        f.get("delta_ks_1m"),     "< 0 → снижение → сигнал"),
        ("Δ КС 3 мес (пп)",        f.get("delta_ks_3m"),     "< 0 → снижение → сигнал"),
        ("Δ КС 6 мес (пп)",        f.get("delta_ks_6m"),     "< 0 → снижение → сигнал"),
        ("Реальная ставка (%)",    f.get("real_rate"),        "высокая → сигнал"),
        ("Наклон кривой 10y-2y",   f.get("slope_10y_2y"),    "< 0 = инверсия"),
    ]

    rows_fw = [
        ("Риторика ЦБ (score)",     f.get("cbr_rhetoric_score"), "> 0 = мягкая → сигнал"),
        ("Недель без изменения КС", f.get("weeks_since_change"),  "долгая пауза → сигнал"),
        ("Повышений за 3 мес",      f.get("n_hikes_3m"),          "0 → сигнал"),
        ("Снижений за 3 мес",       f.get("n_cuts_3m"),           "> 0 → сигнал"),
        ("Спред RUONIA–КС",         f.get("ruonia_spread"),       "< 0 = рынок ждёт снижения"),
    ]

    def fmt(val):
        if val is None: return "—"
        if isinstance(val, str): return val
        if isinstance(val, float) and pd.notna(val): return f"{val:+.2f}"
        if isinstance(val, (int, float)) and pd.notna(val): return str(int(val))
        return "—"

    feat_df = pd.DataFrame([
        {"Признак": name, "Значение": fmt(val), "Интерпретация": hint}
        for name, val, hint in rows_base + rows_fw
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.divider()
