"""
pages/page_portfolio.py — Экран 4 «Портфель»
"""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

BASE       = Path(__file__).parent.parent
RULES_PATH = BASE / "data" / "processed" / "portfolio_selector_rules.json"
OFZ_PATH   = BASE / "data" / "raw" / "ofz_list.csv"

RED      = "#FF0508"
CHARCOAL = "#33373B"
LAVENDER = "#E8E8F4"
SKY_BLUE = "#5FD2FF"
WHITE    = "#FFFFFF"

PIE_COLORS = [CHARCOAL, SKY_BLUE, RED, LAVENDER,
              "rgba(95,210,255,0.6)", "rgba(51,55,59,0.6)"]


def dur_str(d: float) -> str:
    """Правильное склонение: 1 год, 2 года, 5 лет."""
    d = int(d)
    if d == 0:
        return "0 лет"
    if d % 10 == 1 and d % 100 != 11:
        return f"{d} год"
    elif d % 10 in (2, 3, 4) and d % 100 not in (12, 13, 14):
        return f"{d} года"
    else:
        return f"{d} лет"


def _load_rules() -> dict:
    if RULES_PATH.exists():
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _get_portfolio_for_prob(prob: float) -> tuple:
    rules = _load_rules()
    if not rules:
        return _fallback_portfolio(prob)

    signal_label, dur_target = "WAIT", 0.0
    for rule in rules.get("signal_rules", []):
        lo, hi = [float(x) for x in rule["p_range"].split("–")]
        if lo <= prob < hi:
            signal_label = rule["signal"]
            dur_target   = rule["dur_target"]
            break

    # WAIT — сразу возвращаем флоатеры
    if signal_label == "WAIT":
        df = pd.DataFrame([
            {"ISIN": "ОФЗ-ПК",   "Выпуск": "Флоатеры (ОФЗ-ПК)",
             "Купон (%)": "RUONIA+", "Дюрация (лет)": "<0.5", "Доля (%)": 60},
            {"ISIN": "ДЕНРЫНОК", "Выпуск": "Денежный рынок",
             "Купон (%)": "≈КС",    "Дюрация (лет)": "<0.1", "Доля (%)": 40},
        ])
        return signal_label, 0.0, 0.0, df

    # Ищем ближайший пример с портфелем
    examples     = rules.get("examples", [])
    examples_buy = [e for e in examples if e.get("portfolio")]
    best_example = min(examples_buy, key=lambda e: abs(e["p"] - prob))
    portfolio    = best_example.get("portfolio", [])
    actual_dur   = best_example.get("actual_duration", dur_target)

    if portfolio:
        df = pd.DataFrame(portfolio).rename(columns={
            "SECID":          "ISIN",
            "SHORTNAME":      "Выпуск",
            "COUPONPERCENT":  "Купон (%)",
            "duration_years": "Дюрация (лет)",
            "weight_pct":     "Доля (%)",
        })
        df = df[["ISIN", "Выпуск", "Купон (%)", "Дюрация (лет)", "Доля (%)"]]
    else:
        df = pd.DataFrame([
            {"ISIN": "ОФЗ-ПК",   "Выпуск": "Флоатеры (ОФЗ-ПК)",
             "Купон (%)": "RUONIA+", "Дюрация (лет)": "<0.5", "Доля (%)": 60},
            {"ISIN": "ДЕНРЫНОК", "Выпуск": "Денежный рынок",
             "Купон (%)": "≈КС",    "Дюрация (лет)": "<0.1", "Доля (%)": 40},
        ])
    return signal_label, dur_target, actual_dur, df


def _load_ofz_table() -> pd.DataFrame:
    if OFZ_PATH.exists():
        df = pd.read_csv(OFZ_PATH)
        return df[df["is_active"] == True].copy()
    return pd.DataFrame()


def _fallback_portfolio(prob: float) -> tuple:
    ofz = _load_ofz_table()
    if ofz.empty:
        df = pd.DataFrame([{"ISIN": "—", "Выпуск": "Нет данных",
                             "Купон (%)": "—", "Дюрация (лет)": "—", "Доля (%)": 100}])
        return "—", 0.0, 0.0, df

    def _pick(bucket, n, shares):
        sub = (ofz[ofz["bucket"] == bucket]
               .sort_values("duration_years", ascending=False)
               .head(n).reset_index(drop=True))
        sub["Доля (%)"] = shares[:len(sub)]
        return sub[["SECID", "SHORTNAME", "COUPONPERCENT", "duration_years", "Доля (%)"]].rename(
            columns={"SECID": "ISIN", "SHORTNAME": "Выпуск",
                     "COUPONPERCENT": "Купон (%)", "duration_years": "Дюрация (лет)"})

    if prob >= 0.80:
        df, label, d = _pick("long", 5, [20,20,20,20,20]), "STRONG BUY", 9.0
    elif prob >= 0.65:
        df, label, d = _pick("long", 4, [25,25,25,25]),   "BUY",        6.0
    elif prob >= 0.50:
        df, label, d = _pick("mid",  3, [33,33,34]),      "WEAK BUY",   4.0
    else:
        df = pd.DataFrame([
            {"ISIN": "ОФЗ-ПК",   "Выпуск": "Флоатеры",
             "Купон (%)": "RUONIA+", "Дюрация (лет)": "<0.5", "Доля (%)": 60},
            {"ISIN": "ДЕНРЫНОК", "Выпуск": "Денежный рынок",
             "Купон (%)": "≈КС",    "Дюрация (лет)": "<0.1", "Доля (%)": 40},
        ])
        label, d = "WAIT", 0.0

    actual = df["Дюрация (лет)"].apply(
        lambda x: float(x) if str(x).replace(".", "").isdigit() else 0).mean()
    return label, d, actual, df


def _duration_gauge(dur_target: float, actual_dur: float) -> None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(actual_dur, 1),
        number={"suffix": " лет", "font": {"size": 36}},
        title={"text": f"Оптимальная дюрация {dur_str(dur_target)}"},
        gauge={
            "axis": {"range": [0, 12], "ticksuffix": " л"},
            "bar":  {"color": CHARCOAL},
            "steps": [
                {"range": [0,  3], "color": LAVENDER},
                {"range": [3,  7], "color": "rgba(95,210,255,0.25)"},
                {"range": [7, 12], "color": "rgba(95,210,255,0.55)"},
            ],
            "threshold": {
                "line": {"color": RED, "width": 3},
                "value": dur_target,
            },
        },
    ))


def render():
    st.header("Портфель", divider="gray")

    prob = st.slider("Вероятность", 0.0, 1.0, 0.72, 0.01, format="%.2f")

    has_json                          = RULES_PATH.exists()
    signal_label, dur_target, actual_dur, df = _get_portfolio_for_prob(prob)

    _duration_gauge(dur_target, actual_dur)
    st.divider()

    # ── Сигнал — без фона, просто текст ──────────────────────────────────────
    if signal_label in ("BUY", "STRONG BUY"):
        st.markdown(f"**{signal_label}** — целевая дюрация {dur_str(dur_target)}")
    elif signal_label == "WEAK BUY":
        st.markdown(f"**{signal_label}** — целевая дюрация {dur_str(dur_target)}")
    else:
        st.markdown("**WAIT** — флоатеры и денежный рынок")

    st.write("")

    # ── Таблица + пирог ───────────────────────────────────────────────────────
    col_table, col_pie = st.columns([3, 2])



    with col_pie:
        shares  = df["Доля (%)"].tolist()
        labels  = df["Выпуск"].tolist()
        numeric = [(l, s) for l, s in zip(labels, shares)
                   if isinstance(s, (int, float))]
        if numeric:
            fig_pie = go.Figure(go.Pie(
                labels=[x[0] for x in numeric],
                values=[x[1] for x in numeric],
                hole=0.4,
                marker_colors=PIE_COLORS[:len(numeric)],
            ))


    st.divider()

    st.subheader("Сигнал — портфель")
    st.markdown("""
    | P(BUY) | Сигнал | Дюрация | Инструменты |
    |---|---|---|---|
    | 0.00–0.50 | WAIT | 0 лет | Флоатеры / денежный рынок |
    | 0.50–0.65 | WEAK BUY | 4 года | Среднесрочные ОФЗ-ПД (3–5 лет) |
    | 0.65–0.80 | BUY | 6 лет | Долгосрочные ОФЗ-ПД (5–7 лет) |
    | 0.80–1.00 | STRONG BUY | 9 лет | Наиболее долгосрочные ОФЗ-ПД (>7 лет) |
        """)

    st.divider()
