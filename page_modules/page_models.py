"""
pages/page_models.py — Экран «Выбор модели»
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

BASE    = Path(__file__).parent.parent
MC_PATH = BASE / "data" / "processed" / "model_comparison.csv"
BM_PATH = BASE / "data" / "processed" / "baseline_metrics.csv"
LM_PATH = BASE / "data" / "processed" / "lgbm_metrics.csv"
MM_PATH = BASE / "data" / "processed" / "mlp_metrics.csv"
SV_PATH = BASE / "data" / "processed" / "shap_values_last_fold.csv"
BC_PATH = BASE / "data" / "processed" / "baseline_coefficients.csv"

RED      = "#FF0508"
CHARCOAL = "#33373B"
LAVENDER = "#E8E8F4"
SKY_BLUE = "#5FD2FF"


def get_da(df):
    """Находит колонку с DA независимо от названия."""
    for col in ["DA", "DirectAcc", "Accuracy"]:
        if col in df.columns:
            return df[col].tolist()
    return [0.5] * len(df)


def render():
    st.header("Выбор модели", divider="gray")

    st.subheader("Три модели — простое объяснение")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
<div style="background:rgba(232,232,244,0.5);
            padding:16px;border-radius:8px;">
  <div style="font-size:1.1rem;font-weight:700;color:{CHARCOAL};">Ridge (бейзлайн)</div>
  <div style="font-size:0.85rem;margin-top:10px;color:{CHARCOAL};">
    Линейная регрессия с L2-регуляризацией. Предполагает аддитивную зависимость
    таргета от признаков без учёта нелинейных взаимодействий.
    <br><br>
    <b>Плюс:</b> интерпретируемость коэффициентов, устойчивость к мультиколлинеарности<br>
    <b>Минус:</b> не улавливает совместный эффект признаков — например, carry × наклон кривой
  </div>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div style="background:rgba(95,210,255,0.25);
            padding:16px;border-radius:8px;">
  <div style="font-size:1.1rem;font-weight:700;color:{CHARCOAL};">
    LightGBM ✓
  </div>
  <div style="font-size:0.85rem;margin-top:10px;color:{CHARCOAL};">
    Градиентный бустинг над деревьями решений. Каждое дерево минимизирует
    остатки предыдущего, улавливая нелинейные взаимодействия между признаками.
    <br><br>
    <b>Плюс:</b> выше Ridge по DA и Calmar, объяснимость через SHAP-values<br>
    <b>Минус:</b> на малых выборках склонна к запоминанию train-сета
  </div>
</div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
<div style="background:rgba(232,232,244,0.5);
            padding:16px;border-radius:8px;">
  <div style="font-size:1.1rem;font-weight:700;color:{CHARCOAL};">MLP (нейросеть)</div>
  <div style="font-size:0.85rem;margin-top:10px;color:{CHARCOAL};">
    Многослойный перцептрон с dropout-регуляризацией. Теоретически аппроксимирует
    произвольно сложные функции, однако требует достаточного объёма данных.
    <br><br>
    <b>Плюс:</b> гибкая архитектура, не требует явного задания структуры зависимостей<br>
    <b>Минус:</b> на 611 наблюдениях высокая дисперсия оценок и нестабильность по фолдам
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Блок 2: Сравнительная таблица ─────────────────────────────────────────
    st.subheader("Сравнение моделей")

    mc_data = pd.DataFrame({
        "Модель": ["Ridge (бейзлайн)", "LightGBM ✓", "MLP (нейросеть)"],
        "DA": ["44.2%", "37.0%", "40.9%"],
        "Calmar": ["—", "1.97", "0.06"],
        "Max просадка": ["—", "-6%", "-65%"],
        "Интерпретируемость": ["Высокая (коэф.)", "SHAP", "Низкая"],
    })
    st.dataframe(mc_data, use_container_width=True, hide_index=True)


    st.divider()
    st.subheader("Итог: почему LightGBM?")

    reasons = [
        ("Calmar 1.97",       "Лучшее соотношение доходности к риску. CAGR +11.8% при просадке всего -6%."),
        ("Max Drawdown -6%",  "В 4 раза меньше просадка чем у Buy&Hold (-23%)."),
        ("CAGR 11.4%",          "Лучше ruonia на 1.15 пп и лучше чем  buy&hold на 5пп."),
        ("SHAP объяснимость", "Управляющий видит какие признаки повлияли на сигнал."),
        ("Стабильность",      "MLP скачет от 0% до 88% по фолдам. LightGBM стабилен."),
    ]

    for title, desc in reasons:
        st.markdown(f"""
<div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid {LAVENDER};">
  <div style="font-size:1rem;font-weight:600;min-width:200px;color:{CHARCOAL};">{title}</div>
  <div style="font-size:0.9rem;color:{CHARCOAL};">{desc}</div>
</div>""", unsafe_allow_html=True)