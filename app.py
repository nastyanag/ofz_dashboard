"""
app.py — точка входа Streamlit-дашборда
Запуск: streamlit run app.py

Архитектура:
  app.py
    └─ pages/
        ├─ page_data.py        → Экран 1: «Данные» (КС + RGBI/RGBITR)
        ├─ page_signal.py      → Экран 2: «Текущий сигнал» (заглушка → потом реальная модель)
        ├─ page_backtest.py    → Экран 3: «Бэктест» (equity curve vs бенчмарки)
        ├─ page_portfolio.py   → Экран 4: «Портфель» (дюрация по уровню сигнала)
        └─ page_whatif.py      → Экран 5: «Что-если» (слайдеры для жюри)

  data/
    raw/          → CSV после fetch_*.py скриптов (не коммитить в git тяжёлые файлы)
    processed/    → features.csv, backtest_results.csv

  models/
    baseline.pkl  → Ridge-регрессия
    main.pkl      → LightGBM / GBM
    nn.pkl        → MLP

  scripts/
    fetch_cbr_keyrate.py   → тянет КС с cbr.ru
    fetch_moex_iss.py      → тянет RGBI/RGBITR/ОФЗ с iss.moex.com
    fetch_macro.py         → RUONIA, КБД
    features.py            → feature engineering
    model_baseline.py      → обучение Ridge
    model_main.py          → обучение LightGBM
    model_nn.py            → обучение MLP
    backtest.py            → walk-forward бэктест

  references/
    data-sources.md        → endpoint-ы, форматы, пагинация ISS
    ml-methodology.md      → признаки и обоснование, walk-forward
    backtest.md            → бенчмарки, метрики
    dashboard-spec.md      → спека 5 экранов
    financial-logic.md     → теория дюрации, YTM, фаз цикла
    case-defense.md        → ответы на вопросы жюри

  assets/
    pitch-deck-outline.md  → структура 5-минутного питча
"""

import streamlit as st
# Принудительная установка вашей кастомной светлой темы
st._config.set_option("theme.base", "light")
st._config.set_option("theme.primaryColor", "#f63366")
st._config.set_option("theme.backgroundColor", "#FFFFFF")
st._config.set_option("theme.secondaryBackgroundColor", "#F0F2F6")
st._config.set_option("theme.textColor", "#262730")
st._config.set_option("theme.font", "sans serif")

# ── Конфигурация страницы (обязательно первый вызов st) ──────────────────────
st.set_page_config(
    page_title="ОФЗ: оптимизация момента покупки",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Импорт страниц ────────────────────────────────────────────────────────────
from page_modules.page_data import render as render_data
from page_modules.page_signal import render as render_signal
from page_modules.page_portfolio import render as render_portfolio
from page_modules.page_whatif import render as render_whatif
from page_modules.page_models    import render as render_models


# ── Навигация ─────────────────────────────────────────────────────────────────
PAGES = {
    "Данные":          render_data,
    "Текущая вероятность":  render_signal,
    "Выбор модели":          render_models,
    "Портфель":        render_portfolio,
    "Определить момент покупки":        render_whatif,
}

with st.sidebar:
    st.title("ОФЗ-оптимайзер")
    #st.caption("")
    st.divider()
    selection = st.radio("Экран", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Данные: ЦБ РФ, MOEX ISS")

# ── Рендер выбранной страницы ─────────────────────────────────────────────────
PAGES[selection]()
