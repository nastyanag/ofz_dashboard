"""
scripts/model_baseline.py
Logistic Regression (Ridge) + walk-forward на data/processed/features_weekly.csv.

Запуск:  python scripts/model_baseline.py
Выход:
  models/baseline.pkl                     — обученная модель
  models/main.pkl                         — копия (активируется в дашборде)
  data/processed/baseline_predictions.csv — предсказания walk-forward
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

INPUT = Path("data/processed/features_weekly.csv")
MODEL_OUT = Path("models/baseline.pkl")
MAIN_OUT = Path("models/main.pkl")
PREDS_OUT = Path("data/processed/baseline_predictions.csv")

FEATURE_COLS = [
    "key_rate", "delta_ks_1m", "delta_ks_3m", "delta_ks_6m",
    "weeks_since_change", "n_hikes_3m", "n_cuts_3m",
    "real_rate", "real_rate_ma3", "real_rate_change_3m",
    "slope_10y_2y", "slope_10y_1y", "curvature_5y",
    "y_10y", "y_5y", "y_1y",
    "ruonia", "ruonia_spread", "ruonia_spread_ma3",
    "brent_ret_1m", "inflation_yoy", "is_crisis",
    "cycle_phase", "cbr_rhetoric_score",
]

N_FOLDS = 12   # количество фолдов walk-forward
MIN_TRAIN = 104  # минимум ~2 года недельных данных


def walk_forward_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward: обучаем на [0..split], предсказываем [split..split+step].
    Таргет: target_90d > 0 → класс 1 (BUY), иначе 0 (WAIT).
    Возвращает DataFrame: date, target_90d, y_prob, y_pred, fold.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    step = (n - MIN_TRAIN) // N_FOLDS
    all_preds = []

    for fold in range(N_FOLDS):
        train_end = MIN_TRAIN + fold * step
        test_end  = min(train_end + step, n)

        train = df.iloc[:train_end]
        test  = df.iloc[train_end:test_end]
        if len(test) == 0:
            continue

        # Бинаризуем таргет: избыточная доходность > 0 → BUY
        y_tr = (train["target_90d"] > 0).astype(int)
        y_te = (test["target_90d"]  > 0).astype(int)

        X_tr = train[FEATURE_COLS].fillna(0)
        X_te = test[FEATURE_COLS].fillna(0)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ])
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob) if len(y_te.unique()) > 1 else 0.5
        print(f"Fold {fold+1:2d} | train={train_end:4d} | "
              f"test={len(test):3d} | AUC={auc:.3f}")

        chunk = test[["date", "target_90d"]].copy()
        chunk["y_prob"] = y_prob
        chunk["y_pred"] = (y_prob >= 0.5).astype(int)
        chunk["y_true"] = y_te.values
        chunk["fold"]   = fold
        all_preds.append(chunk)

    return pd.concat(all_preds, ignore_index=True)


def train_final(df: pd.DataFrame) -> Pipeline:
    """Финальная модель на всех данных."""
    y = (df["target_90d"] > 0).astype(int)
    X = df[FEATURE_COLS].fillna(0)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    model.fit(X, y)
    return model


if __name__ == "__main__":
    if not INPUT.exists():
        print(f"❌ Файл {INPUT} не найден.")
        print("   Убедись что features_weekly.csv лежит в data/processed/")
        exit(1)

    df = pd.read_csv(INPUT, parse_dates=["date"])
    print(f"Загружено: {len(df)} строк, {len(FEATURE_COLS)} признаков\n")

    # Walk-forward CV
    print("── Walk-forward CV ──────────────────────────────────")
    preds = walk_forward_cv(df)

    # Итоговые метрики
    auc = roc_auc_score(preds["y_true"], preds["y_prob"])
    acc = accuracy_score(preds["y_true"], preds["y_pred"])
    print(f"\n── Итоговые метрики ─────────────────────────────────")
    print(f"AUC:       {auc:.3f}")
    print(f"Accuracy:  {acc:.3f}")
    print(classification_report(
        preds["y_true"], preds["y_pred"],
        target_names=["WAIT", "BUY"],
    ))

    # Сохранение моделей
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    final = train_final(df)
    joblib.dump(final, MODEL_OUT)
    joblib.dump(final, MAIN_OUT)
    print(f"✅ models/baseline.pkl  сохранён")
    print(f"✅ models/main.pkl      сохранён  ← дашборд переключится автоматически")

    # Сохранение предсказаний
    PREDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDS_OUT, index=False)
    print(f"✅ data/processed/baseline_predictions.csv  сохранён")