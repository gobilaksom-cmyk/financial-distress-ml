"""
Financial distress (bankruptcy) prediction with machine learning.

Author: Dr Vikneswaran Manual, PhD (Finance)

Data: Polish Companies Bankruptcy dataset (Tomczak, 2016), UCI Machine
Learning Repository, id 365. 64 financial ratios per firm-year; the target
is whether the firm went bankrupt within the forecasting horizon.
This script uses the 1-year-ahead horizon (5year.arff: ratios from the
5th year, status after one year).

Models compared:
  1. Logistic regression (the classic econometric benchmark, cf. Ohlson 1980)
  2. Random forest
  3. Gradient boosting (histogram-based)

Evaluation: stratified train/test split, ROC-AUC, and recall on distressed
firms, because missing a failing firm is costlier than a false alarm.

Run:  pip install -r requirements.txt && python distress_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

DATA = Path(__file__).parent / "data" / "5year.arff"
SEED = 42


def load(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    raw, _ = arff.loadarff(path)
    df = pd.DataFrame(raw)
    y = df.pop("class").astype(str).str.strip("b'").astype(int)
    X = df.astype(float).replace([np.inf, -np.inf], np.nan)
    # Winsorise extreme ratio values at the 1st/99th percentiles
    X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1)
    return X, y


def main() -> None:
    X, y = load(DATA)
    print(f"Firm-years: {len(y):,} | distressed: {y.sum():,} ({y.mean():.1%})")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    models = {
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400, class_weight="balanced_subsample",
                min_samples_leaf=2, n_jobs=-1, random_state=SEED,
            ),
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, class_weight="balanced",
            random_state=SEED,
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        rep = classification_report(y_te, pred, output_dict=True, zero_division=0)
        rows.append({
            "Model": name,
            "ROC-AUC": round(roc_auc_score(y_te, p), 3),
            "Recall (distressed)": round(rep["1"]["recall"], 3),
            "Precision (distressed)": round(rep["1"]["precision"], 3),
        })

    results = pd.DataFrame(rows).set_index("Model")
    print("\nTest-set results (25% hold-out):\n")
    print(results.to_string())
    results.to_csv(Path(__file__).parent / "results.csv")

    rf = models["Random forest"][-1]
    imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
    print("\nTop 10 ratios (random forest importance):\n")
    print(imp.round(4).to_string())


if __name__ == "__main__":
    main()
