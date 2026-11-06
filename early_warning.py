"""
An early-warning view of corporate financial distress.

Author: Dr Vikneswaran Manual, PhD (Finance)

Builds on distress_model.py with three additions a finance researcher
cares about:
  1. Benchmark: Altman's Z'-score for private firms (Altman, 1983), the
     model practitioners still use, against machine learning.
  2. Robustness: 5-fold stratified cross-validation, not one split.
  3. Early-warning threshold: instead of the default 0.5 cut-off, pick the
     threshold that catches at least 80% of failing firms on training
     folds, then report the cost in false alarms on the hold-out set.
  4. Explainability: SHAP values show which ratios push each firm toward
     distress, so a credit officer can see why a firm was flagged.

Run:  python early_warning.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from distress_model import DATA, SEED, load

RATIO_NAMES = {
    "Attr3": "Working capital / total assets",
    "Attr6": "Retained earnings / total assets",
    "Attr7": "EBIT / total assets",
    "Attr8": "Book equity / total liabilities",
    "Attr9": "Sales / total assets",
    "Attr13": "(Gross profit + depreciation) / sales",
    "Attr16": "(Gross profit + depreciation) / total liabilities",
    "Attr21": "Sales growth",
    "Attr22": "Operating profit / total assets",
    "Attr24": "Gross profit (3 years) / total assets",
    "Attr25": "(Equity - share capital) / total assets",
    "Attr26": "(Net profit + depreciation) / total liabilities",
    "Attr27": "Operating profit / financial expenses",
    "Attr34": "Operating expenses / total liabilities",
    "Attr35": "Profit on sales / total assets",
    "Attr39": "Profit on sales / sales",
    "Attr41": "Total liabilities / operating cash proxy",
    "Attr42": "Operating profit / sales",
    "Attr46": "(Current assets - inventory) / short-term liabilities",
    "Attr48": "EBITDA / total assets",
    "Attr58": "Total costs / total sales",
}
OUT = Path(__file__).parent / "outputs"


def altman_z_prime(X: pd.DataFrame) -> pd.Series:
    """Altman (1983) Z'-score for private firms, using book equity."""
    x = X.fillna(X.median())
    return (0.717 * x["Attr3"] + 0.847 * x["Attr6"] + 3.107 * x["Attr7"]
            + 0.420 * x["Attr8"] + 0.998 * x["Attr9"])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    X, y = load(DATA)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    # 1. Altman Z' benchmark (low Z = distress, so use -Z as the score)
    z_te = altman_z_prime(X_te)
    z_auc = roc_auc_score(y_te, -z_te)
    z_flag = (z_te < 1.23).astype(int)          # Altman's distress zone
    rows = [{"Model": "Altman Z' (zone < 1.23)", "ROC-AUC": z_auc,
             "Recall": recall_score(y_te, z_flag),
             "Precision": precision_score(y_te, z_flag, zero_division=0)}]

    # 2. Gradient boosting with 5-fold CV on the training set
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                        class_weight="balanced", random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(gb, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    cv_auc = roc_auc_score(y_tr, oof)

    # 3. Early-warning threshold: highest cut-off with recall >= 80% out-of-fold
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    ok = np.where(rec[:-1] >= 0.80)[0]
    t = float(thr[ok[-1]])

    gb.fit(X_tr, y_tr)
    p_te = gb.predict_proba(X_te)[:, 1]
    for name, cut in [("Gradient boosting (cut-off 0.50)", 0.5),
                      (f"Gradient boosting (early warning, cut-off {t:.3f})", t)]:
        f = (p_te >= cut).astype(int)
        rows.append({"Model": name, "ROC-AUC": roc_auc_score(y_te, p_te),
                     "Recall": recall_score(y_te, f),
                     "Precision": precision_score(y_te, f, zero_division=0)})

    res = pd.DataFrame(rows).set_index("Model").round(3)
    res.to_csv(OUT / "early_warning_results.csv")
    print(f"5-fold CV ROC-AUC (training set): {cv_auc:.3f}\n")
    print(res.to_string())

    # 4. SHAP explanations
    sample = X_te.sample(800, random_state=SEED)
    explainer = shap.TreeExplainer(gb)
    sv = explainer.shap_values(sample)
    mean_abs = pd.Series(np.abs(sv).mean(0), index=X.columns).nlargest(10)
    top = mean_abs.rename(index=lambda c: f"{RATIO_NAMES.get(c, c)} ({c})")
    top.round(4).to_csv(OUT / "shap_top10.csv", header=["mean |SHAP|"])
    print("\nTop 10 drivers of distress risk (mean |SHAP|):\n")
    print(top.round(4).to_string())

    named = sample.rename(columns=lambda c: RATIO_NAMES.get(c, c))
    shap.summary_plot(sv, named, max_display=10, show=False)
    plt.title("What drives predicted distress? (SHAP, gradient boosting)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150)
    print(f"\nSaved charts and tables to {OUT}/")


if __name__ == "__main__":
    main()
