"""Benchmark: custom BoostingClassifier vs LightGBM.

Both models get identical budgets: 300 trees max, depth 4, lr 0.1,
row subsampling 0.8, feature subsampling 0.8, early stopping on a
validation split. Reported metric is ROC-AUC on a held-out test set.

Run:  python examples/benchmark_boosting.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from boosting import BoostingClassifier


def bench(X, y, name):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    Xtr2, Xval, ytr2, yval = train_test_split(
        Xtr, ytr, test_size=0.25, random_state=42, stratify=ytr
    )

    t0 = time.time()
    ours = BoostingClassifier(
        n_estimators=300, learning_rate=0.1,
        base_model_params={"max_depth": 4},
        bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8,
        early_stopping_rounds=30, eval_metric="val_roc_auc",
        verbose=False, random_state=0,
    )
    ours.fit(Xtr2, ytr2, eval_set=(Xval, yval), use_best_model=True)
    auc_ours = roc_auc_score(yte, ours.predict_proba(Xte)[:, 1])
    t_ours = time.time() - t0

    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1,
    )
    lgbm.fit(Xtr2, ytr2, eval_set=[(Xval, yval)],
             callbacks=[lgb.early_stopping(30, verbose=False)])
    auc_lgbm = roc_auc_score(yte, lgbm.predict_proba(Xte)[:, 1])
    t_lgbm = time.time() - t0

    print(f"== {name} (n={len(y)}) ==")
    print(f"  BoostingClassifier: test ROC-AUC = {auc_ours:.4f}  ({t_ours:.1f}s)")
    print(f"  LightGBM:           test ROC-AUC = {auc_lgbm:.4f}  ({t_lgbm:.1f}s)")
    print(f"  gap: {100 * (auc_lgbm - auc_ours):+.2f} p.p.\n")


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    bench(X, y, "breast_cancer")

    X, y = make_classification(
        n_samples=20_000, n_features=30, n_informative=15,
        flip_y=0.05, random_state=1,
    )
    bench(X, y, "synthetic-20k")
