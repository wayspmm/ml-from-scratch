# ml-from-scratch

Classic ML algorithms implemented from scratch in NumPy, benchmarked against
production libraries. Built to understand *why* the standard implementations
work, not just how to call them.

## What's inside

### `boosting/` — Gradient boosting classifier
A CatBoost-style gradient boosting implementation over regression trees:

- logistic loss with analytic gradient/hessian, optimal step size (gamma) via line search
- **Bernoulli and Bayesian bootstrap** (bagging temperature, as in CatBoost)
- **random subspace method (RSM)** — per-tree feature subsampling
- **early stopping** on a validation set with `use_best_model`
- **mean-target encoding** for categorical features with smoothing
- training history tracking + feature importances

**Benchmark vs LightGBM** (identical budgets: 300 trees, depth 4, lr 0.1,
subsample 0.8, feature fraction 0.8, early stopping; ROC-AUC on a held-out
test set; see `examples/benchmark_boosting.py`):

| dataset | BoostingClassifier | LightGBM | gap |
|---|---|---|---|
| breast_cancer (n=569) | 0.9725 | 0.9870 | 1.45 p.p. |
| synthetic (n=20 000)  | 0.9665 | 0.9730 | 0.65 p.p. |

Within 0.7–1.5 p.p. ROC-AUC of LightGBM — the remaining gap is mostly
histogram-based split finding and leaf-wise growth, which are out of scope here.

```python
from boosting import BoostingClassifier

model = BoostingClassifier(
    n_estimators=300, learning_rate=0.1,
    base_model_params={"max_depth": 4},
    bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8,
    early_stopping_rounds=30, eval_metric="val_roc_auc",
)
model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
proba = model.predict_proba(X_test)
```

### `trees/` — Decision tree
CART-style classification tree supporting numeric and categorical features:

- fully **vectorized split search**: all thresholds of a feature are evaluated
  in one pass via cumulative sums (no Python loop over thresholds)
- categorical splits via target-rate ordering (the classic CART trick)
- `max_depth`, `min_samples_split`, `min_samples_leaf` pruning

### `linear/` — Linear regression + optimizers
Linear regression decoupled from optimization (Strategy pattern):

- optimizers: full-batch GD, **SGD**, **SAG**, **Momentum**, **Adam**,
  and a closed-form solution (normal equations / SVD pseudo-inverse)
- learning-rate schedules: constant, time decay
- losses: MSE, **L2 regularization** as a wrapper over any base loss
  (bias term excluded from the penalty)

## Setup

```bash
pip install -r requirements.txt
python examples/benchmark_boosting.py
```

## Layout

```
boosting/boosting.py      # BoostingClassifier, MeanTargetEncoder
trees/decision_tree.py    # DecisionTree, find_best_split
linear/interfaces.py      # abstract contracts (loss / optimizer / model)
linear/optimizers.py      # GD, SGD, SAG, Momentum, Adam, analytic
linear/regression.py      # MSELoss, L2Regularization, CustomLinearRegression
examples/                 # reproducible benchmarks
```
