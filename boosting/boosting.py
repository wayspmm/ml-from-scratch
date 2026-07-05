from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class MeanTargetEncoder:

    def __init__(self, cat_features: Iterable[int] | None = None, smoothing: float = 1.0):
        self.cat_features = list(cat_features) if cat_features is not None else []
        self.smoothing = smoothing
        self.maps_: dict[int, pd.Series] = {}
        self.global_mean_: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanTargetEncoder":
        y_bin = (np.asarray(y) == 1).astype(float)
        self.global_mean_ = float(y_bin.mean())
        for j in self.cat_features:
            col = X[:, j]
            uniq, inv = np.unique(col, return_inverse=True)
            sums = np.bincount(inv, weights=y_bin)
            counts = np.bincount(inv)
            enc = (sums + self.smoothing * self.global_mean_) / (counts + self.smoothing)
            self.maps_[j] = pd.Series(enc, index=uniq)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        is_object = (X.dtype == object)
        if is_object:
            X_out = np.zeros(X.shape, dtype=float)
            for j in range(X.shape[1]):
                if j in self.cat_features:
                    s = self.maps_[j]
                    X_out[:, j] = pd.Series(X[:, j]).map(s).fillna(self.global_mean_).values
                else:
                    X_out[:, j] = X[:, j].astype(float)
            return X_out
        X_out = X.astype(float).copy()
        for j in self.cat_features:
            s = self.maps_[j]
            X_out[:, j] = pd.Series(X[:, j]).map(s).fillna(self.global_mean_).values
        return X_out

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class BoostingClassifier(ClassifierMixin):

    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        cat_features: Iterable[int] | None = None,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = None,
        rsm: float = 1.0,
    ):
        super().__init__()
        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.gammas = []
        self.random_state = random_state
        self.verbose = verbose
        self.history = defaultdict(list)
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric if eval_metric else "val_loss"
        self.classes_ = None

        self.cat_features = list(cat_features) if cat_features is not None else None
        self._encoder: MeanTargetEncoder | None = None

        assert bootstrap_type in (None, "Bernoulli", "Bayesian"), "unknown bootstrap_type"
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm

        self._n_features: int | None = None
        self._feature_idx: list = []

        self._rng = np.random.default_rng(random_state)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z) + 1e-15).mean()
        self.grad_fn = lambda y, z: -y / (1 + np.exp(np.clip(y * z, -500, 500)))
        self.hess_fn = lambda y, z: np.clip(
            self.sigmoid(y * z) * (1 - self.sigmoid(y * z)), 1e-6, None
        )

    def _maybe_fit_encoder(self, X, y):
        if self.cat_features is None or len(self.cat_features) == 0:
            return np.asarray(X, dtype=float)
        self._encoder = MeanTargetEncoder(cat_features=self.cat_features)
        return self._encoder.fit_transform(X, y)

    def _maybe_transform(self, X):
        if self._encoder is None:
            return np.asarray(X, dtype=float)
        return self._encoder.transform(X)

    def _get_sample_weight(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if self.bootstrap_type is None:
            return np.arange(n), np.ones(n)
        if self.bootstrap_type == "Bernoulli":
            if self.subsample >= 1.0:
                return np.arange(n), np.ones(n)
            mask = self._rng.random(n) < self.subsample
            if not mask.any():
                mask[self._rng.integers(0, n)] = True
            idx = np.where(mask)[0]
            return idx, np.ones(len(idx))
        if self.bootstrap_type == "Bayesian":
            u = self._rng.uniform(1e-12, 1.0, size=n)
            w = (-np.log(u)) ** self.bagging_temperature
            return np.arange(n), w
        return np.arange(n), np.ones(n)

    def _get_features(self, n_features: int) -> np.ndarray:
        if self.rsm >= 1.0:
            return np.arange(n_features)
        k = max(1, int(round(self.rsm * n_features)))
        return np.sort(self._rng.choice(n_features, size=k, replace=False))

    def partial_fit(self, X, y, current_preds):
        y_t = np.where(y == self.classes_[0], -1, 1)
        antigrads = -self.grad_fn(y_t, current_preds)

        idx, w = self._get_sample_weight(len(y))
        feat = self._get_features(X.shape[1])
        self._feature_idx.append(feat)
        X_sub = X[idx][:, feat]
        target_sub = antigrads[idx]
        sample_weight = w[idx] if self.bootstrap_type == "Bayesian" else None

        params = dict(self.base_model_params)
        if self.base_model_class is DecisionTreeRegressor and self.random_state is not None:
            params["random_state"] = self.random_state + len(self.models)

        model = self.base_model_class(**params)
        if sample_weight is not None:
            model.fit(X_sub, target_sub, sample_weight=sample_weight)
        else:
            model.fit(X_sub, target_sub)

        new_preds = model.predict(X[:, feat])
        gamma = self._find_optimal_gamma(y_t, current_preds, new_preds)
        self.models.append(model)
        self.gammas.append(gamma)
        return current_preds + self.learning_rate * gamma * new_preds, new_preds

    def fit(self, X_train, y_train, eval_set=None, use_best_model=False):
        self.models = []
        self.gammas = []
        self._feature_idx = []
        self.history = defaultdict(list)

        X_train_enc = self._maybe_fit_encoder(X_train, y_train)
        self._n_features = X_train_enc.shape[1]
        self.classes_ = np.unique(y_train)
        n = X_train_enc.shape[0]

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_enc = self._maybe_transform(X_val)
        else:
            X_val_enc, y_val = None, None

        current_preds = np.zeros(n)
        val_preds = np.zeros(X_val_enc.shape[0]) if eval_set is not None else None

        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range, desc="Boosting")

        best_val_metric = np.inf if "loss" in self.eval_metric else -np.inf
        best_iter = 0
        patience = 0

        for step in estimator_range:
            current_preds, new_train = self.partial_fit(X_train_enc, y_train, current_preds)

            y_train_t = np.where(y_train == self.classes_[0], -1, 1)
            train_loss = self.loss_fn(y_train_t, current_preds)
            train_auc = roc_auc_score(y_train == 1, self.sigmoid(current_preds))
            self.history["train_loss"].append(train_loss)
            self.history["train_roc_auc"].append(train_auc)

            if eval_set is not None:
                y_val_t = np.where(y_val == self.classes_[0], -1, 1)
                feat = self._feature_idx[-1]
                val_preds = val_preds + self.learning_rate * self.gammas[-1] * \
                    self.models[-1].predict(X_val_enc[:, feat])
                val_loss = self.loss_fn(y_val_t, val_preds)
                val_auc = roc_auc_score(y_val == 1, self.sigmoid(val_preds))
                self.history["val_loss"].append(val_loss)
                self.history["val_roc_auc"].append(val_auc)

                cur = self.history[self.eval_metric][-1]
                is_better = (cur < best_val_metric) if "loss" in self.eval_metric \
                    else (cur > best_val_metric)
                if is_better or step == 0:
                    best_val_metric = cur
                    best_iter = step
                    patience = 0
                else:
                    patience += 1
                if (self.early_stopping_rounds is not None
                        and patience >= self.early_stopping_rounds):
                    if self.verbose:
                        tqdm.write(f"Early stopping at iter {step}. Best iter: {best_iter}")
                    break

        if use_best_model and eval_set is not None:
            cutoff = best_iter + 1
            self.models = self.models[:cutoff]
            self.gammas = self.gammas[:cutoff]
            self._feature_idx = self._feature_idx[:cutoff]
            for key in self.history:
                self.history[key] = self.history[key][:cutoff]

        for key in self.history:
            self.history[key] = np.array(self.history[key])
        return self

    def predict_proba(self, X):
        X = self._maybe_transform(X)
        if len(self.models) == 0:
            p1 = np.full(X.shape[0], 0.5)
            return np.column_stack([1 - p1, p1])
        z = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            feat = self._feature_idx[i] if i < len(self._feature_idx) else np.arange(X.shape[1])
            z += self.learning_rate * self.gammas[i] * self.models[i].predict(X[:, feat])
        p1 = self.sigmoid(z)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba[:, 1] >= 0.5, self.classes_[1], self.classes_[0])

    def _find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(0, 1, 100)
        losses = [self.loss_fn(y, old_predictions + g * new_predictions) for g in gammas]
        return gammas[int(np.argmin(losses))]

    def score(self, X, y):
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])

    def plot_history(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        plt.figure(figsize=(10, 6))
        for key in keys:
            if key in self.history:
                plt.plot(self.history[key], label=key, linewidth=2)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Metric Value", fontsize=12)
        plt.title("Boosting Learning History", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_feature_importance(self, X=None, y=None, type: str = "split") -> np.ndarray:
        if type != "split":
            raise NotImplementedError("Only type='split' is implemented")
        if self._n_features is None:
            raise RuntimeError("Model is not fitted yet")
        importances = np.zeros(self._n_features)
        gsum = 0.0
        for i, model in enumerate(self.models):
            gamma = abs(self.gammas[i])
            feat = self._feature_idx[i] if i < len(self._feature_idx) \
                else np.arange(self._n_features)
            imp_local = getattr(model, "feature_importances_", None)
            if imp_local is None:
                continue
            importances[feat] += gamma * imp_local
            gsum += gamma
        if gsum > 0:
            importances /= gsum
        s = importances.sum()
        if s > 0:
            importances /= s
        return importances
