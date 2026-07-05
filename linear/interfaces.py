"""Abstract interfaces shared by the linear regression module.

These base classes decouple the model, the loss function and the
optimizer, so each part can be swapped independently (Strategy pattern).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LearningRateSchedule(ABC):
    """Learning rate as a function of the iteration number."""

    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        ...


class LossFunction(ABC):
    """Differentiable loss: value and gradient w.r.t. weights."""

    @abstractmethod
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        ...

    @abstractmethod
    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        ...


class LossFunctionClosedFormMixin(ABC):
    """Mixin for losses that admit a closed-form minimizer."""

    @abstractmethod
    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...


class LinearRegressionInterface(ABC):
    """Minimal contract the optimizers rely on."""

    w: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    loss_history: list
    loss_function: LossFunction

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def compute_loss(self, X_batch=None, y_batch=None) -> float:
        ...

    @abstractmethod
    def compute_gradients(self, X_batch=None, y_batch=None) -> np.ndarray:
        ...


class AbstractOptimizer(ABC):
    """Optimizer bound to a model via set_model()."""

    model: LinearRegressionInterface | None = None

    def set_model(self, model: LinearRegressionInterface) -> None:
        self.model = model

    @abstractmethod
    def optimize(self) -> None:
        ...
