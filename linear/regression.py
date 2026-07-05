import numpy as np 
from .interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from .optimizers import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC


class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        residuals = X @ w - y
        return np.mean(residuals ** 2)

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        residuals = X @ w - y
        return (2 / n) * X.T @ residuals

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.analytic_solution_func(X, y)
        
    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(X.T @ X) @ X.T @ y
    
    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_pinv = np.linalg.pinv(X)
        return X_pinv @ y


class L2Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0):
        self.core_loss = core_loss
        self.mu_rate = mu_rate

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.gradient(X, y, w)
        penalty = 2 * self.mu_rate * np.copy(w)
        penalty[0] = 0
        return core_part + penalty

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        core_loss = self.core_loss.loss(X, y, w)
        penalty = self.mu_rate * np.sum(w[1:] ** 2)
        return core_loss + penalty


class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_bias @ self.w

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        if X_batch is None or y_batch is None:
            X_batch, y_batch = self.X_train, self.y_train
        X_bias = np.hstack([np.ones((X_batch.shape[0], 1)), X_batch])
        return self.loss_function.gradient(X_bias, y_batch, self.w)

    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        if X_batch is None or y_batch is None:
            X_batch, y_batch = self.X_train, self.y_train
        X_bias = np.hstack([np.ones((X_batch.shape[0], 1)), X_batch])
        return self.loss_function.loss(X_bias, y_batch, self.w)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train, self.y_train = X, y
        self.w = np.zeros(X.shape[1] + 1)

        if isinstance(self.optimizer, AnalyticSolutionOptimizer):
            X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
            self.w = self.loss_function.analytic_solution(X_bias, y)
            self.loss_history.append(self.compute_loss())
        else:
            self.optimizer.optimize()