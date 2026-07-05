import numpy as np
from abc import ABC, abstractmethod
from .interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface

class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr

class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        res = (self.s0 / (self.s0 + iteration)) ** self.p
        return self.lambda_ * res

class BaseDescent(AbstractOptimizer, ABC):
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        pass

    def _step(self) -> np.ndarray:
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        initial_loss = self.model.compute_loss()
        self.model.loss_history = [initial_loss]
        for i in range(self.max_iter):
            delta = self._step()
            current_loss = self.model.compute_loss()
            self.model.loss_history.append(current_loss)
            if (np.any(np.isnan(delta))) or (np.linalg.norm(delta) ** 2 < self.tolerance):
                break

class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        grad = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)
        delta = -lr * grad
        self.model.w += delta
        return delta

class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        n = self.model.X_train.shape[0]
        indices = np.random.randint(0, n, size=self.batch_size)
        X_batch = self.model.X_train[indices]
        y_batch = self.model.y_train[indices]
        grad = self.model.compute_gradients(X_batch, y_batch)
        lr = self.lr_schedule.get_lr(self.iteration)
        delta = -lr * grad
        self.model.w += delta
        return delta

class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X, y = self.model.X_train, self.model.y_train
        n, d = X.shape
        if self.grad_memory is None:
            self.grad_memory = np.zeros((n, d))
            self.grad_sum = np.zeros(d)
        indices = np.random.randint(0, n, size=self.batch_size)
        for idx in indices:
            old_grad = self.grad_memory[idx].copy()
            X_i = X[idx].reshape(1, -1)
            y_i = y[idx].reshape(1)
            new_grad = self.model.compute_gradients(X_i, y_i)
            self.grad_memory[idx] = new_grad
            self.grad_sum += (new_grad - old_grad)
        avg_grad = self.grad_sum / n
        lr = self.lr_schedule.get_lr(self.iteration)
        delta = -lr * avg_grad
        self.model.w += delta
        return delta

class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        grad = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)
        if self.velocity is None:
            self.velocity = np.zeros_like(grad)
        self.velocity = self.beta * self.velocity + lr * grad
        delta = -self.velocity
        self.model.w += delta
        return delta

class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        grad = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** (self.iteration + 1))
        v_hat = self.v / (1 - self.beta2 ** (self.iteration + 1))
        delta = -lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.model.w += delta
        return delta

class AnalyticSolutionOptimizer(AbstractOptimizer):
    def __init__(self):
        self.model = None

    def optimize(self) -> None:
        w_opt = self.model.loss_function.analytic_solution(self.model.X_train, self.model.y_train)
        self.model.w = w_opt
        final_loss = self.model.compute_loss()
        self.model.loss_history = [final_loss]