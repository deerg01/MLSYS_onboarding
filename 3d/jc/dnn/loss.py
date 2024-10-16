from typing import Any
import numpy as np


class Loss:
    def __call__(self, y_, y) -> Any:
        pass


class MSE(Loss):   
    def __call__(self, y_, y) -> Any:
        return np.mean((y_ - y)**2, axis=0)


class CrossEntropyLoss(Loss):
    def __call__(self, y_, y) -> Any:
        return -np.sum(y_ * np.log(y + 1e-9)) / y.shape[0]
