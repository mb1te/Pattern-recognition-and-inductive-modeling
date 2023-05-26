from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import f1_score


class BestModelDetector(ABC):
    metric = f1_score
    X = None
    y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def print_best_model_info(self):
        ...
