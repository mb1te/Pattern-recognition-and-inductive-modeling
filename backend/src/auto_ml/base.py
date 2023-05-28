from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import f1_score, make_scorer


class BestModelDetector(ABC):
    scorer = make_scorer(f1_score, average="weighted")
    X = None
    y = None
    clf = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def print_best_model_info(self):
        ...
