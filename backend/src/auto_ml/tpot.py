import numpy as np
from base import BestModelDetector


class TPOTDetector(BestModelDetector):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
