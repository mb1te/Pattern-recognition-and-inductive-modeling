import numpy as np
from tpot import TPOTClassifier

from src.auto_ml.base import BestModelDetector


class TPOTDetector(BestModelDetector):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.clf = TPOTClassifier(
            generations=5,
            population_size=20,
            verbosity=2,
            scoring=self.scorer,
        )

        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def print_best_model_info(self):
        return self.clf.fitted_pipeline_
