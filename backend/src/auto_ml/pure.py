import itertools

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from src.auto_ml.base import BestModelDetector


class PurePythonModelDetector(BestModelDetector):
    def __init__(self) -> None:
        self.models = {
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "mlp": MLPClassifier(),
        }

        self.param_space = {
            "random_forest": {
                "n_estimators": (5, 10, 25, 50, 100, 200),
                "max_depth": (1, 2, 3, 5, 10, 20),
            },
            "gradient_boosting": {
                "n_estimators": (5, 10, 25, 50, 100, 200),
                "learning_rate": (0.001, 0.01, 0.05, 0.1, 0.2),
            },
            "mlp": {
                "hidden_layer_sizes": (1, 5, 10, 25, 50, 100, 200, 500, 1000),
                "alpha": (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2),
            },
        }
        self.best_params = {}
        self.best_score = 0
        self.best_model = None

    def grid_search(self, model_name: str, space: dict, X: np.ndarray, y: np.ndarray) -> None:
        model = self.models[model_name]
        best_score = 0
        best_params = None

        for params in itertools.product(*space.values()):
            params_dict = dict(zip(space.keys(), params))
            model.set_params(**params_dict)

            scorer = make_scorer(f1_score, average="weighted")
            score = np.mean(cross_val_score(model, X, y, cv=5, scoring=scorer))
            if score > best_score:
                best_score = score
                best_params = params_dict

        self.best_params[model_name] = best_params
        model.set_params(**best_params)
        model.fit(X, y)

        if best_score > self.best_score:
            self.best_score = best_score
            self.best_model = model_name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for model_name in self.models.keys():
            space = self.param_space[model_name]
            self.grid_search(model_name, space, X, y)

    def print_best_model_info(self) -> None:
        print(
            "model",
            self.best_model,
            "params",
            self.best_params[self.best_model],
            "train_f1",
            self.best_score,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.models[self.best_model].predict(X)
