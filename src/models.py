from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class STLModel:
    model: LogisticRegression

    def predict(self, X):
        return (self.model.decision_function(X) >= 0).astype(int)

    def scores(self, X):
        return self.model.decision_function(X)


@dataclass
class ITLModel:
    models: dict

    def predict(self, X, s):
        preds = np.zeros(X.shape[0], dtype=int)
        for g, model in self.models.items():
            mask = s == g
            if mask.any():
                preds[mask] = (model.decision_function(X[mask]) >= 0).astype(int)
        return preds

    def scores(self, X, s):
        scores = np.zeros(X.shape[0])
        for g, model in self.models.items():
            mask = s == g
            if mask.any():
                scores[mask] = model.decision_function(X[mask])
        return scores


def train_stl(X_train, y_train, C=1.0):
    model = LogisticRegression(
        C=C,
        max_iter=200,
        solver="liblinear",
        random_state=0,
    )
    model.fit(X_train, y_train)
    return STLModel(model=model)


def train_itl(X_train, y_train, s_train, C=1.0):
    models = {}
    for g in np.unique(s_train):
        mask = s_train == g
        clf = LogisticRegression(
            C=C,
            max_iter=200,
            solver="liblinear",
            random_state=0,
        )
        clf.fit(X_train[mask], y_train[mask])
        models[int(g)] = clf
    return ITLModel(models=models)
