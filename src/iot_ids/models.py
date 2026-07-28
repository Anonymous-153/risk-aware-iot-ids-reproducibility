from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iot_ids.metrics import class_weights


class OptionalDependencyError(RuntimeError):
    pass


@dataclass
class CentroidClassifier:
    classes_: np.ndarray | None = None
    centroids_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None) -> "CentroidClassifier":
        del sample_weight
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels)
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([x[y == class_label].mean(axis=0) for class_label in self.classes_])
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.centroids_ is None:
            raise ValueError("classifier must be fit before predict_proba")
        x = np.asarray(features, dtype=float)
        distances = np.stack([np.linalg.norm(x - centroid, axis=1) for centroid in self.centroids_], axis=1)
        logits = -distances
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]


def sample_weights(labels: np.ndarray) -> np.ndarray:
    weights = class_weights(np.asarray(labels))
    return np.asarray([weights[label] for label in labels], dtype=float)


def make_classifier(model_name: str, seed: int, xgboost_device: str | None = None):
    normalized = model_name.lower().replace("-", "_")
    if normalized == "centroid":
        return CentroidClassifier()

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
    except ImportError as exc:
        raise OptionalDependencyError(
            "Install experiment dependencies with `python -m pip install -r requirements.txt` "
            "before running logistic_regression, random_forest, xgboost, or mlp."
        ) from exc

    if normalized in {"logistic", "logistic_regression", "lr"}:
        return LogisticRegression(max_iter=5000, solver="lbfgs", random_state=seed)
    if normalized in {"random_forest", "rf"}:
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if normalized in {"mlp", "neural_network"}:
        return MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=80,
            early_stopping=True,
            n_iter_no_change=5,
            batch_size=2048,
            random_state=seed,
        )
    if normalized in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise OptionalDependencyError("Install xgboost before running the xgboost model.") from exc
        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": seed,
        }
        if xgboost_device:
            params["device"] = xgboost_device
        return XGBClassifier(**params)
    raise ValueError(f"unknown model {model_name!r}")


def fit_classifier(classifier, features: np.ndarray, labels: np.ndarray, use_sample_weights: bool) -> None:
    weights = sample_weights(labels) if use_sample_weights else None
    if weights is None:
        classifier.fit(features, labels)
        return
    try:
        classifier.fit(features, labels, sample_weight=weights)
    except TypeError:
        classifier.fit(features, labels)
