from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Hashable

import numpy as np


def _as_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    if np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("probabilities must be in [0, 1]")
    return matrix


def _label_key(label: Hashable) -> Hashable:
    return label.item() if hasattr(label, "item") else label


def _finite_sample_threshold(scores: np.ndarray, alpha: float) -> float:
    sorted_scores = np.sort(scores)
    rank = ceil((scores.size + 1) * (1 - alpha))
    rank = min(max(rank, 1), scores.size)
    return float(sorted_scores[rank - 1])


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0 < value < 1:
        raise ValueError("alpha must be between 0 and 1")
    return value


def _adaptive_score(row: np.ndarray, class_index: int) -> float:
    order = np.argsort(-row, kind="stable")
    sorted_probabilities = row[order]
    class_rank = int(np.where(order == class_index)[0][0])
    return float(np.cumsum(sorted_probabilities)[class_rank])


@dataclass
class ClassConditionalConformal:
    """Class-conditional split conformal predictor for classifier probabilities.

    Nonconformity is ``1 - P(true_class)``. A test label is included in the
    prediction set when its class-specific nonconformity score is below the
    threshold calibrated on held-out calibration data.
    """

    alpha: float = 0.1
    classes_: np.ndarray | None = None
    thresholds_: dict[Hashable, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.alpha = _validate_alpha(self.alpha)

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        classes: np.ndarray | list[Hashable] | None = None,
    ) -> "ClassConditionalConformal":
        matrix = _as_probability_matrix(probabilities)
        y = np.asarray(labels)
        inferred_classes = np.asarray(classes if classes is not None else np.unique(y))
        if matrix.shape[1] != len(inferred_classes):
            raise ValueError("number of probability columns must match number of classes")

        self.classes_ = inferred_classes
        self.thresholds_ = {}
        for class_index, class_label in enumerate(self.classes_):
            class_scores = 1.0 - matrix[y == class_label, class_index]
            if class_scores.size == 0:
                raise ValueError(f"no calibration examples for class {class_label!r}")
            self.thresholds_[_label_key(class_label)] = _finite_sample_threshold(class_scores, self.alpha)
        return self

    def predict_sets(self, probabilities: np.ndarray) -> list[set[Hashable]]:
        if self.classes_ is None or not self.thresholds_:
            raise ValueError("conformal predictor must be fit before predict_sets")
        matrix = _as_probability_matrix(probabilities)
        if matrix.shape[1] != len(self.classes_):
            raise ValueError("number of probability columns must match fitted classes")

        prediction_sets: list[set[Hashable]] = []
        for row in matrix:
            labels: set[Hashable] = set()
            for class_index, class_label in enumerate(self.classes_):
                key = _label_key(class_label)
                if 1.0 - row[class_index] <= self.thresholds_[key]:
                    labels.add(key)
            if not labels:
                fallback = self.classes_[int(np.argmax(row))]
                labels.add(_label_key(fallback))
            prediction_sets.append(labels)
        return prediction_sets


@dataclass
class ClassConditionalAdaptiveConformal:
    """Class-conditional adaptive prediction-set conformal predictor.

    The score for a class is the cumulative probability mass encountered when
    classes are sorted from most to least likely and scanned until that class is
    reached. This is an APS-style score; low scores produce singleton sets,
    while high thresholds allow lower-ranked labels into uncertain predictions.
    """

    alpha: float = 0.1
    classes_: np.ndarray | None = None
    thresholds_: dict[Hashable, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.alpha = _validate_alpha(self.alpha)

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        classes: np.ndarray | list[Hashable] | None = None,
    ) -> "ClassConditionalAdaptiveConformal":
        matrix = _as_probability_matrix(probabilities)
        y = np.asarray(labels)
        inferred_classes = np.asarray(classes if classes is not None else np.unique(y))
        if matrix.shape[1] != len(inferred_classes):
            raise ValueError("number of probability columns must match number of classes")

        self.classes_ = inferred_classes
        self.thresholds_ = {}
        for class_index, class_label in enumerate(self.classes_):
            class_rows = matrix[y == class_label]
            if class_rows.size == 0:
                raise ValueError(f"no calibration examples for class {class_label!r}")
            class_scores = np.asarray([_adaptive_score(row, class_index) for row in class_rows])
            self.thresholds_[_label_key(class_label)] = _finite_sample_threshold(class_scores, self.alpha)
        return self

    def predict_sets(self, probabilities: np.ndarray) -> list[set[Hashable]]:
        if self.classes_ is None or not self.thresholds_:
            raise ValueError("conformal predictor must be fit before predict_sets")
        matrix = _as_probability_matrix(probabilities)
        if matrix.shape[1] != len(self.classes_):
            raise ValueError("number of probability columns must match fitted classes")

        prediction_sets: list[set[Hashable]] = []
        for row in matrix:
            labels: set[Hashable] = set()
            for class_index, class_label in enumerate(self.classes_):
                key = _label_key(class_label)
                if _adaptive_score(row, class_index) <= self.thresholds_[key]:
                    labels.add(key)
            if not labels:
                fallback = self.classes_[int(np.argmax(row))]
                labels.add(_label_key(fallback))
            prediction_sets.append(labels)
        return prediction_sets


@dataclass
class ClassSpecificAlphaConformal:
    """Class-conditional split conformal predictor with per-class error budgets."""

    alpha: float = 0.1
    alpha_by_class: dict[Hashable, float] = field(default_factory=dict)
    classes_: np.ndarray | None = None
    thresholds_: dict[Hashable, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.alpha = _validate_alpha(self.alpha)
        self.alpha_by_class = {_label_key(label): _validate_alpha(alpha) for label, alpha in self.alpha_by_class.items()}

    def _alpha_for_class(self, class_label: Hashable) -> float:
        return self.alpha_by_class.get(_label_key(class_label), self.alpha)

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        classes: np.ndarray | list[Hashable] | None = None,
    ) -> "ClassSpecificAlphaConformal":
        matrix = _as_probability_matrix(probabilities)
        y = np.asarray(labels)
        inferred_classes = np.asarray(classes if classes is not None else np.unique(y))
        if matrix.shape[1] != len(inferred_classes):
            raise ValueError("number of probability columns must match number of classes")

        self.classes_ = inferred_classes
        self.thresholds_ = {}
        for class_index, class_label in enumerate(self.classes_):
            class_scores = 1.0 - matrix[y == class_label, class_index]
            if class_scores.size == 0:
                raise ValueError(f"no calibration examples for class {class_label!r}")
            self.thresholds_[_label_key(class_label)] = _finite_sample_threshold(
                class_scores,
                self._alpha_for_class(class_label),
            )
        return self

    def predict_sets(self, probabilities: np.ndarray) -> list[set[Hashable]]:
        if self.classes_ is None or not self.thresholds_:
            raise ValueError("conformal predictor must be fit before predict_sets")
        matrix = _as_probability_matrix(probabilities)
        if matrix.shape[1] != len(self.classes_):
            raise ValueError("number of probability columns must match fitted classes")

        prediction_sets: list[set[Hashable]] = []
        for row in matrix:
            labels: set[Hashable] = set()
            for class_index, class_label in enumerate(self.classes_):
                key = _label_key(class_label)
                if 1.0 - row[class_index] <= self.thresholds_[key]:
                    labels.add(key)
            if not labels:
                fallback = self.classes_[int(np.argmax(row))]
                labels.add(_label_key(fallback))
            prediction_sets.append(labels)
        return prediction_sets


def evaluate_prediction_sets(prediction_sets: list[set[Hashable]], labels: np.ndarray) -> dict[str, float]:
    if len(prediction_sets) != len(labels):
        raise ValueError("prediction_sets and labels must have the same length")
    if not prediction_sets:
        return {"conformal_coverage": 0.0, "average_set_size": 0.0, "singleton_rate": 0.0}

    y = [label.item() if hasattr(label, "item") else label for label in labels]
    covered = [label in prediction_set for label, prediction_set in zip(y, prediction_sets)]
    set_sizes = [len(prediction_set) for prediction_set in prediction_sets]
    return {
        "conformal_coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(set_sizes)),
        "singleton_rate": float(np.mean([size == 1 for size in set_sizes])),
    }
