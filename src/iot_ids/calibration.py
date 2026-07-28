from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)


@dataclass
class BinnedProbabilityCalibrator:
    n_bins: int = 10
    bin_values_: np.ndarray | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> "BinnedProbabilityCalibrator":
        matrix = np.asarray(probabilities, dtype=float)
        y = np.asarray(labels)
        class_values = np.asarray(classes)
        self.bin_values_ = np.zeros((matrix.shape[1], self.n_bins), dtype=float)

        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        for class_index, class_label in enumerate(class_values):
            target = (y == class_label).astype(float)
            for bin_index in range(self.n_bins):
                lower = edges[bin_index]
                upper = edges[bin_index + 1]
                if bin_index == 0:
                    mask = (matrix[:, class_index] >= lower) & (matrix[:, class_index] <= upper)
                else:
                    mask = (matrix[:, class_index] > lower) & (matrix[:, class_index] <= upper)
                if np.any(mask):
                    self.bin_values_[class_index, bin_index] = float(np.mean(target[mask]))
                else:
                    self.bin_values_[class_index, bin_index] = (lower + upper) / 2
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.bin_values_ is None:
            raise ValueError("calibrator must be fit before transform")
        matrix = np.asarray(probabilities, dtype=float)
        calibrated = np.zeros_like(matrix)
        bin_ids = np.minimum((matrix * self.n_bins).astype(int), self.n_bins - 1)
        for class_index in range(matrix.shape[1]):
            calibrated[:, class_index] = self.bin_values_[class_index, bin_ids[:, class_index]]
        row_sums = calibrated.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] == 0
        calibrated[zero_rows] = matrix[zero_rows]
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.clip(row_sums, 1e-12, None)


@dataclass
class PlattProbabilityCalibrator:
    """One-vs-rest sigmoid calibration fitted on held-out probabilities."""

    models_: list[object] | None = None
    classes_: np.ndarray | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> "PlattProbabilityCalibrator":
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for Platt calibration") from exc

        matrix = _clip_probabilities(probabilities)
        y = np.asarray(labels)
        class_values = np.asarray(classes)
        if matrix.ndim != 2:
            raise ValueError("probabilities must be a 2D array")
        if matrix.shape[1] != len(class_values):
            raise ValueError("number of probability columns must match number of classes")

        self.classes_ = class_values
        self.models_ = []
        for class_index, class_label in enumerate(class_values):
            target = (y == class_label).astype(int)
            if len(np.unique(target)) < 2:
                raise ValueError(f"cannot fit Platt calibrator without positives and negatives for {class_label!r}")
            logits = np.log(matrix[:, class_index] / (1.0 - matrix[:, class_index])).reshape(-1, 1)
            model = LogisticRegression(solver="lbfgs")
            model.fit(logits, target)
            self.models_.append(model)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.models_ is None or self.classes_ is None:
            raise ValueError("calibrator must be fit before transform")
        matrix = _clip_probabilities(probabilities)
        if matrix.ndim != 2:
            raise ValueError("probabilities must be a 2D array")
        if matrix.shape[1] != len(self.models_):
            raise ValueError("number of probability columns must match fitted classes")

        calibrated = np.zeros_like(matrix)
        for class_index, model in enumerate(self.models_):
            logits = np.log(matrix[:, class_index] / (1.0 - matrix[:, class_index])).reshape(-1, 1)
            calibrated[:, class_index] = model.predict_proba(logits)[:, 1]
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.clip(row_sums, 1e-12, None)
