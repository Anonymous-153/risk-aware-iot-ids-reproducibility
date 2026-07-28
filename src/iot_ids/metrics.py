from __future__ import annotations

from collections import Counter

import numpy as np


def brier_score_multiclass(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> float:
    matrix = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels)
    class_list = list(classes)
    class_to_index = {label: index for index, label in enumerate(class_list)}
    one_hot = np.zeros_like(matrix)
    for row_index, label in enumerate(y):
        one_hot[row_index, class_to_index[label]] = 1.0
    return float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    classes: np.ndarray | None = None,
) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    matrix = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels)
    class_values = np.asarray(classes if classes is not None else np.arange(matrix.shape[1]))
    predictions = class_values[np.argmax(matrix, axis=1)]
    confidences = np.max(matrix, axis=1)
    correct = predictions == y

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(correct[mask]))
        confidence = float(np.mean(confidences[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return float(ece)


def macro_f1(labels: np.ndarray, predictions: np.ndarray, classes: np.ndarray) -> float:
    scores: list[float] = []
    for class_label in classes:
        tp = int(np.sum((labels == class_label) & (predictions == class_label)))
        fp = int(np.sum((labels != class_label) & (predictions == class_label)))
        fn = int(np.sum((labels == class_label) & (predictions != class_label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray, classes: np.ndarray) -> float:
    recalls: list[float] = []
    for class_label in classes:
        support = int(np.sum(labels == class_label))
        if support == 0:
            continue
        recalls.append(float(np.mean(predictions[labels == class_label] == class_label)))
    return float(np.mean(recalls)) if recalls else 0.0


def matthews_corrcoef_binary(labels: np.ndarray, predictions: np.ndarray, positive_label: int | str = 1) -> float:
    y = np.asarray(labels)
    p = np.asarray(predictions)
    tp = int(np.sum((y == positive_label) & (p == positive_label)))
    tn = int(np.sum((y != positive_label) & (p != positive_label)))
    fp = int(np.sum((y != positive_label) & (p == positive_label)))
    fn = int(np.sum((y == positive_label) & (p != positive_label)))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(((tp * tn) - (fp * fn)) / denominator) if denominator else 0.0


def class_weights(labels: np.ndarray) -> dict[object, float]:
    counts = Counter(labels.tolist())
    total = sum(counts.values())
    n_classes = len(counts)
    return {label: total / (n_classes * count) for label, count in counts.items()}
