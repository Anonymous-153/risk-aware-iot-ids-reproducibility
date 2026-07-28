from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd

from iot_ids.calibration import BinnedProbabilityCalibrator, PlattProbabilityCalibrator
from iot_ids.conformal import (
    ClassConditionalAdaptiveConformal,
    ClassConditionalConformal,
    ClassSpecificAlphaConformal,
    evaluate_prediction_sets,
)
from iot_ids.metrics import (
    balanced_accuracy,
    brier_score_multiclass,
    expected_calibration_error,
    macro_f1,
    matthews_corrcoef_binary,
)
from iot_ids.models import fit_classifier, make_classifier
from iot_ids.preprocessing import TabularPreprocessor


@dataclass(frozen=True)
class ExperimentRequest:
    dataset_name: str
    splits: dict[str, pd.DataFrame]
    output_dir: Path
    model_names: list[str]
    ablations: list[str]
    seeds: list[int]
    alpha: float = 0.1
    label_column: str = "target"
    xgboost_device: str | None = None
    latency_repeats: int = 3


def _ablation_flags(name: str) -> tuple[bool, bool, bool]:
    normalized = name.lower().replace("-", "_")
    if normalized == "baseline":
        return False, False, False
    if normalized in {"cost_sensitive", "cost_sensitive_only"}:
        return True, False, False
    if normalized in {"calibration", "calibration_only"}:
        return False, True, False
    if normalized in {"platt_calibration", "platt_calibration_only"}:
        return False, True, False
    if normalized in {"full", "coverage_aware_full", "threat_aware_full", "platt_full"}:
        return True, True, True
    if normalized.startswith("risk_aware_full") or normalized.startswith("risk_aware_platt_full"):
        return True, True, True
    raise ValueError(f"unknown ablation {name!r}")


def _risk_aware_alpha_by_class(ablation: str, default_alpha: float) -> dict[int, float]:
    normalized = ablation.lower().replace("-", "_")
    if normalized in {"risk_aware_full", "risk_aware_platt_full"}:
        return {0: default_alpha, 1: default_alpha}
    match = re.fullmatch(
        r"risk_aware_(?:platt_)?full_b([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)_a([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)",
        normalized,
    )
    if not match:
        raise ValueError(
            "risk-aware ablations must be named like "
            "'risk_aware_full_b0.05_a0.005'"
        )
    benign_alpha, attack_alpha = (float(value) for value in match.groups())
    return {0: benign_alpha, 1: attack_alpha}


def _conformal_predictor_for_ablation(ablation: str, alpha: float):
    normalized = ablation.lower().replace("-", "_")
    if normalized.startswith("risk_aware_full") or normalized.startswith("risk_aware_platt_full"):
        return ClassSpecificAlphaConformal(alpha=alpha, alpha_by_class=_risk_aware_alpha_by_class(normalized, alpha))
    if normalized in {"coverage_aware_full", "threat_aware_full"}:
        return ClassConditionalAdaptiveConformal(alpha=alpha)
    return ClassConditionalConformal(alpha=alpha)


def _probability_calibrator_for_ablation(ablation: str):
    normalized = ablation.lower().replace("-", "_")
    if "platt" in normalized:
        return PlattProbabilityCalibrator()
    return BinnedProbabilityCalibrator()


def _ensure_positive_label_in_sets(prediction_sets: list[set[int]], classes: np.ndarray) -> list[set[int]]:
    positive_label = 1 if 1 in set(classes.tolist()) else classes[-1]
    positive_key = int(positive_label.item() if hasattr(positive_label, "item") else positive_label)
    return [set(prediction_set) | {positive_key} for prediction_set in prediction_sets]


def _split_calibration_roles(
    calibration: pd.DataFrame,
    label_column: str,
    seed: int,
    probability_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < probability_fraction < 1:
        raise ValueError("probability_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    probability_indices: list[int] = []
    conformal_indices: list[int] = []

    for _, group in calibration[label_column].groupby(calibration[label_column]):
        indices = group.index.to_numpy().copy()
        if len(indices) < 2:
            raise ValueError(
                "full ablation requires at least two calibration examples per class "
                "to separate probability and conformal calibration roles"
            )
        rng.shuffle(indices)
        probability_count = int(round(len(indices) * probability_fraction))
        probability_count = min(max(probability_count, 1), len(indices) - 1)
        probability_indices.extend(indices[:probability_count].tolist())
        conformal_indices.extend(indices[probability_count:].tolist())

    return calibration.loc[sorted(probability_indices)], calibration.loc[sorted(conformal_indices)]


def _positive_probability(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if 1 in set(classes.tolist()):
        index = int(np.where(classes == 1)[0][0])
    else:
        index = probabilities.shape[1] - 1
    return probabilities[:, index]


def _average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels)
    s = np.asarray(scores)
    positive_count = int(np.sum(y == 1))
    if positive_count == 0:
        return 0.0
    order = np.argsort(-s)
    y_sorted = (y[order] == 1).astype(int)
    true_positives = np.cumsum(y_sorted)
    precision = true_positives / (np.arange(len(y_sorted)) + 1)
    return float(np.sum(precision * y_sorted) / positive_count)


def _classwise_set_metrics(
    labels: np.ndarray,
    prediction_sets: list[set[int]],
    positive_label: int = 1,
) -> dict[str, float]:
    y = np.asarray(labels)
    set_sizes = np.asarray([len(prediction_set) for prediction_set in prediction_sets], dtype=float)
    covered = np.asarray([label in prediction_set for label, prediction_set in zip(y, prediction_sets)], dtype=float)
    output: dict[str, float] = {}
    for metric_prefix, mask in [
        ("benign", y != positive_label),
        ("attack", y == positive_label),
    ]:
        if not np.any(mask):
            output[f"{metric_prefix}_coverage"] = 0.0
            output[f"{metric_prefix}_average_set_size"] = 0.0
            output[f"{metric_prefix}_singleton_rate"] = 0.0
            continue
        output[f"{metric_prefix}_coverage"] = float(np.mean(covered[mask]))
        output[f"{metric_prefix}_average_set_size"] = float(np.mean(set_sizes[mask]))
        output[f"{metric_prefix}_singleton_rate"] = float(np.mean(set_sizes[mask] == 1))
    return output


def _metrics_row(
    dataset_name: str,
    model_name: str,
    ablation: str,
    seed: int,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    prediction_sets: list[set[int]],
    model_only_latency_ms_per_sample: float,
    end_to_end_latency_ms_per_sample: float,
    latency_repeats: int,
    latency_batch_size: int,
    probability_calibration_size: int,
    conformal_calibration_size: int,
) -> dict[str, float | int | str]:
    conformal_metrics = evaluate_prediction_sets(prediction_sets, labels)
    classwise_metrics = _classwise_set_metrics(labels, prediction_sets)
    return {
        "dataset": dataset_name,
        "model": model_name,
        "ablation": ablation,
        "seed": seed,
        "macro_f1": macro_f1(labels, predictions, classes),
        "mcc": matthews_corrcoef_binary(labels, predictions),
        "balanced_accuracy": balanced_accuracy(labels, predictions, classes),
        "macro_pr_auc": _average_precision_binary(labels, _positive_probability(probabilities, classes)),
        "brier_score": brier_score_multiclass(probabilities, labels, classes),
        "ece": expected_calibration_error(probabilities, labels, classes=classes),
        "latency_ms_per_sample": end_to_end_latency_ms_per_sample,
        "model_only_latency_ms_per_sample": model_only_latency_ms_per_sample,
        "end_to_end_latency_ms_per_sample": end_to_end_latency_ms_per_sample,
        "latency_repeats": latency_repeats,
        "latency_batch_size": latency_batch_size,
        "probability_calibration_size": probability_calibration_size,
        "conformal_calibration_size": conformal_calibration_size,
        **conformal_metrics,
        **classwise_metrics,
    }


def _raw_prediction_rows(
    dataset_name: str,
    model_name: str,
    ablation: str,
    seed: int,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    prediction_sets: list[set[int]],
) -> list[dict[str, float | int | str]]:
    rows = []
    for row_id, (label, prediction, probability_row, prediction_set) in enumerate(
        zip(labels, predictions, probabilities, prediction_sets)
    ):
        row: dict[str, float | int | str] = {
            "dataset": dataset_name,
            "model": model_name,
            "ablation": ablation,
            "seed": seed,
            "row_id": row_id,
            "true_label": int(label),
            "predicted_label": int(prediction),
            "prediction_set": "|".join(str(int(value)) for value in sorted(prediction_set)),
            "prediction_set_size": len(prediction_set),
            "covered": int(label in prediction_set),
        }
        for class_index, class_label in enumerate(classes):
            row[f"probability_{int(class_label)}"] = float(probability_row[class_index])
        rows.append(row)
    return rows


def _append_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _summary_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = metrics.select_dtypes(include="number").columns.difference(["seed"])
    summary = metrics.groupby(["dataset", "model", "ablation"])[list(numeric_columns)].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary


def _mean_latency_ms_per_sample(callable_prediction, sample_count: int, repeats: int) -> float:
    repeat_count = max(1, int(repeats))
    durations: list[float] = []
    for _ in range(repeat_count):
        start = perf_counter()
        callable_prediction()
        durations.append(perf_counter() - start)
    return 1000.0 * float(np.mean(durations)) / max(sample_count, 1)


def _predict_from_frame(
    frame: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    classifier,
    classes: np.ndarray,
    calibrator,
    conformal,
    ablation: str,
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    features = preprocessor.transform(frame)
    probabilities = classifier.predict_proba(features)
    if calibrator is not None:
        probabilities = calibrator.transform(probabilities)
    predictions = classes[np.argmax(probabilities, axis=1)]
    if conformal is None:
        prediction_sets = [{int(prediction)} for prediction in predictions]
    else:
        prediction_sets = conformal.predict_sets(probabilities)
        if ablation.lower().replace("-", "_") == "threat_aware_full":
            prediction_sets = _ensure_positive_label_in_sets(prediction_sets, classes)
    return predictions, probabilities, prediction_sets


def _write_partial_artifacts(output_dir: Path, metric_rows: list[dict[str, float | int | str]]) -> None:
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics_by_run.csv", index=False)
    _summary_from_metrics(metrics).to_csv(output_dir / "summary.csv", index=False)


def run_experiment(request: ExperimentRequest) -> None:
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name in ["raw_predictions.csv", "metrics_by_run.csv", "summary.csv"]:
        artifact_path = output_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()

    train = request.splits["train"]
    calibration = request.splits["calibration"]
    test = request.splits["test"]
    label_columns = [request.label_column, "label", "Label"]

    metrics_rows: list[dict[str, float | int | str]] = []
    completed_runs = 0
    total_runs = len(request.seeds) * len(request.model_names) * len(request.ablations)

    for seed in request.seeds:
        for model_name in request.model_names:
            for ablation in request.ablations:
                use_costs, use_calibration, use_conformal = _ablation_flags(ablation)
                probability_calibration = calibration
                conformal_calibration = calibration
                if use_calibration and use_conformal:
                    probability_calibration, conformal_calibration = _split_calibration_roles(
                        calibration,
                        request.label_column,
                        seed,
                    )

                preprocessor = TabularPreprocessor(label_columns=label_columns).fit(train)
                x_train = preprocessor.transform(train)
                x_probability_calibration = (
                    preprocessor.transform(probability_calibration) if use_calibration else np.empty((0, x_train.shape[1]))
                )
                x_conformal_calibration = (
                    preprocessor.transform(conformal_calibration) if use_conformal else np.empty((0, x_train.shape[1]))
                )
                x_test = preprocessor.transform(test)
                y_train = train[request.label_column].to_numpy()
                y_probability_calibration = probability_calibration[request.label_column].to_numpy()
                y_conformal_calibration = conformal_calibration[request.label_column].to_numpy()
                y_test = test[request.label_column].to_numpy()

                classifier = make_classifier(model_name, seed, xgboost_device=request.xgboost_device)
                fit_classifier(classifier, x_train, y_train, use_sample_weights=use_costs)
                classes = np.asarray(classifier.classes_)
                probability_calibration_probabilities = (
                    classifier.predict_proba(x_probability_calibration) if use_calibration else None
                )
                conformal_calibration_probabilities = (
                    classifier.predict_proba(x_conformal_calibration) if use_conformal else None
                )

                model_only_latency_ms_per_sample = _mean_latency_ms_per_sample(
                    lambda: classifier.predict_proba(x_test),
                    len(test),
                    request.latency_repeats,
                )
                test_probabilities = classifier.predict_proba(x_test)

                calibrator = None
                if use_calibration:
                    assert probability_calibration_probabilities is not None
                    calibrator = _probability_calibrator_for_ablation(ablation).fit(
                        probability_calibration_probabilities,
                        y_probability_calibration,
                        classes,
                    )
                    if conformal_calibration_probabilities is not None:
                        conformal_calibration_probabilities = calibrator.transform(conformal_calibration_probabilities)
                    test_probabilities = calibrator.transform(test_probabilities)

                predictions = classes[np.argmax(test_probabilities, axis=1)]
                conformal = None
                if use_conformal:
                    assert conformal_calibration_probabilities is not None
                    conformal = _conformal_predictor_for_ablation(ablation, request.alpha).fit(
                        conformal_calibration_probabilities,
                        y_conformal_calibration,
                        classes=classes,
                    )
                    prediction_sets = conformal.predict_sets(test_probabilities)
                    if ablation.lower().replace("-", "_") == "threat_aware_full":
                        prediction_sets = _ensure_positive_label_in_sets(prediction_sets, classes)
                else:
                    prediction_sets = [{int(prediction)} for prediction in predictions]

                end_to_end_latency_ms_per_sample = _mean_latency_ms_per_sample(
                    lambda: _predict_from_frame(
                        test,
                        preprocessor,
                        classifier,
                        classes,
                        calibrator,
                        conformal,
                        ablation,
                    ),
                    len(test),
                    request.latency_repeats,
                )

                metrics_rows.append(
                    _metrics_row(
                        request.dataset_name,
                        model_name,
                        ablation,
                        seed,
                        y_test,
                        predictions,
                        test_probabilities,
                        classes,
                        prediction_sets,
                        model_only_latency_ms_per_sample,
                        end_to_end_latency_ms_per_sample,
                        max(1, int(request.latency_repeats)),
                        len(test),
                        len(probability_calibration) if use_calibration else 0,
                        len(conformal_calibration) if use_conformal else 0,
                    )
                )
                _append_rows(
                    output_dir / "raw_predictions.csv",
                    _raw_prediction_rows(
                        request.dataset_name,
                        model_name,
                        ablation,
                        seed,
                        y_test,
                        predictions,
                        test_probabilities,
                        classes,
                        prediction_sets,
                    ),
                )
                _write_partial_artifacts(output_dir, metrics_rows)
                completed_runs += 1
                print(
                    f"[{request.dataset_name}] completed {completed_runs}/{total_runs}: "
                    f"seed={seed} model={model_name} ablation={ablation}",
                    flush=True,
                )
