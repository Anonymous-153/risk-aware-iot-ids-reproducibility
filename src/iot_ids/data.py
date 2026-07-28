from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from iot_ids.leakage import LeakageReport, audit_splits, check_forbidden_columns


DEFAULT_LABEL_COLUMNS = ("target", "label", "Label", "class", "Class", "attack", "Attack")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label_column: str
    positive_labels: list[str]
    derive_label_from_path: bool = False
    benign_path_markers: tuple[str, ...] = ("Benign",)
    benign_label: str = "Benign"
    attack_label: str = "Attack"


def infer_label_column(frame: pd.DataFrame) -> str:
    candidates = ("label", "Label", "class", "Class", "attack", "Attack")
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError("could not infer label column; provide one in the dataset spec")


def normalize_binary_labels(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    if spec.label_column not in frame.columns:
        raise ValueError(f"missing label column {spec.label_column!r}")
    normalized = frame.copy()
    positives = {label.lower() for label in spec.positive_labels}
    normalized["target"] = normalized[spec.label_column].astype(str).str.lower().isin(positives).astype(int)
    return normalized


def infer_reference_columns(paths: Sequence[str | Path], label_column: str) -> list[str] | None:
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            continue
        columns = list(pd.read_csv(csv_path, nrows=0).columns)
        if label_column in columns:
            return [str(column) for column in columns]
    return None


def _path_label(path: Path, spec: DatasetSpec) -> str:
    parts = {part.casefold() for part in path.parts}
    benign_markers = {marker.casefold() for marker in spec.benign_path_markers}
    return spec.benign_label if parts.intersection(benign_markers) else spec.attack_label


def read_csv_with_optional_header(
    path: str | Path,
    spec: DatasetSpec | None = None,
    reference_columns: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, nrows=nrows)
    if spec and spec.label_column not in frame.columns and reference_columns:
        if len(frame.columns) == len(reference_columns):
            frame = pd.read_csv(csv_path, header=None, names=list(reference_columns), nrows=nrows)
    if spec and spec.derive_label_from_path:
        frame[spec.label_column] = _path_label(csv_path, spec)
    return frame


def _read_csv_chunks_with_optional_header(
    path: str | Path,
    spec: DatasetSpec,
    reference_columns: Sequence[str] | None,
    chunksize: int,
):
    csv_path = Path(path)
    preview = pd.read_csv(csv_path, nrows=0)
    read_kwargs: dict[str, object] = {"chunksize": chunksize}
    if spec.label_column not in preview.columns and reference_columns and len(preview.columns) == len(reference_columns):
        read_kwargs.update({"header": None, "names": list(reference_columns)})
    return pd.read_csv(csv_path, **read_kwargs)


def read_csv_many(paths: list[str | Path], spec: DatasetSpec | None = None) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one CSV path is required")
    frames = []
    reference_columns = infer_reference_columns(paths, spec.label_column) if spec else None
    for path in paths:
        csv_path = Path(path)
        frame = read_csv_with_optional_header(csv_path, spec=spec, reference_columns=reference_columns)
        frame["source_file"] = csv_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write_sampling_manifest(
    manifest_path: str | Path,
    dataset_name: str,
    seen_counts: dict[tuple[str, int], int],
    selected: pd.DataFrame,
) -> None:
    selected_counts = selected.groupby(["source_file", "target"]).size().to_dict()
    rows = []
    for (source_file, target), rows_seen in sorted(seen_counts.items()):
        rows.append(
            {
                "dataset": dataset_name,
                "path": source_file,
                "target": target,
                "rows_seen": rows_seen,
                "rows_selected": int(selected_counts.get((source_file, target), 0)),
            }
        )
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)


def sample_csv_many(
    paths: list[str | Path],
    spec: DatasetSpec,
    max_rows_per_label: int,
    seed: int,
    chunksize: int = 100_000,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one CSV path is required")
    if max_rows_per_label <= 0:
        raise ValueError("max_rows_per_label must be positive")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    rng = np.random.default_rng(seed)
    reference_columns = infer_reference_columns(paths, spec.label_column)
    reservoirs: dict[int, pd.DataFrame] = {}
    seen_counts: dict[tuple[str, int], int] = {}

    for path in paths:
        csv_path = Path(path)
        for chunk in _read_csv_chunks_with_optional_header(csv_path, spec, reference_columns, chunksize):
            if spec.derive_label_from_path:
                chunk[spec.label_column] = _path_label(csv_path, spec)
            chunk["source_file"] = str(csv_path)
            normalized = normalize_binary_labels(chunk, spec)
            normalized["_sample_key"] = rng.random(len(normalized))

            for target, group in normalized.groupby("target"):
                target_key = int(target)
                count_key = (str(csv_path), target_key)
                seen_counts[count_key] = seen_counts.get(count_key, 0) + len(group)
                current = reservoirs.get(target_key)
                candidates = group if current is None else pd.concat([current, group], ignore_index=True)
                if len(candidates) > max_rows_per_label:
                    candidates = candidates.nsmallest(max_rows_per_label, "_sample_key")
                reservoirs[target_key] = candidates.reset_index(drop=True)

    if not reservoirs:
        raise ValueError("sampling produced no rows")

    sampled = pd.concat([reservoirs[target] for target in sorted(reservoirs)], ignore_index=True)
    sampled = sampled.drop(columns=["_sample_key"]).reset_index(drop=True)
    if manifest_path is not None:
        _write_sampling_manifest(manifest_path, spec.name, seen_counts, sampled)
    return sampled


def _count_csv_many_by_label(
    paths: list[str | Path],
    spec: DatasetSpec,
    chunksize: int,
) -> tuple[dict[tuple[str, int], int], dict[int, int]]:
    reference_columns = infer_reference_columns(paths, spec.label_column)
    seen_counts: dict[tuple[str, int], int] = {}
    label_counts: dict[int, int] = {}
    for path in paths:
        csv_path = Path(path)
        for chunk in _read_csv_chunks_with_optional_header(csv_path, spec, reference_columns, chunksize):
            if spec.derive_label_from_path:
                chunk[spec.label_column] = _path_label(csv_path, spec)
            normalized = normalize_binary_labels(chunk, spec)
            for target, group in normalized.groupby("target"):
                target_key = int(target)
                count_key = (str(csv_path), target_key)
                seen_counts[count_key] = seen_counts.get(count_key, 0) + len(group)
                label_counts[target_key] = label_counts.get(target_key, 0) + len(group)
    return seen_counts, label_counts


def _proportional_targets(
    label_counts: dict[int, int],
    max_total_rows: int,
    min_rows_per_label: int,
) -> dict[int, int]:
    total_seen = sum(label_counts.values())
    if total_seen == 0:
        raise ValueError("sampling produced no rows")
    targets = {
        label: min(count, max(min_rows_per_label, int(round(max_total_rows * count / total_seen))))
        for label, count in label_counts.items()
    }
    while sum(targets.values()) > max_total_rows:
        reducible = [label for label, target in targets.items() if target > min(min_rows_per_label, label_counts[label])]
        if not reducible:
            break
        label = max(reducible, key=lambda item: targets[item])
        targets[label] -= 1
    while sum(targets.values()) < max_total_rows:
        expandable = [label for label, target in targets.items() if target < label_counts[label]]
        if not expandable:
            break
        label = max(expandable, key=lambda item: label_counts[item] - targets[item])
        targets[label] += 1
    return targets


def sample_csv_many_proportional(
    paths: list[str | Path],
    spec: DatasetSpec,
    max_total_rows: int,
    min_rows_per_label: int,
    seed: int,
    chunksize: int = 100_000,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one CSV path is required")
    if max_total_rows <= 0:
        raise ValueError("max_total_rows must be positive")
    if min_rows_per_label <= 0:
        raise ValueError("min_rows_per_label must be positive")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    seen_counts, label_counts = _count_csv_many_by_label(paths, spec, chunksize)
    target_counts = _proportional_targets(label_counts, max_total_rows, min_rows_per_label)
    rng = np.random.default_rng(seed)
    reference_columns = infer_reference_columns(paths, spec.label_column)
    reservoirs: dict[int, pd.DataFrame] = {}

    for path in paths:
        csv_path = Path(path)
        for chunk in _read_csv_chunks_with_optional_header(csv_path, spec, reference_columns, chunksize):
            if spec.derive_label_from_path:
                chunk[spec.label_column] = _path_label(csv_path, spec)
            chunk["source_file"] = str(csv_path)
            normalized = normalize_binary_labels(chunk, spec)
            normalized["_sample_key"] = rng.random(len(normalized))

            for target, group in normalized.groupby("target"):
                target_key = int(target)
                target_count = target_counts.get(target_key, 0)
                if target_count <= 0:
                    continue
                current = reservoirs.get(target_key)
                candidates = group if current is None else pd.concat([current, group], ignore_index=True)
                if len(candidates) > target_count:
                    candidates = candidates.nsmallest(target_count, "_sample_key")
                reservoirs[target_key] = candidates.reset_index(drop=True)

    if not reservoirs:
        raise ValueError("sampling produced no rows")

    sampled = pd.concat([reservoirs[target] for target in sorted(reservoirs)], ignore_index=True)
    sampled = sampled.drop(columns=["_sample_key"]).reset_index(drop=True)
    if manifest_path is not None:
        _write_sampling_manifest(manifest_path, spec.name, seen_counts, sampled)
    return sampled


def _stratified_indices(labels: pd.Series, seed: int, test_size: float, calibration_size: float) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    calibration_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in labels.groupby(labels):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_size))) if len(indices) >= 3 else 1
        remaining = len(indices) - n_test
        n_calibration = max(1, int(round(remaining * calibration_size))) if remaining >= 3 else max(0, remaining - 1)
        test_indices.extend(indices[:n_test].tolist())
        calibration_indices.extend(indices[n_test : n_test + n_calibration].tolist())
        train_indices.extend(indices[n_test + n_calibration :].tolist())

    return {
        "train": sorted(train_indices),
        "calibration": sorted(calibration_indices),
        "test": sorted(test_indices),
    }


def _label_columns(spec: DatasetSpec | None = None) -> list[str]:
    labels = list(DEFAULT_LABEL_COLUMNS)
    if spec and spec.label_column not in labels:
        labels.append(spec.label_column)
    return labels


def _deduplicate_for_random_split(frame: pd.DataFrame, label_columns: list[str]) -> tuple[pd.DataFrame, int]:
    forbidden = set(check_forbidden_columns(frame, label_columns=label_columns)) | {"source_file"}
    label_set = set(label_columns)
    dedup_columns = [column for column in frame.columns if column not in label_set and column not in forbidden]
    if not dedup_columns:
        return frame.reset_index(drop=True), 0

    deduplicated = frame.drop_duplicates(subset=dedup_columns, keep="first").reset_index(drop=True)
    return deduplicated, len(frame) - len(deduplicated)


def split_frame(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    seed: int,
    test_size: float = 0.2,
    calibration_size: float = 0.2,
) -> dict[str, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < calibration_size < 1:
        raise ValueError("calibration_size must be between 0 and 1")

    normalized = normalize_binary_labels(frame, spec)
    label_columns = _label_columns(spec)
    deduplicated, deduplicated_rows = _deduplicate_for_random_split(normalized, label_columns)
    split_indices = _stratified_indices(deduplicated["target"], seed, test_size, calibration_size)
    splits = {name: deduplicated.loc[indices].reset_index(drop=True) for name, indices in split_indices.items()}
    for split in splits.values():
        split.attrs["deduplicated_rows"] = deduplicated_rows
    return splits


def split_train_calibration(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    spec: DatasetSpec,
    seed: int,
    calibration_size: float = 0.2,
) -> dict[str, pd.DataFrame]:
    if not 0 < calibration_size < 1:
        raise ValueError("calibration_size must be between 0 and 1")
    normalized_train = normalize_binary_labels(train_frame, spec)
    normalized_test = normalize_binary_labels(test_frame, spec)
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    calibration_indices: list[int] = []

    for _, group in normalized_train["target"].groupby(normalized_train["target"]):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_calibration = max(1, int(round(len(indices) * calibration_size))) if len(indices) >= 3 else 1
        calibration_indices.extend(indices[:n_calibration].tolist())
        train_indices.extend(indices[n_calibration:].tolist())

    return {
        "train": normalized_train.loc[sorted(train_indices)].reset_index(drop=True),
        "calibration": normalized_train.loc[sorted(calibration_indices)].reset_index(drop=True),
        "test": normalized_test.reset_index(drop=True),
    }


def _drop_forbidden_columns(splits: dict[str, pd.DataFrame], label_columns: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    dropped = sorted(
        set().union(*(set(check_forbidden_columns(frame, label_columns=label_columns)) for frame in splits.values()))
        | {"source_file"}
    )
    cleaned = {name: frame.drop(columns=[column for column in dropped if column in frame.columns]) for name, frame in splits.items()}
    return cleaned, dropped


def write_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path, dataset_name: str) -> None:
    destination = Path(output_dir) / dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    label_columns = list(DEFAULT_LABEL_COLUMNS)
    deduplicated_rows = max((int(frame.attrs.get("deduplicated_rows", 0)) for frame in splits.values()), default=0)
    cleaned_splits, dropped = _drop_forbidden_columns(splits, label_columns=label_columns)
    for split_name, frame in cleaned_splits.items():
        frame.to_csv(destination / f"{split_name}.csv", index=False)
    report = audit_splits(
        cleaned_splits["train"],
        cleaned_splits["calibration"],
        cleaned_splits["test"],
        label_columns=label_columns,
    )
    report = LeakageReport(
        forbidden_columns=report.forbidden_columns,
        train_test_duplicate_rows=report.train_test_duplicate_rows,
        train_calibration_duplicate_rows=report.train_calibration_duplicate_rows,
        calibration_test_duplicate_rows=report.calibration_test_duplicate_rows,
        dropped_forbidden_columns=dropped,
        deduplicated_rows=deduplicated_rows,
    )
    pd.DataFrame([report.as_dict()]).to_csv(destination / "leakage_report.csv", index=False)
