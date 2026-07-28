from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


FORBIDDEN_COLUMN_PATTERNS = (
    "flowid",
    "srcip",
    "sourceip",
    "dstip",
    "destinationip",
    "attackcat",
    "timestamp",
)


def _normalize_column_name(column: str) -> str:
    return "".join(ch for ch in column.lower() if ch.isalnum())


def check_forbidden_columns(
    frame: pd.DataFrame,
    label_columns: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    labels = {_normalize_column_name(column) for column in (label_columns or [])}
    forbidden: list[str] = []
    for column in frame.columns:
        normalized = _normalize_column_name(str(column))
        if normalized in labels:
            continue
        if any(pattern == normalized or pattern in normalized for pattern in FORBIDDEN_COLUMN_PATTERNS):
            forbidden.append(str(column))
    return forbidden


def _row_hashes(frame: pd.DataFrame, label_columns: list[str] | tuple[str, ...]) -> set[int]:
    label_set = set(label_columns)
    columns = [column for column in frame.columns if column not in label_set]
    if not columns:
        return set()
    normalized = frame.loc[:, columns].copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    normalized = normalized.fillna("<NA>").astype(str)
    return set(pd.util.hash_pandas_object(normalized, index=False).astype("uint64").tolist())


def count_cross_split_duplicates(
    left: pd.DataFrame,
    right: pd.DataFrame,
    label_columns: list[str] | tuple[str, ...],
) -> int:
    common = [column for column in left.columns if column in set(right.columns)]
    if not common:
        return 0
    left_hashes = _row_hashes(left.loc[:, common], label_columns)
    right_hashes = _row_hashes(right.loc[:, common], label_columns)
    return len(left_hashes.intersection(right_hashes))


@dataclass(frozen=True)
class LeakageReport:
    forbidden_columns: list[str]
    train_test_duplicate_rows: int
    train_calibration_duplicate_rows: int = 0
    calibration_test_duplicate_rows: int = 0
    dropped_forbidden_columns: list[str] = field(default_factory=list)
    deduplicated_rows: int = 0

    @property
    def passed(self) -> bool:
        return (
            not self.forbidden_columns
            and self.train_test_duplicate_rows == 0
            and self.train_calibration_duplicate_rows == 0
            and self.calibration_test_duplicate_rows == 0
        )

    def as_dict(self) -> dict[str, int | bool | str]:
        return {
            "passed": self.passed,
            "forbidden_columns": ", ".join(self.forbidden_columns),
            "train_test_duplicate_rows": self.train_test_duplicate_rows,
            "train_calibration_duplicate_rows": self.train_calibration_duplicate_rows,
            "calibration_test_duplicate_rows": self.calibration_test_duplicate_rows,
            "dropped_forbidden_columns": ", ".join(self.dropped_forbidden_columns),
            "deduplicated_rows": self.deduplicated_rows,
        }


def audit_splits(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    label_columns: list[str] | tuple[str, ...],
) -> LeakageReport:
    forbidden = sorted(
        set(check_forbidden_columns(train, label_columns))
        | set(check_forbidden_columns(calibration, label_columns))
        | set(check_forbidden_columns(test, label_columns))
    )
    return LeakageReport(
        forbidden_columns=forbidden,
        train_calibration_duplicate_rows=count_cross_split_duplicates(train, calibration, label_columns),
        train_test_duplicate_rows=count_cross_split_duplicates(train, test, label_columns),
        calibration_test_duplicate_rows=count_cross_split_duplicates(calibration, test, label_columns),
    )
