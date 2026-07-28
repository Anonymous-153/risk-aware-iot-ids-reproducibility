from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from iot_ids.leakage import check_forbidden_columns


def _finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


@dataclass
class TabularPreprocessor:
    label_columns: list[str]
    feature_columns_: list[str] = field(default_factory=list)
    numeric_columns_: list[str] = field(default_factory=list)
    categorical_columns_: list[str] = field(default_factory=list)
    medians_: dict[str, float] = field(default_factory=dict)
    means_: dict[str, float] = field(default_factory=dict)
    scales_: dict[str, float] = field(default_factory=dict)
    category_maps_: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_columns_: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        label_set = set(self.label_columns)
        automatic_drops = {"source_file"}
        forbidden = set(check_forbidden_columns(frame, label_columns=self.label_columns))
        self.dropped_columns_ = sorted(label_set | automatic_drops | forbidden)
        self.feature_columns_ = [column for column in frame.columns if column not in set(self.dropped_columns_)]

        self.numeric_columns_ = []
        self.categorical_columns_ = []
        self.medians_ = {}
        self.means_ = {}
        self.scales_ = {}
        self.category_maps_ = {}

        for column in self.feature_columns_:
            if is_numeric_dtype(frame[column]):
                self.numeric_columns_.append(column)
                numeric = _finite_numeric(frame[column])
                median = float(numeric.median()) if not numeric.dropna().empty else 0.0
                filled = numeric.fillna(median)
                mean = float(filled.mean())
                scale = float(filled.std(ddof=0))
                self.medians_[column] = median
                self.means_[column] = mean
                self.scales_[column] = scale if np.isfinite(scale) and scale > 0 else 1.0
            else:
                self.categorical_columns_.append(column)
                values = sorted(frame[column].fillna("<NA>").astype(str).unique().tolist())
                self.category_maps_[column] = {value: index for index, value in enumerate(values)}
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns_:
            raise ValueError("preprocessor must be fit before transform")
        columns: list[np.ndarray] = []
        for column in self.feature_columns_:
            if column in self.numeric_columns_:
                values = _finite_numeric(frame[column]).fillna(self.medians_[column]).to_numpy(float)
                values = (values - self.means_[column]) / self.scales_[column]
            else:
                mapping = self.category_maps_[column]
                values = (
                    frame[column]
                    .fillna("<NA>")
                    .astype(str)
                    .map(lambda value: mapping.get(value, -1))
                    .to_numpy(float)
                )
            columns.append(values.reshape(-1, 1))
        return np.hstack(columns) if columns else np.empty((len(frame), 0))

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)
