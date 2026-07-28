from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_LABELS = {
    "macro_f1": "Macro-F1",
    "mcc": "MCC",
    "balanced_accuracy": "Bal. Acc.",
    "macro_pr_auc": "Macro PR-AUC",
    "brier_score": "Brier",
    "ece": "ECE",
    "conformal_coverage": "Coverage",
    "attack_coverage": "Attack Cov.",
    "benign_coverage": "Benign Cov.",
    "average_set_size": "Set Size",
    "singleton_rate": "Singleton",
    "latency_ms_per_sample": "End-to-end latency (ms/sample)",
    "model_only_latency_ms_per_sample": "Model latency",
    "end_to_end_latency_ms_per_sample": "End-to-end latency",
}

DISPLAY_LABELS = {
    "cic_iot_diad_2024": "CIC IoT-DIAD",
    "cic_iot_diad_2024_raw_proportion": "CIC raw-prop.",
    "unsw_nb15": "UNSW-NB15",
    "logistic_regression": "LogReg",
    "random_forest": "RF",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "baseline": "baseline",
    "calibration": "calibration",
    "cost_sensitive": "cost-sensitive",
    "full": "full",
    "platt_calibration": "Platt calibration",
    "platt_full": "Platt full",
    "threat_aware_full": "threat-aware full",
    "risk_aware_full_b0.01_a0.001": "risk-aware 0.01/0.001",
    "risk_aware_full_b0.01_a0.005": "risk-aware 0.01/0.005",
    "risk_aware_full_b0.01_a0.01": "risk-aware 0.01/0.01",
    "risk_aware_full_b0.05_a0.001": "risk-aware 0.05/0.001",
    "risk_aware_full_b0.05_a0.005": "risk-aware 0.05/0.005",
    "risk_aware_full_b0.05_a0.01": "risk-aware 0.05/0.01",
    "risk_aware_platt_full_b0.01_a0.001": "Platt risk-aware 0.01/0.001",
    "risk_aware_platt_full_b0.01_a0.005": "Platt risk-aware 0.01/0.005",
    "risk_aware_platt_full_b0.01_a0.01": "Platt risk-aware 0.01/0.01",
    "risk_aware_platt_full_b0.05_a0.001": "Platt risk-aware 0.05/0.001",
    "risk_aware_platt_full_b0.05_a0.005": "Platt risk-aware 0.05/0.005",
    "risk_aware_platt_full_b0.05_a0.01": "Platt risk-aware 0.05/0.01",
}


def _escape_latex(value: object) -> str:
    return str(value).replace("_", r"\_")


def _display_value(value: object) -> str:
    return DISPLAY_LABELS.get(str(value), str(value))


def _format_mean_std(row: pd.Series, metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std_column = f"{metric}_std"
    if std_column not in row or pd.isna(row[std_column]):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {float(row[std_column]):.3f}"


def metric_table_to_latex(
    summary: pd.DataFrame,
    metrics: list[str],
    caption: str = "Machine-generated aggregate experiment metrics.",
    label: str = "tab:main-metrics",
) -> str:
    required = {"dataset", "model", "ablation"} | {f"{metric}_mean" for metric in metrics}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError("summary is missing required columns: " + ", ".join(missing))

    ordered = summary.sort_values(["dataset", "model", "ablation"]).reset_index(drop=True)
    metric_headers = [METRIC_LABELS.get(metric, metric.replace("_", " ").title()) for metric in metrics]
    columns = ["Dataset", "Model", "Ablation", *metric_headers]
    alignment = "lll" + "r" * len(metrics)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{@{{}}{alignment}@{{}}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for _, row in ordered.iterrows():
        values = [
            _escape_latex(_display_value(row["dataset"])),
            _escape_latex(_display_value(row["model"])),
            _escape_latex(_display_value(row["ablation"])),
            *[_format_mean_std(row, metric) for metric in metrics],
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_metric_tables(
    summary_path: str | Path,
    output_dir: str | Path,
    metrics: list[str] | None = None,
    caption: str = "Machine-generated aggregate experiment metrics.",
    label: str = "tab:main-metrics",
) -> list[Path]:
    selected_metrics = metrics or ["macro_f1", "mcc", "balanced_accuracy", "ece", "conformal_coverage"]
    summary = pd.read_csv(summary_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "main_metrics.tex"
    output_path.write_text(
        metric_table_to_latex(summary, selected_metrics, caption=caption, label=label),
        encoding="utf-8",
    )
    return [output_path]


def coverage_by_class(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "ablation", "true_label", "covered"}
    missing = sorted(required.difference(raw_predictions.columns))
    if missing:
        raise ValueError("raw predictions are missing required columns: " + ", ".join(missing))

    frame = raw_predictions.copy()
    frame["covered"] = pd.to_numeric(frame["covered"], errors="raise")
    summary = (
        frame.groupby(["dataset", "model", "ablation", "true_label"], as_index=False)
        .agg(coverage=("covered", "mean"), sample_count=("covered", "size"))
        .sort_values(["dataset", "model", "ablation", "true_label"])
        .reset_index(drop=True)
    )
    return summary


def _prediction_set_size(value: object) -> int:
    text = str(value)
    if not text or text == "nan":
        return 0
    return len(text.split("|"))


def set_size_distribution(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "ablation", "prediction_set"}
    missing = sorted(required.difference(raw_predictions.columns))
    if missing:
        raise ValueError("raw predictions are missing required columns: " + ", ".join(missing))

    frame = raw_predictions.copy()
    frame["set_size"] = frame["prediction_set"].map(_prediction_set_size)
    distribution = (
        frame.groupby(["dataset", "model", "ablation", "set_size"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["dataset", "model", "ablation", "set_size"])
        .reset_index(drop=True)
    )
    return distribution


def confidence_diagnostics(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "ablation", "seed", "true_label", "covered"}
    missing = sorted(required.difference(raw_predictions.columns))
    if missing:
        raise ValueError("raw predictions are missing required columns: " + ", ".join(missing))
    probability_columns = sorted(column for column in raw_predictions.columns if str(column).startswith("probability_"))
    if not probability_columns:
        raise ValueError("raw predictions do not contain probability_* columns")

    frame = raw_predictions.copy()
    frame["covered"] = pd.to_numeric(frame["covered"], errors="raise").astype(int)
    for column in probability_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    probability_frame = frame[probability_columns]
    frame["max_probability"] = probability_frame.max(axis=1)
    sorted_probabilities = probability_frame.to_numpy(dtype=float)
    if sorted_probabilities.shape[1] >= 2:
        top_two = -np.sort(-sorted_probabilities, axis=1)[:, :2]
        frame["probability_margin"] = top_two[:, 0] - top_two[:, 1]
    else:
        frame["probability_margin"] = 0.0

    true_label_probability: list[float] = []
    for _, row in frame.iterrows():
        probability_column = f"probability_{int(row['true_label'])}"
        if probability_column not in frame.columns:
            true_label_probability.append(float("nan"))
        else:
            true_label_probability.append(float(row[probability_column]))
    frame["true_label_probability"] = true_label_probability
    frame["max_probability_6dp"] = frame["max_probability"].round(6)
    frame["true_label_probability_6dp"] = frame["true_label_probability"].round(6)

    diagnostics = (
        frame.groupby(["dataset", "model", "ablation", "covered"], as_index=False)
        .agg(
            sample_count=("covered", "size"),
            mean_true_label_probability=("true_label_probability", "mean"),
            median_true_label_probability=("true_label_probability", "median"),
            mean_max_probability=("max_probability", "mean"),
            median_max_probability=("max_probability", "median"),
            mean_probability_margin=("probability_margin", "mean"),
            median_probability_margin=("probability_margin", "median"),
            unique_max_probability_6dp=("max_probability_6dp", "nunique"),
            unique_true_label_probability_6dp=("true_label_probability_6dp", "nunique"),
        )
        .sort_values(["dataset", "model", "ablation", "covered"])
        .reset_index(drop=True)
    )
    return diagnostics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_environment_metadata(
    config_path: str | Path,
    command: list[str],
    package_versions: dict[str, str],
    gpu_info: str | None,
) -> dict[str, object]:
    resolved_config = Path(config_path)
    return {
        "command": list(command),
        "config_path": str(resolved_config),
        "config_sha256": _sha256(resolved_config),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu_info": gpu_info or "not detected",
        "package_versions": dict(sorted(package_versions.items())),
    }
