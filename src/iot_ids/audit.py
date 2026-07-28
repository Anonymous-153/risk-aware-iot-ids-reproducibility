from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_ARTIFACTS = ("metrics_by_run.csv", "summary.csv")
REQUIRED_METRIC_COLUMNS = (
    "dataset",
    "model",
    "ablation",
    "seed",
    "macro_f1",
    "mcc",
    "balanced_accuracy",
    "macro_pr_auc",
    "brier_score",
    "ece",
    "latency_ms_per_sample",
    "conformal_coverage",
    "average_set_size",
    "singleton_rate",
)
REQUIRED_RAW_COLUMNS = (
    "dataset",
    "model",
    "ablation",
    "seed",
    "row_id",
    "true_label",
    "predicted_label",
    "prediction_set",
    "covered",
)

CONFORMAL_ABLATIONS = {"full", "coverage_aware_full", "threat_aware_full"}


@dataclass(frozen=True)
class AuditReport:
    output_dir: Path
    failures: list[str]
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_lines(self) -> list[str]:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Audit status: {status}", f"Output directory: {self.output_dir}"]
        if self.failures:
            lines.append("Failures:")
            lines.extend(f"- {failure}" for failure in self.failures)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return lines


def _missing_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return [column for column in required if column not in frame.columns]


def _candidate_leakage_reports(root: Path) -> list[Path]:
    candidates: list[Path] = []
    resolved = root.resolve()
    dataset_name = resolved.name
    for parent in [resolved, *resolved.parents]:
        project_report = parent / "data" / "processed" / dataset_name / "leakage_report.csv"
        if project_report.exists():
            candidates.append(project_report)
        sibling_report = parent / "data" / "processed" / "*" / "leakage_report.csv"
        candidates.extend(sorted(parent.glob(str(sibling_report.relative_to(parent)))))
    return sorted(set(candidates))


def _is_conformal_ablation(name: object) -> bool:
    normalized = str(name).lower()
    return normalized in CONFORMAL_ABLATIONS or normalized.startswith("risk_aware_full")


def audit_results(
    output_dir: str | Path,
    min_seeds: int = 5,
    min_conformal_coverage: float = 0.8,
    require_raw_predictions: bool = True,
) -> AuditReport:
    root = Path(output_dir)
    failures: list[str] = []
    warnings: list[str] = []

    required_artifacts = (*REQUIRED_ARTIFACTS, "raw_predictions.csv") if require_raw_predictions else REQUIRED_ARTIFACTS
    for artifact in required_artifacts:
        if not (root / artifact).exists():
            failures.append(f"missing required artifact: {artifact}")

    metrics_path = root / "metrics_by_run.csv"
    raw_path = root / "raw_predictions.csv"
    summary_path = root / "summary.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        missing = _missing_columns(metrics, REQUIRED_METRIC_COLUMNS)
        failures.extend(f"metrics_by_run.csv missing column: {column}" for column in missing)
        if not metrics.empty and "seed" in metrics.columns:
            seed_count = metrics["seed"].nunique()
            if seed_count < min_seeds:
                failures.append(f"formal results require at least {min_seeds} seeds; found {seed_count}")
        if not metrics.empty and {"conformal_coverage", "ablation"}.issubset(metrics.columns):
            conformal_rows = metrics[metrics["ablation"].map(_is_conformal_ablation)]
            if conformal_rows.empty:
                warnings.append("no full conformal ablation rows found; coverage threshold was not evaluated")
            min_coverage = float(conformal_rows["conformal_coverage"].min()) if not conformal_rows.empty else 1.0
            if not conformal_rows.empty and min_coverage < min_conformal_coverage:
                failures.append(
                    f"minimum conformal coverage {min_coverage:.4f} is below threshold {min_conformal_coverage:.4f}"
                )
        if metrics.empty:
            failures.append("metrics_by_run.csv contains no rows")

    if raw_path.exists():
        raw = pd.read_csv(raw_path, dtype={"prediction_set": str}, low_memory=False)
        missing = _missing_columns(raw, REQUIRED_RAW_COLUMNS)
        failures.extend(f"raw_predictions.csv missing column: {column}" for column in missing)
        if raw.empty:
            failures.append("raw_predictions.csv contains no rows")
    elif not require_raw_predictions:
        warnings.append("raw_predictions.csv not included; raw prediction-set audit skipped")

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "index" in summary.columns:
            failures.append("summary.csv contains stray index column")
        if summary.empty:
            failures.append("summary.csv contains no rows")

    leakage_reports = _candidate_leakage_reports(root)
    for report_path in leakage_reports:
        report = pd.read_csv(report_path)
        if "passed" in report.columns and not bool(report["passed"].iloc[0]):
            failures.append(f"leakage report failed: {report_path}")
    if not leakage_reports:
        warnings.append("no leakage_report.csv found near output directory; verify data preparation separately")

    return AuditReport(output_dir=root, failures=failures, warnings=warnings)
