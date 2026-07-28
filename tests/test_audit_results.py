import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.audit import audit_results


REQUIRED_METRICS = {
    "dataset": "toy",
    "model": "centroid",
    "ablation": "full",
    "seed": 11,
    "macro_f1": 0.8,
    "mcc": 0.6,
    "balanced_accuracy": 0.8,
    "macro_pr_auc": 0.9,
    "brier_score": 0.1,
    "ece": 0.05,
    "latency_ms_per_sample": 0.2,
    "conformal_coverage": 0.9,
    "average_set_size": 1.2,
    "singleton_rate": 0.85,
}


class AuditResultsTests(unittest.TestCase):
    def test_audit_passes_for_smoke_mode_with_required_files_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame([REQUIRED_METRICS]).to_csv(root / "metrics_by_run.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "toy",
                        "model": "centroid",
                        "ablation": "full",
                        "seed": 11,
                        "row_id": 0,
                        "true_label": 1,
                        "predicted_label": 1,
                        "prediction_set": "1",
                        "covered": 1,
                    }
                ]
            ).to_csv(root / "raw_predictions.csv", index=False)
            pd.DataFrame([{"dataset": "toy", "model": "centroid", "ablation": "full"}]).to_csv(
                root / "summary.csv", index=False
            )

            report = audit_results(root, min_seeds=1, min_conformal_coverage=0.8)

        self.assertTrue(report.passed)
        self.assertEqual(report.failures, [])

    def test_audit_finds_matching_processed_leakage_report_from_project_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            output_dir = project / "results" / "toy"
            processed_dir = project / "data" / "processed" / "toy"
            output_dir.mkdir(parents=True)
            processed_dir.mkdir(parents=True)
            pd.DataFrame([REQUIRED_METRICS]).to_csv(output_dir / "metrics_by_run.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "toy",
                        "model": "centroid",
                        "ablation": "full",
                        "seed": 11,
                        "row_id": 0,
                        "true_label": 1,
                        "predicted_label": 1,
                        "prediction_set": "1",
                        "covered": 1,
                    }
                ]
            ).to_csv(output_dir / "raw_predictions.csv", index=False)
            pd.DataFrame([{"dataset": "toy"}]).to_csv(output_dir / "summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "passed": True,
                        "forbidden_columns": "",
                        "train_test_duplicate_rows": 0,
                        "train_calibration_duplicate_rows": 0,
                        "calibration_test_duplicate_rows": 0,
                    }
                ]
            ).to_csv(processed_dir / "leakage_report.csv", index=False)

            report = audit_results(output_dir, min_seeds=1)

        self.assertTrue(report.passed)
        self.assertEqual(report.warnings, [])

    def test_audit_fails_when_required_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame([REQUIRED_METRICS]).to_csv(root / "metrics_by_run.csv", index=False)

            report = audit_results(root, min_seeds=1)

        self.assertFalse(report.passed)
        self.assertIn("missing required artifact: raw_predictions.csv", report.failures)
        self.assertIn("missing required artifact: summary.csv", report.failures)

    def test_audit_can_skip_missing_raw_predictions_for_public_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame([REQUIRED_METRICS]).to_csv(root / "metrics_by_run.csv", index=False)
            pd.DataFrame([{"dataset": "toy"}]).to_csv(root / "summary.csv", index=False)

            report = audit_results(root, min_seeds=1, require_raw_predictions=False)

        self.assertTrue(report.passed)
        self.assertIn("raw_predictions.csv not included; raw prediction-set audit skipped", report.warnings)

    def test_audit_fails_formal_results_with_too_few_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame([REQUIRED_METRICS]).to_csv(root / "metrics_by_run.csv", index=False)
            pd.DataFrame([{"prediction_set": "1"}]).to_csv(root / "raw_predictions.csv", index=False)
            pd.DataFrame([{"dataset": "toy"}]).to_csv(root / "summary.csv", index=False)

            report = audit_results(root, min_seeds=5)

        self.assertFalse(report.passed)
        self.assertIn("formal results require at least 5 seeds; found 1", report.failures)

    def test_audit_fails_low_conformal_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = dict(REQUIRED_METRICS)
            row["conformal_coverage"] = 0.4
            pd.DataFrame([row]).to_csv(root / "metrics_by_run.csv", index=False)
            pd.DataFrame([{"prediction_set": "1"}]).to_csv(root / "raw_predictions.csv", index=False)
            pd.DataFrame([{"dataset": "toy"}]).to_csv(root / "summary.csv", index=False)

            report = audit_results(root, min_seeds=1, min_conformal_coverage=0.8)

        self.assertFalse(report.passed)
        self.assertIn("minimum conformal coverage 0.4000 is below threshold 0.8000", report.failures)

    def test_audit_ignores_baseline_coverage_for_conformal_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = dict(REQUIRED_METRICS)
            baseline["ablation"] = "baseline"
            baseline["conformal_coverage"] = 0.4
            full = dict(REQUIRED_METRICS)
            full["ablation"] = "full"
            full["conformal_coverage"] = 0.9
            pd.DataFrame([baseline, full]).to_csv(root / "metrics_by_run.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "toy",
                        "model": "centroid",
                        "ablation": "full",
                        "seed": 11,
                        "row_id": 0,
                        "true_label": 1,
                        "predicted_label": 1,
                        "prediction_set": "1",
                        "covered": 1,
                    }
                ]
            ).to_csv(root / "raw_predictions.csv", index=False)
            pd.DataFrame([{"dataset": "toy"}]).to_csv(root / "summary.csv", index=False)

            report = audit_results(root, min_seeds=1, min_conformal_coverage=0.8)

        self.assertTrue(report.passed)

    def test_audit_checks_coverage_aware_full_ablation_for_conformal_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = dict(REQUIRED_METRICS)
            row["ablation"] = "coverage_aware_full"
            row["conformal_coverage"] = 0.4
            pd.DataFrame([row]).to_csv(root / "metrics_by_run.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "toy",
                        "model": "centroid",
                        "ablation": "coverage_aware_full",
                        "seed": 11,
                        "row_id": 0,
                        "true_label": 1,
                        "predicted_label": 1,
                        "prediction_set": "0|1",
                        "covered": 1,
                    }
                ]
            ).to_csv(root / "raw_predictions.csv", index=False)
            pd.DataFrame([{"dataset": "toy"}]).to_csv(root / "summary.csv", index=False)

            report = audit_results(root, min_seeds=1, min_conformal_coverage=0.8)

        self.assertFalse(report.passed)
        self.assertIn("minimum conformal coverage 0.4000 is below threshold 0.8000", report.failures)

    def test_audit_checks_threat_aware_full_ablation_for_conformal_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = dict(REQUIRED_METRICS)
            row["ablation"] = "threat_aware_full"
            row["conformal_coverage"] = 0.4
            pd.DataFrame([row]).to_csv(root / "metrics_by_run.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "toy",
                        "model": "centroid",
                        "ablation": "threat_aware_full",
                        "seed": 11,
                        "row_id": 0,
                        "true_label": 1,
                        "predicted_label": 1,
                        "prediction_set": "0|1",
                        "covered": 1,
                    }
                ]
            ).to_csv(root / "raw_predictions.csv", index=False)
            pd.DataFrame([{"dataset": "toy"}]).to_csv(root / "summary.csv", index=False)

            report = audit_results(root, min_seeds=1, min_conformal_coverage=0.8)

        self.assertFalse(report.passed)
        self.assertIn("minimum conformal coverage 0.4000 is below threshold 0.8000", report.failures)

    def test_audit_checks_risk_aware_full_ablation_for_conformal_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = dict(REQUIRED_METRICS)
            row["ablation"] = "risk_aware_full_b0.05_a0.005"
            row["conformal_coverage"] = 0.4
            pd.DataFrame([row]).to_csv(root / "metrics_by_run.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "toy",
                        "model": "centroid",
                        "ablation": "risk_aware_full_b0.05_a0.005",
                        "seed": 11,
                        "row_id": 0,
                        "true_label": 1,
                        "predicted_label": 1,
                        "prediction_set": "0|1",
                        "covered": 1,
                    }
                ]
            ).to_csv(root / "raw_predictions.csv", index=False)
            pd.DataFrame([{"dataset": "toy"}]).to_csv(root / "summary.csv", index=False)

            report = audit_results(root, min_seeds=1, min_conformal_coverage=0.8)

        self.assertFalse(report.passed)
        self.assertIn("minimum conformal coverage 0.4000 is below threshold 0.8000", report.failures)

    def test_audit_reads_raw_prediction_sets_without_dtype_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame([REQUIRED_METRICS]).to_csv(root / "metrics_by_run.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "toy",
                        "model": "centroid",
                        "ablation": "full",
                        "seed": 11,
                        "row_id": 0,
                        "true_label": 1,
                        "predicted_label": 1,
                        "prediction_set": "1",
                        "covered": 1,
                    },
                    {
                        "dataset": "toy",
                        "model": "centroid",
                        "ablation": "full",
                        "seed": 11,
                        "row_id": 1,
                        "true_label": 0,
                        "predicted_label": 1,
                        "prediction_set": "0|1",
                        "covered": 1,
                    },
                ]
            ).to_csv(root / "raw_predictions.csv", index=False)
            pd.DataFrame([{"dataset": "toy"}]).to_csv(root / "summary.csv", index=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = audit_results(root, min_seeds=1)

        self.assertTrue(report.passed)
        self.assertFalse([warning for warning in caught if "DtypeWarning" in warning.category.__name__])


if __name__ == "__main__":
    unittest.main()
