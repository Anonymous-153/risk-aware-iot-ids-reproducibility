import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_manuscript_tables import (
    calibration_baseline_table_to_latex,
    deployment_cost_table_to_latex,
    unsw_failure_diagnostic_table_to_latex,
    write_manuscript_tables,
)


def _summary_rows() -> pd.DataFrame:
    rows = []
    for dataset in ["cic_iot_diad_2024", "unsw_nb15"]:
        for model in ["logistic_regression", "random_forest"]:
            for ablation in [
                "full",
                "risk_aware_full_b0.01_a0.001",
                "platt_full",
                "risk_aware_platt_full_b0.01_a0.001",
            ]:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "ablation": ablation,
                        "macro_f1_mean": 0.7,
                        "macro_f1_std": 0.01,
                        "ece_mean": 0.1,
                        "ece_std": 0.01,
                        "brier_score_mean": 0.2,
                        "brier_score_std": 0.02,
                        "conformal_coverage_mean": 0.9,
                        "conformal_coverage_std": 0.01,
                        "attack_coverage_mean": 0.8,
                        "attack_coverage_std": 0.02,
                        "average_set_size_mean": 1.4,
                        "average_set_size_std": 0.03,
                        "singleton_rate_mean": 0.6,
                        "singleton_rate_std": 0.04,
                        "model_only_latency_ms_per_sample_mean": 0.001,
                        "model_only_latency_ms_per_sample_std": 0.0001,
                        "end_to_end_latency_ms_per_sample_mean": 0.008,
                        "end_to_end_latency_ms_per_sample_std": 0.0002,
                    }
                )
    return pd.DataFrame(rows)


class ManuscriptTableTests(unittest.TestCase):
    def test_deployment_table_separates_model_only_and_end_to_end_latency(self) -> None:
        raw_summary = _summary_rows().assign(dataset="cic_iot_diad_2024_raw_proportion")

        latex = deployment_cost_table_to_latex(_summary_rows(), raw_summary)

        self.assertIn("Model-only", latex)
        self.assertIn("End-to-end", latex)

    def test_failure_table_reports_probability_margin(self) -> None:
        diagnostics = pd.DataFrame(
            [
                {
                    "dataset": "unsw_nb15",
                    "model": "logistic_regression",
                    "ablation": "risk_aware_full_b0.01_a0.001",
                    "covered": 0,
                    "sample_count": 5,
                    "mean_true_label_probability": 0.2,
                    "mean_max_probability": 0.8,
                    "mean_probability_margin": 0.6,
                    "unique_max_probability_6dp": 3,
                },
                {
                    "dataset": "unsw_nb15",
                    "model": "random_forest",
                    "ablation": "risk_aware_full_b0.01_a0.001",
                    "covered": 0,
                    "sample_count": 7,
                    "mean_true_label_probability": 0.1,
                    "mean_max_probability": 0.9,
                    "mean_probability_margin": 0.8,
                    "unique_max_probability_6dp": 2,
                },
            ]
        )

        latex = unsw_failure_diagnostic_table_to_latex(_summary_rows(), diagnostics)

        self.assertIn("Margin", latex)
        self.assertIn("Unique", latex)

    def test_calibration_baseline_table_includes_platt_rows(self) -> None:
        latex = calibration_baseline_table_to_latex(_summary_rows())

        self.assertIn("Risk-aware Platt", latex)

    def test_write_manuscript_tables_writes_main_manuscript_inputs(self) -> None:
        diagnostics = pd.DataFrame(
            [
                {
                    "dataset": "unsw_nb15",
                    "model": "logistic_regression",
                    "ablation": "risk_aware_full_b0.01_a0.001",
                    "covered": 0,
                    "sample_count": 5,
                    "mean_true_label_probability": 0.2,
                    "mean_max_probability": 0.8,
                    "mean_probability_margin": 0.6,
                    "unique_max_probability_6dp": 3,
                },
                {
                    "dataset": "unsw_nb15",
                    "model": "random_forest",
                    "ablation": "risk_aware_full_b0.01_a0.001",
                    "covered": 0,
                    "sample_count": 7,
                    "mean_true_label_probability": 0.1,
                    "mean_max_probability": 0.9,
                    "mean_probability_margin": 0.8,
                    "unique_max_probability_6dp": 2,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            risk_path = root / "risk.csv"
            raw_path = root / "raw.csv"
            diagnostics_path = root / "diagnostics.csv"
            output_dir = root / "tables"
            _summary_rows().to_csv(risk_path, index=False)
            _summary_rows().assign(dataset="cic_iot_diad_2024_raw_proportion").to_csv(raw_path, index=False)
            diagnostics.to_csv(diagnostics_path, index=False)

            written = write_manuscript_tables(risk_path, raw_path, diagnostics_path, output_dir)

        self.assertEqual(
            [path.name for path in written],
            ["deployment_cost_metrics.tex", "unsw_failure_diagnostics_metrics.tex", "calibration_baseline_metrics.tex"],
        )


if __name__ == "__main__":
    unittest.main()
