import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.reporting import (
    collect_environment_metadata,
    confidence_diagnostics,
    coverage_by_class,
    metric_table_to_latex,
    set_size_distribution,
    write_metric_tables,
)


class ReportingTests(unittest.TestCase):
    def test_metric_table_to_latex_formats_mean_and_std(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "dataset": "unsw_nb15",
                    "model": "logistic_regression",
                    "ablation": "full",
                    "macro_f1_mean": 0.81234,
                    "macro_f1_std": 0.01234,
                    "ece_mean": 0.03456,
                    "ece_std": 0.00456,
                }
            ]
        )

        latex = metric_table_to_latex(summary, metrics=["macro_f1", "ece"])

        self.assertIn("Macro-F1", latex)
        self.assertIn("ECE", latex)
        self.assertIn("0.812 $\\pm$ 0.012", latex)
        self.assertIn("0.035 $\\pm$ 0.005", latex)
        self.assertIn("UNSW-NB15", latex)
        self.assertIn("LogReg", latex)

    def test_write_metric_tables_creates_tables_from_summary(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "dataset": "smoke",
                    "model": "centroid",
                    "ablation": "baseline",
                    "macro_f1_mean": 0.5,
                    "macro_f1_std": 0.0,
                    "mcc_mean": 0.0,
                    "mcc_std": 0.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "summary.csv"
            output_dir = root / "tables"
            summary.to_csv(summary_path, index=False)

            written = write_metric_tables(
                summary_path,
                output_dir,
                metrics=["macro_f1", "mcc"],
                caption="Smoke metrics.",
                label="tab:smoke-metrics",
            )

            self.assertEqual(written, [output_dir / "main_metrics.tex"])
            latex = written[0].read_text(encoding="utf-8")
            self.assertIn("centroid", latex)
            self.assertIn(r"\caption{Smoke metrics.}", latex)
            self.assertIn(r"\label{tab:smoke-metrics}", latex)

    def test_collect_environment_metadata_records_command_config_hash_and_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text('{"seed": 42}', encoding="utf-8")

            metadata = collect_environment_metadata(
                config_path=config_path,
                command=["python", "train.py", "--config", str(config_path)],
                package_versions={"numpy": "2.5.1"},
                gpu_info="NVIDIA RTX test",
            )

        self.assertEqual(metadata["command"], ["python", "train.py", "--config", str(config_path)])
        self.assertEqual(metadata["config_sha256"], "1e5792df46fc86638a2abe2c52f3b40568a4f4cae33116b7eb3a69c0048d1afe")
        self.assertEqual(metadata["gpu_info"], "NVIDIA RTX test")
        self.assertEqual(metadata["package_versions"]["numpy"], "2.5.1")
        self.assertIn("python_version", metadata)

    def test_environment_metadata_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            metadata = collect_environment_metadata(
                config_path=config_path,
                command=["python", "prepare_data.py"],
                package_versions={},
                gpu_info=None,
            )

        json.dumps(metadata)

    def test_coverage_by_class_summarizes_raw_predictions(self) -> None:
        raw = pd.DataFrame(
            [
                {"dataset": "toy", "model": "rf", "ablation": "full", "seed": 1, "true_label": 0, "covered": 1},
                {"dataset": "toy", "model": "rf", "ablation": "full", "seed": 1, "true_label": 0, "covered": 0},
                {"dataset": "toy", "model": "rf", "ablation": "full", "seed": 1, "true_label": 1, "covered": 1},
            ]
        )

        summary = coverage_by_class(raw)

        self.assertEqual(summary.loc[summary["true_label"].eq(0), "sample_count"].iloc[0], 2)
        self.assertAlmostEqual(summary.loc[summary["true_label"].eq(0), "coverage"].iloc[0], 0.5)
        self.assertAlmostEqual(summary.loc[summary["true_label"].eq(1), "coverage"].iloc[0], 1.0)

    def test_set_size_distribution_parses_prediction_set_strings(self) -> None:
        raw = pd.DataFrame(
            [
                {"dataset": "toy", "model": "rf", "ablation": "full", "seed": 1, "prediction_set": "0"},
                {"dataset": "toy", "model": "rf", "ablation": "full", "seed": 1, "prediction_set": "0|1"},
                {"dataset": "toy", "model": "rf", "ablation": "full", "seed": 2, "prediction_set": "1"},
            ]
        )

        distribution = set_size_distribution(raw)

        singleton = distribution[distribution["set_size"].eq(1)].iloc[0]
        doubleton = distribution[distribution["set_size"].eq(2)].iloc[0]
        self.assertEqual(singleton["count"], 2)
        self.assertEqual(doubleton["count"], 1)

    def test_confidence_diagnostics_summarizes_covered_and_uncovered_predictions(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "dataset": "toy",
                    "model": "rf",
                    "ablation": "full",
                    "seed": 1,
                    "true_label": 0,
                    "covered": 1,
                    "probability_0": 0.9,
                    "probability_1": 0.1,
                },
                {
                    "dataset": "toy",
                    "model": "rf",
                    "ablation": "full",
                    "seed": 1,
                    "true_label": 1,
                    "covered": 0,
                    "probability_0": 0.8,
                    "probability_1": 0.2,
                },
            ]
        )

        diagnostics = confidence_diagnostics(raw)

        self.assertEqual(set(diagnostics["covered"].tolist()), {0, 1})
        self.assertAlmostEqual(diagnostics.loc[diagnostics["covered"].eq(1), "mean_true_label_probability"].iloc[0], 0.9)
        self.assertAlmostEqual(diagnostics.loc[diagnostics["covered"].eq(0), "mean_true_label_probability"].iloc[0], 0.2)
        self.assertAlmostEqual(diagnostics.loc[diagnostics["covered"].eq(0), "mean_max_probability"].iloc[0], 0.8)
        self.assertAlmostEqual(diagnostics.loc[diagnostics["covered"].eq(0), "mean_probability_margin"].iloc[0], 0.6)
        self.assertEqual(diagnostics.loc[diagnostics["covered"].eq(0), "unique_max_probability_6dp"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
