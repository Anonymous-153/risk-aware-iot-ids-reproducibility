import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.data import DatasetSpec, split_frame
from iot_ids import pipeline
from iot_ids.pipeline import ExperimentRequest, run_experiment


def _toy_frame(samples_per_class: int = 8) -> pd.DataFrame:
    benign_duration = [0.1 + 0.01 * index for index in range(samples_per_class)]
    attack_duration = [3.1 + 0.01 * index for index in range(samples_per_class)]
    return pd.DataFrame(
        {
            "duration": benign_duration + attack_duration,
            "packets": [5 + index % 4 for index in range(samples_per_class)]
            + [30 + index % 4 for index in range(samples_per_class)],
            "proto": ["tcp" if index % 2 == 0 else "udp" for index in range(samples_per_class)] * 2,
            "label": ["Benign"] * samples_per_class + ["Attack"] * samples_per_class,
        }
    )


class PipelineSmokeTests(unittest.TestCase):
    def test_run_experiment_writes_required_artifacts(self) -> None:
        frame = _toy_frame()
        splits = split_frame(
            frame,
            DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"]),
            seed=3,
            test_size=0.25,
            calibration_size=0.25,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            request = ExperimentRequest(
                dataset_name="toy",
                splits=splits,
                output_dir=Path(tmpdir),
                model_names=["centroid"],
                ablations=["baseline", "full"],
                seeds=[3],
            )
            run_experiment(request)

            raw_predictions = pd.read_csv(Path(tmpdir) / "raw_predictions.csv")
            metrics = pd.read_csv(Path(tmpdir) / "metrics_by_run.csv")
            summary = pd.read_csv(Path(tmpdir) / "summary.csv")

        self.assertIn("conformal_coverage", metrics.columns)
        self.assertIn("average_set_size", metrics.columns)
        self.assertIn("latency_ms_per_sample", metrics.columns)
        self.assertIn("prediction_set", raw_predictions.columns)
        self.assertNotIn("index", summary.columns)
        self.assertEqual(set(summary["dataset"]), {"toy"})

    def test_run_experiment_flushes_completed_runs_before_later_failure(self) -> None:
        frame = _toy_frame()
        splits = split_frame(
            frame,
            DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"]),
            seed=3,
            test_size=0.25,
            calibration_size=0.25,
        )

        original_make_classifier = pipeline.make_classifier

        def make_classifier_or_fail(model_name: str, seed: int, xgboost_device: str | None = None):
            if model_name == "boom":
                raise RuntimeError("intentional later failure")
            return original_make_classifier(model_name, seed, xgboost_device=xgboost_device)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            request = ExperimentRequest(
                dataset_name="toy",
                splits=splits,
                output_dir=output_dir,
                model_names=["centroid", "boom"],
                ablations=["baseline"],
                seeds=[3],
            )
            with patch("iot_ids.pipeline.make_classifier", side_effect=make_classifier_or_fail):
                with self.assertRaises(RuntimeError):
                    run_experiment(request)

            metrics = pd.read_csv(output_dir / "metrics_by_run.csv")
            raw_predictions = pd.read_csv(output_dir / "raw_predictions.csv")

        self.assertEqual(metrics["model"].tolist(), ["centroid"])
        self.assertFalse(raw_predictions.empty)

    def test_full_pipeline_splits_probability_and_conformal_calibration_roles(self) -> None:
        frame = _toy_frame()
        splits = split_frame(
            frame,
            DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"]),
            seed=5,
            test_size=0.25,
            calibration_size=0.4,
        )
        seen: dict[str, int] = {}

        class RecordingCalibrator:
            def fit(self, probabilities, labels, classes):
                del probabilities, classes
                seen["probability_calibration_size"] = len(labels)
                return self

            def transform(self, probabilities):
                return probabilities

        class RecordingConformal:
            def __init__(self, alpha: float = 0.1):
                del alpha

            def fit(self, probabilities, labels, classes):
                del probabilities, classes
                seen["conformal_calibration_size"] = len(labels)
                return self

            def predict_sets(self, probabilities):
                return [{0, 1} for _ in range(len(probabilities))]

        with tempfile.TemporaryDirectory() as tmpdir:
            request = ExperimentRequest(
                dataset_name="toy",
                splits=splits,
                output_dir=Path(tmpdir),
                model_names=["centroid"],
                ablations=["full"],
                seeds=[5],
            )
            with (
                patch("iot_ids.pipeline.BinnedProbabilityCalibrator", RecordingCalibrator),
                patch("iot_ids.pipeline.ClassConditionalConformal", RecordingConformal),
            ):
                run_experiment(request)

        self.assertEqual(
            seen["probability_calibration_size"] + seen["conformal_calibration_size"],
            len(splits["calibration"]),
        )
        self.assertLess(seen["probability_calibration_size"], len(splits["calibration"]))
        self.assertLess(seen["conformal_calibration_size"], len(splits["calibration"]))

    def test_coverage_aware_full_ablation_writes_experiment_outputs(self) -> None:
        frame = _toy_frame()
        splits = split_frame(
            frame,
            DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"]),
            seed=7,
            test_size=0.25,
            calibration_size=0.4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            request = ExperimentRequest(
                dataset_name="toy",
                splits=splits,
                output_dir=Path(tmpdir),
                model_names=["centroid"],
                ablations=["coverage_aware_full"],
                seeds=[7],
            )
            run_experiment(request)

            metrics = pd.read_csv(Path(tmpdir) / "metrics_by_run.csv")
            raw_predictions = pd.read_csv(Path(tmpdir) / "raw_predictions.csv")

        self.assertEqual(metrics["ablation"].tolist(), ["coverage_aware_full"])
        self.assertFalse(raw_predictions.empty)

    def test_threat_aware_full_ablation_includes_positive_label_in_each_prediction_set(self) -> None:
        frame = _toy_frame()
        splits = split_frame(
            frame,
            DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"]),
            seed=13,
            test_size=0.25,
            calibration_size=0.4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            request = ExperimentRequest(
                dataset_name="toy",
                splits=splits,
                output_dir=Path(tmpdir),
                model_names=["centroid"],
                ablations=["threat_aware_full"],
                seeds=[13],
            )
            run_experiment(request)

            raw_predictions = pd.read_csv(Path(tmpdir) / "raw_predictions.csv", dtype={"prediction_set": str})

        self.assertTrue(raw_predictions["prediction_set"].str.contains("1").all())

    def test_risk_aware_full_ablation_writes_classwise_coverage_metrics(self) -> None:
        frame = _toy_frame(samples_per_class=12)
        splits = split_frame(
            frame,
            DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"]),
            seed=17,
            test_size=0.25,
            calibration_size=0.4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            request = ExperimentRequest(
                dataset_name="toy",
                splits=splits,
                output_dir=Path(tmpdir),
                model_names=["centroid"],
                ablations=["risk_aware_full_b0.05_a0.005"],
                seeds=[17],
            )
            run_experiment(request)

            metrics = pd.read_csv(Path(tmpdir) / "metrics_by_run.csv")
            raw_predictions = pd.read_csv(Path(tmpdir) / "raw_predictions.csv")

        self.assertEqual(metrics["ablation"].tolist(), ["risk_aware_full_b0.05_a0.005"])
        self.assertIn("benign_coverage", metrics.columns)
        self.assertIn("attack_coverage", metrics.columns)
        self.assertIn("prediction_set_size", raw_predictions.columns)
        self.assertTrue((raw_predictions["prediction_set_size"] >= 1).all())


if __name__ == "__main__":
    unittest.main()
