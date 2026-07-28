import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.data import DatasetSpec, split_frame
from iot_ids.pipeline import ExperimentRequest, run_experiment


def _toy_frame(samples_per_class: int = 12) -> pd.DataFrame:
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


class PipelineLatencyTests(unittest.TestCase):
    def test_run_experiment_records_model_only_and_end_to_end_latency(self) -> None:
        frame = _toy_frame()
        splits = split_frame(
            frame,
            DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"]),
            seed=23,
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
                seeds=[23],
                latency_repeats=2,
            )
            run_experiment(request)

            metrics = pd.read_csv(Path(tmpdir) / "metrics_by_run.csv")

        self.assertIn("model_only_latency_ms_per_sample", metrics.columns)
        self.assertIn("end_to_end_latency_ms_per_sample", metrics.columns)
        self.assertIn("latency_repeats", metrics.columns)
        self.assertEqual(int(metrics["latency_repeats"].iloc[0]), 2)
        self.assertGreater(float(metrics["model_only_latency_ms_per_sample"].iloc[0]), 0.0)
        self.assertGreater(float(metrics["end_to_end_latency_ms_per_sample"].iloc[0]), 0.0)
        self.assertGreaterEqual(
            float(metrics["end_to_end_latency_ms_per_sample"].iloc[0]),
            float(metrics["model_only_latency_ms_per_sample"].iloc[0]),
        )

    def test_platt_full_ablation_writes_standard_metrics(self) -> None:
        frame = _toy_frame()
        splits = split_frame(
            frame,
            DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"]),
            seed=29,
            test_size=0.25,
            calibration_size=0.4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            request = ExperimentRequest(
                dataset_name="toy",
                splits=splits,
                output_dir=Path(tmpdir),
                model_names=["centroid"],
                ablations=["platt_full", "risk_aware_platt_full_b0.05_a0.005"],
                seeds=[29],
                latency_repeats=1,
            )
            run_experiment(request)

            metrics = pd.read_csv(Path(tmpdir) / "metrics_by_run.csv")

        self.assertEqual(metrics["ablation"].tolist(), ["platt_full", "risk_aware_platt_full_b0.05_a0.005"])
        self.assertIn("attack_coverage", metrics.columns)
        self.assertIn("end_to_end_latency_ms_per_sample", metrics.columns)


if __name__ == "__main__":
    unittest.main()
