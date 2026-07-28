import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import summarize


def _metric_row(dataset: str) -> dict[str, float | int | str]:
    return {
        "dataset": dataset,
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


class EvaluateTests(unittest.TestCase):
    def test_summarize_filters_to_requested_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for dataset in ["smoke", "unsw_nb15"]:
                dataset_dir = root / dataset
                dataset_dir.mkdir()
                pd.DataFrame([_metric_row(dataset)]).to_csv(dataset_dir / "metrics_by_run.csv", index=False)

            summary = summarize(root, include_datasets=["unsw_nb15"])

        self.assertEqual(summary["dataset"].tolist(), ["unsw_nb15"])


if __name__ == "__main__":
    unittest.main()
