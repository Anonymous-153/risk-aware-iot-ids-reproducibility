import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.metrics import brier_score_multiclass, expected_calibration_error


class MetricsTests(unittest.TestCase):
    def test_brier_score_multiclass_uses_one_hot_targets(self) -> None:
        probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])
        labels = np.array([0, 1])

        score = brier_score_multiclass(probabilities, labels, classes=np.array([0, 1]))

        self.assertAlmostEqual(score, 0.20)

    def test_expected_calibration_error_bins_confidences(self) -> None:
        probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]])
        labels = np.array([0, 1, 1, 0])

        ece = expected_calibration_error(probabilities, labels, n_bins=2)

        self.assertAlmostEqual(ece, 0.25)


if __name__ == "__main__":
    unittest.main()
