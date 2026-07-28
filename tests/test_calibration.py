import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.calibration import PlattProbabilityCalibrator


class PlattProbabilityCalibratorTests(unittest.TestCase):
    def test_binary_platt_calibrator_preserves_probability_shape_and_order(self) -> None:
        probabilities = np.array(
            [
                [0.95, 0.05],
                [0.80, 0.20],
                [0.30, 0.70],
                [0.05, 0.95],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        classes = np.array([0, 1])

        calibrator = PlattProbabilityCalibrator().fit(probabilities, labels, classes)
        calibrated = calibrator.transform(np.array([[0.90, 0.10], [0.10, 0.90]]))

        self.assertEqual(calibrated.shape, (2, 2))
        self.assertTrue(np.all(calibrated >= 0.0))
        self.assertTrue(np.all(calibrated <= 1.0))
        np.testing.assert_allclose(calibrated.sum(axis=1), np.ones(2))
        self.assertGreater(calibrated[1, 1], calibrated[0, 1])


if __name__ == "__main__":
    unittest.main()
