import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.conformal import ClassConditionalAdaptiveConformal, ClassConditionalConformal, ClassSpecificAlphaConformal


class ClassConditionalConformalTests(unittest.TestCase):
    def test_thresholds_are_fit_per_true_class(self) -> None:
        probabilities = np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.60, 0.40],
                [0.20, 0.80],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        conformal = ClassConditionalConformal(alpha=0.25).fit(probabilities, labels)

        self.assertAlmostEqual(conformal.thresholds_[0], 0.20)
        self.assertAlmostEqual(conformal.thresholds_[1], 0.60)

    def test_prediction_sets_include_classes_below_threshold_and_never_empty(self) -> None:
        conformal = ClassConditionalConformal(alpha=0.25)
        conformal.classes_ = np.array([0, 1])
        conformal.thresholds_ = {0: 0.20, 1: 0.40}

        probabilities = np.array(
            [
                [0.85, 0.15],
                [0.30, 0.70],
                [0.45, 0.55],
            ]
        )

        prediction_sets = conformal.predict_sets(probabilities)

        self.assertEqual(prediction_sets[0], {0})
        self.assertEqual(prediction_sets[1], {1})
        self.assertEqual(prediction_sets[2], {1})


class ClassSpecificAlphaConformalTests(unittest.TestCase):
    def test_thresholds_use_class_specific_alpha_values(self) -> None:
        probabilities = np.array(
            [
                [0.99, 0.01],
                [0.90, 0.10],
                [0.80, 0.20],
                [0.70, 0.30],
                [0.30, 0.70],
                [0.20, 0.80],
                [0.10, 0.90],
                [0.01, 0.99],
            ]
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        conformal = ClassSpecificAlphaConformal(alpha_by_class={0: 0.5, 1: 0.01}).fit(probabilities, labels)

        self.assertAlmostEqual(conformal.thresholds_[0], 0.20)
        self.assertAlmostEqual(conformal.thresholds_[1], 0.30)

    def test_missing_class_alpha_falls_back_to_default_alpha(self) -> None:
        probabilities = np.array([[0.90, 0.10], [0.80, 0.20], [0.20, 0.80], [0.10, 0.90]])
        labels = np.array([0, 0, 1, 1])

        conformal = ClassSpecificAlphaConformal(alpha=0.25, alpha_by_class={1: 0.01}).fit(probabilities, labels)

        self.assertAlmostEqual(conformal.thresholds_[0], 0.20)
        self.assertAlmostEqual(conformal.thresholds_[1], 0.20)


class ClassConditionalAdaptiveConformalTests(unittest.TestCase):
    def test_thresholds_are_fit_from_cumulative_probability_scores(self) -> None:
        probabilities = np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.60, 0.40],
                [0.20, 0.80],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        conformal = ClassConditionalAdaptiveConformal(alpha=0.25).fit(probabilities, labels)

        self.assertAlmostEqual(conformal.thresholds_[0], 0.90)
        self.assertAlmostEqual(conformal.thresholds_[1], 1.00)

    def test_prediction_sets_use_cumulative_probability_scores(self) -> None:
        conformal = ClassConditionalAdaptiveConformal(alpha=0.25)
        conformal.classes_ = np.array([0, 1])
        conformal.thresholds_ = {0: 0.85, 1: 0.95}

        probabilities = np.array(
            [
                [0.85, 0.15],
                [0.30, 0.70],
                [0.45, 0.55],
            ]
        )

        prediction_sets = conformal.predict_sets(probabilities)

        self.assertEqual(prediction_sets[0], {0})
        self.assertEqual(prediction_sets[1], {1})
        self.assertEqual(prediction_sets[2], {1})


if __name__ == "__main__":
    unittest.main()
