import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.preprocessing import TabularPreprocessor


class PreprocessingTests(unittest.TestCase):
    def test_preprocessor_drops_labels_and_forbidden_columns(self) -> None:
        train = pd.DataFrame(
            {
                "duration": [1.0, 2.0],
                "proto": ["tcp", "udp"],
                "src_ip": ["10.0.0.1", "10.0.0.2"],
                "label": ["Benign", "Attack"],
                "target": [0, 1],
            }
        )

        preprocessor = TabularPreprocessor(label_columns=["label", "target"]).fit(train)

        self.assertEqual(preprocessor.feature_columns_, ["duration", "proto"])

    def test_preprocessor_maps_unknown_categories_without_refitting(self) -> None:
        train = pd.DataFrame({"proto": ["tcp", "udp"], "target": [0, 1]})
        test = pd.DataFrame({"proto": ["icmp"], "target": [1]})

        preprocessor = TabularPreprocessor(label_columns=["target"]).fit(train)
        transformed = preprocessor.transform(test)

        self.assertEqual(transformed.tolist(), [[-1.0]])

    def test_numeric_columns_are_standardized_with_training_statistics(self) -> None:
        train = pd.DataFrame({"duration": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        test = pd.DataFrame({"duration": [4.0], "target": [1]})

        preprocessor = TabularPreprocessor(label_columns=["target"]).fit(train)
        transformed_train = preprocessor.transform(train)
        transformed_test = preprocessor.transform(test)

        self.assertAlmostEqual(float(transformed_train.mean()), 0.0)
        self.assertAlmostEqual(float(transformed_train.std(ddof=0)), 1.0)
        self.assertAlmostEqual(float(transformed_test[0, 0]), 2.449489742783178)

    def test_numeric_columns_replace_nonfinite_values_with_training_median(self) -> None:
        train = pd.DataFrame({"rate": [1.0, float("inf"), 3.0, None], "target": [0, 1, 0, 1]})
        test = pd.DataFrame({"rate": [float("-inf"), 5.0], "target": [1, 0]})

        preprocessor = TabularPreprocessor(label_columns=["target"]).fit(train)
        transformed_train = preprocessor.transform(train)
        transformed_test = preprocessor.transform(test)

        self.assertTrue(np.isfinite(transformed_train).all())
        self.assertTrue(np.isfinite(transformed_test).all())
        self.assertAlmostEqual(preprocessor.medians_["rate"], 2.0)
        self.assertAlmostEqual(float(transformed_test[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
