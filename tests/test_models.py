import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.models import make_classifier


class ModelFactoryTests(unittest.TestCase):
    def test_logistic_regression_uses_high_iteration_stable_solver(self) -> None:
        classifier = make_classifier("logistic_regression", seed=7)

        params = classifier.get_params()
        self.assertEqual(params["solver"], "lbfgs")
        self.assertGreaterEqual(params["max_iter"], 5000)

    def test_xgboost_device_is_configurable(self) -> None:
        classifier = make_classifier("xgboost", seed=7, xgboost_device="cuda")

        self.assertEqual(classifier.get_params()["device"], "cuda")

    def test_mlp_is_small_enough_for_full_matrix(self) -> None:
        classifier = make_classifier("mlp", seed=7)

        params = classifier.get_params()
        self.assertEqual(params["hidden_layer_sizes"], (32,))
        self.assertLessEqual(params["max_iter"], 80)
        self.assertGreaterEqual(params["batch_size"], 1024)


if __name__ == "__main__":
    unittest.main()
