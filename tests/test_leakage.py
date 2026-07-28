import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.leakage import LeakageReport, check_forbidden_columns, count_cross_split_duplicates


class LeakageTests(unittest.TestCase):
    def test_forbidden_columns_are_detected_case_insensitively(self) -> None:
        frame = pd.DataFrame(
            {
                "Flow_ID": ["a"],
                "src_ip": ["10.0.0.1"],
                "Timestamp": ["2026-07-23 12:00:00"],
                "duration": [1.0],
                "Label": ["Benign"],
            }
        )

        forbidden = check_forbidden_columns(frame)

        self.assertEqual(forbidden, ["Flow_ID", "src_ip", "Timestamp"])

    def test_duplicate_count_ignores_label_columns(self) -> None:
        train = pd.DataFrame(
            {
                "duration": [1.0, 2.0],
                "packets": [10, 20],
                "label": ["Benign", "Attack"],
            }
        )
        test = pd.DataFrame(
            {
                "duration": [2.0, 3.0],
                "packets": [20, 30],
                "label": ["Different", "Attack"],
            }
        )

        count = count_cross_split_duplicates(train, test, label_columns=["label"])

        self.assertEqual(count, 1)

    def test_leakage_report_status_fails_when_duplicates_exist(self) -> None:
        report = LeakageReport(forbidden_columns=[], train_test_duplicate_rows=2)

        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
