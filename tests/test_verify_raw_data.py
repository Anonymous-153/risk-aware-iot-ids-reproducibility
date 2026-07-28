import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_raw_data import verify_config


class VerifyRawDataTests(unittest.TestCase):
    def test_verify_config_records_readable_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "sample.csv"
            pd.DataFrame(
                {
                    "feature": [1.0, 2.0],
                    "label": ["Benign", "Attack"],
                }
            ).to_csv(raw_path, index=False)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "toy",
                                "label_column": "label",
                                "positive_labels": ["Attack"],
                                "raw_paths": [str(raw_path)],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            failures, records = verify_config(config_path)

        self.assertEqual(failures, [])
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["label_column_found"])
        self.assertEqual(records[0]["rows_checked"], 2)
        self.assertEqual(len(records[0]["sha256"]), 64)

    def test_verify_config_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_path = root / "missing.csv"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "toy",
                                "label_column": "label",
                                "positive_labels": ["Attack"],
                                "raw_paths": [str(missing_path)],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            failures, records = verify_config(config_path)

        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["exists"])
        self.assertIn("missing raw data file", failures[0])

    def test_verify_config_expands_recursive_globs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "flow" / "Benign"
            nested.mkdir(parents=True)
            raw_path = nested / "sample.csv"
            pd.DataFrame({"feature": [1.0], "Label": ["Benign"]}).to_csv(raw_path, index=False)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "toy",
                                "label_column": "Label",
                                "positive_labels": ["Attack"],
                                "raw_paths": [str(root / "**" / "*.csv")],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            failures, records = verify_config(config_path)

        self.assertEqual(failures, [])
        self.assertEqual([Path(record["path"]).name for record in records], ["sample.csv"])


if __name__ == "__main__":
    unittest.main()
