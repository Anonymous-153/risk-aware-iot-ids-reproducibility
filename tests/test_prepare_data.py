import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iot_ids.data import (
    DatasetSpec,
    read_csv_many,
    sample_csv_many,
    sample_csv_many_proportional,
    split_frame,
    split_train_calibration,
    write_splits,
)
from prepare_data import _expand_paths


class PrepareDataTests(unittest.TestCase):
    def test_split_frame_writes_train_calibration_test_without_overlap(self) -> None:
        frame = pd.DataFrame(
            {
                "feature": list(range(12)),
                "label": ["Benign", "Attack"] * 6,
            }
        )
        spec = DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"])

        splits = split_frame(frame, spec, seed=7, test_size=0.25, calibration_size=0.25)

        self.assertEqual(set(splits.keys()), {"train", "calibration", "test"})
        self.assertEqual(sum(len(split) for split in splits.values()), 12)
        feature_sets = [set(split["feature"].tolist()) for split in splits.values()]
        self.assertTrue(feature_sets[0].isdisjoint(feature_sets[1]))
        self.assertTrue(feature_sets[0].isdisjoint(feature_sets[2]))
        self.assertTrue(feature_sets[1].isdisjoint(feature_sets[2]))

    def test_split_frame_deduplicates_before_random_split(self) -> None:
        frame = pd.DataFrame(
            {
                "feature": [1, 1, 2, 2, 3, 4, 5, 6],
                "source_file": ["a.csv", "b.csv", "a.csv", "b.csv", "c.csv", "c.csv", "d.csv", "d.csv"],
                "label": ["Benign", "Benign", "Attack", "Attack", "Benign", "Attack", "Benign", "Attack"],
            }
        )
        spec = DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"])

        splits = split_frame(frame, spec, seed=7, test_size=0.25, calibration_size=0.25)

        self.assertEqual(sum(len(split) for split in splits.values()), 6)
        self.assertEqual(splits["train"].attrs["deduplicated_rows"], 2)

    def test_split_train_calibration_preserves_external_test_set(self) -> None:
        train = pd.DataFrame(
            {
                "feature": list(range(10)),
                "label": ["Benign", "Attack"] * 5,
            }
        )
        test = pd.DataFrame(
            {
                "feature": [100, 101, 102, 103],
                "label": ["Benign", "Attack", "Benign", "Attack"],
            }
        )
        spec = DatasetSpec(name="toy", label_column="label", positive_labels=["Attack"])

        splits = split_train_calibration(train, test, spec, seed=11, calibration_size=0.3)

        self.assertEqual(splits["test"]["feature"].tolist(), [100, 101, 102, 103])
        self.assertEqual(sum(len(split) for split in splits.values()), 14)

    def test_write_splits_drops_forbidden_columns_before_saving(self) -> None:
        splits = {
            "train": pd.DataFrame({"feature": [1], "source_file": ["a.csv"], "attack_cat": ["Generic"], "target": [1]}),
            "calibration": pd.DataFrame({"feature": [2], "source_file": ["b.csv"], "attack_cat": ["Normal"], "target": [0]}),
            "test": pd.DataFrame({"feature": [3], "source_file": ["c.csv"], "attack_cat": ["Normal"], "target": [0]}),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_splits(splits, output_dir, "toy")
            saved_train = pd.read_csv(output_dir / "toy" / "train.csv")
            leakage_report = pd.read_csv(output_dir / "toy" / "leakage_report.csv")

        self.assertNotIn("attack_cat", saved_train.columns)
        self.assertNotIn("source_file", saved_train.columns)
        self.assertTrue(bool(leakage_report["passed"].iloc[0]))
        self.assertEqual(leakage_report["dropped_forbidden_columns"].iloc[0], "attack_cat, source_file")
        self.assertEqual(int(leakage_report["deduplicated_rows"].iloc[0]), 0)

    def test_expand_paths_supports_recursive_globs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested" / "deeper"
            nested.mkdir(parents=True)
            first = root / "top.csv"
            second = nested / "inner.csv"
            first.write_text("feature,label\n1,Benign\n", encoding="utf-8")
            second.write_text("feature,label\n2,Attack\n", encoding="utf-8")

            paths = _expand_paths([str(root / "**" / "*.csv")])

        self.assertEqual([path.name for path in paths], ["inner.csv", "top.csv"])

    def test_read_csv_many_uses_reference_header_for_headerless_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            headered = root / "Benign" / "benign.csv"
            headerless = root / "DoS" / "tcp.csv"
            headered.parent.mkdir(parents=True)
            headerless.parent.mkdir(parents=True)
            headered.write_text("feature,Label\n1,Benign\n", encoding="utf-8")
            headerless.write_text("2,NeedManualLabel\n", encoding="utf-8")
            spec = DatasetSpec(
                name="toy",
                label_column="Label",
                positive_labels=["Attack"],
            )

            frame = read_csv_many([headered, headerless], spec)

        self.assertEqual(frame["feature"].tolist(), [1, 2])
        self.assertEqual(frame["Label"].tolist(), ["Benign", "NeedManualLabel"])

    def test_read_csv_many_can_derive_binary_label_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benign = root / "Benign" / "benign.csv"
            attack = root / "DoS" / "attack.csv"
            benign.parent.mkdir(parents=True)
            attack.parent.mkdir(parents=True)
            benign.write_text("feature,Label\n1,NeedManualLabel\n", encoding="utf-8")
            attack.write_text("feature,Label\n2,NeedManualLabel\n", encoding="utf-8")
            spec = DatasetSpec(
                name="toy",
                label_column="Label",
                positive_labels=["Attack"],
                derive_label_from_path=True,
                benign_path_markers=("Benign",),
            )

            frame = read_csv_many([benign, attack], spec)

        self.assertEqual(frame["Label"].tolist(), ["Benign", "Attack"])

    def test_sample_csv_many_caps_each_binary_label_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benign = root / "Benign" / "benign.csv"
            attack = root / "Mirai" / "attack.csv"
            manifest = root / "sampling_manifest.csv"
            benign.parent.mkdir(parents=True)
            attack.parent.mkdir(parents=True)
            benign.write_text(
                "feature,Label\n" + "\n".join(f"{index},NeedManualLabel" for index in range(10)) + "\n",
                encoding="utf-8",
            )
            attack.write_text(
                "feature,Label\n" + "\n".join(f"{100 + index},NeedManualLabel" for index in range(10)) + "\n",
                encoding="utf-8",
            )
            spec = DatasetSpec(
                name="toy",
                label_column="Label",
                positive_labels=["Attack"],
                derive_label_from_path=True,
                benign_path_markers=("Benign",),
            )

            frame = sample_csv_many(
                [benign, attack],
                spec,
                max_rows_per_label=3,
                seed=7,
                chunksize=4,
                manifest_path=manifest,
            )
            manifest_frame = pd.read_csv(manifest)

        self.assertEqual(frame["Label"].value_counts().to_dict(), {"Benign": 3, "Attack": 3})
        self.assertEqual(set(manifest_frame["target"].tolist()), {0, 1})
        self.assertEqual(manifest_frame["rows_seen"].sum(), 20)
        self.assertEqual(manifest_frame["rows_selected"].sum(), 6)

    def test_sample_csv_many_uses_reference_header_for_headerless_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            headered = root / "Benign" / "benign.csv"
            headerless = root / "DoS" / "tcp.csv"
            headered.parent.mkdir(parents=True)
            headerless.parent.mkdir(parents=True)
            headered.write_text("feature,Label\n1,NeedManualLabel\n2,NeedManualLabel\n", encoding="utf-8")
            headerless.write_text("101,NeedManualLabel\n102,NeedManualLabel\n", encoding="utf-8")
            spec = DatasetSpec(
                name="toy",
                label_column="Label",
                positive_labels=["Attack"],
                derive_label_from_path=True,
                benign_path_markers=("Benign",),
            )

            frame = sample_csv_many([headered, headerless], spec, max_rows_per_label=2, seed=7, chunksize=1)

        self.assertEqual(sorted(frame["feature"].tolist()), [1, 2, 101, 102])
        self.assertEqual(frame["Label"].value_counts().to_dict(), {"Benign": 2, "Attack": 2})

    def test_sample_csv_many_proportional_preserves_attack_heavy_ratio_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benign = root / "Benign" / "benign.csv"
            attack = root / "Mirai" / "attack.csv"
            manifest = root / "proportional_manifest.csv"
            benign.parent.mkdir(parents=True)
            attack.parent.mkdir(parents=True)
            benign.write_text(
                "feature,Label\n" + "\n".join(f"{index},NeedManualLabel" for index in range(10)) + "\n",
                encoding="utf-8",
            )
            attack.write_text(
                "feature,Label\n" + "\n".join(f"{100 + index},NeedManualLabel" for index in range(90)) + "\n",
                encoding="utf-8",
            )
            spec = DatasetSpec(
                name="toy",
                label_column="Label",
                positive_labels=["Attack"],
                derive_label_from_path=True,
                benign_path_markers=("Benign",),
            )

            frame = sample_csv_many_proportional(
                [benign, attack],
                spec,
                max_total_rows=20,
                min_rows_per_label=2,
                seed=7,
                chunksize=11,
                manifest_path=manifest,
            )
            manifest_frame = pd.read_csv(manifest)

        self.assertEqual(frame["target"].value_counts().to_dict(), {1: 18, 0: 2})
        self.assertEqual(manifest_frame["rows_seen"].sum(), 100)
        self.assertEqual(manifest_frame["rows_selected"].sum(), 20)


if __name__ == "__main__":
    unittest.main()
