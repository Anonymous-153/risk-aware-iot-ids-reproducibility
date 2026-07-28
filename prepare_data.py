from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from iot_ids.data import (
    DatasetSpec,
    read_csv_many,
    sample_csv_many,
    sample_csv_many_proportional,
    split_frame,
    split_train_calibration,
    write_splits,
)


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(path) for path in glob.glob(pattern, recursive=True)]
        if matched:
            paths.extend(sorted(matched))
        else:
            paths.append(Path(pattern))
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing raw data files: " + ", ".join(str(path) for path in missing))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/calibration/test CSV splits.")
    parser.add_argument("--config", required=True, help="Experiment JSON config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepared_dir = Path(config.get("prepared_dir", "data/processed"))
    seed = int(config.get("seed", 42))
    test_size = float(config.get("test_size", 0.2))
    calibration_size = float(config.get("calibration_size", 0.2))

    for dataset in config["datasets"]:
        spec = DatasetSpec(
            name=dataset["name"],
            label_column=dataset["label_column"],
            positive_labels=dataset["positive_labels"],
            derive_label_from_path=bool(dataset.get("derive_label_from_path", False)),
            benign_path_markers=tuple(dataset.get("benign_path_markers", ["Benign"])),
        )
        if "official_split" in dataset:
            train_paths = _expand_paths(dataset["official_split"]["train"])
            test_paths = _expand_paths(dataset["official_split"]["test"])
            train = read_csv_many(train_paths, spec)
            test = read_csv_many(test_paths, spec)
            splits = split_train_calibration(train, test, spec, seed=seed, calibration_size=calibration_size)
        else:
            raw_paths = _expand_paths(dataset["raw_paths"])
            sampling = dataset.get("sampling")
            if sampling:
                if sampling.get("strategy", "per_label_cap") == "proportional":
                    frame = sample_csv_many_proportional(
                        raw_paths,
                        spec,
                        max_total_rows=int(sampling["max_total_rows"]),
                        min_rows_per_label=int(sampling.get("min_rows_per_label", 1_000)),
                        seed=seed,
                        chunksize=int(sampling.get("chunksize", 100_000)),
                        manifest_path=sampling.get("manifest"),
                    )
                else:
                    frame = sample_csv_many(
                        raw_paths,
                        spec,
                        max_rows_per_label=int(sampling["max_rows_per_label"]),
                        seed=seed,
                        chunksize=int(sampling.get("chunksize", 100_000)),
                        manifest_path=sampling.get("manifest"),
                    )
            else:
                frame = read_csv_many(raw_paths, spec)
            splits = split_frame(frame, spec, seed=seed, test_size=test_size, calibration_size=calibration_size)
        write_splits(splits, prepared_dir, spec.name)
        print(f"prepared {spec.name} -> {prepared_dir / spec.name}")


if __name__ == "__main__":
    main()
