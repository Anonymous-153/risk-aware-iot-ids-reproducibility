from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.data import DatasetSpec, infer_reference_columns, read_csv_with_optional_header


KNOWN_SHA256: dict[str, str] = {
    # Hugging Face mirror metadata for Mouwiya/UNSW-NB15-small, used only as
    # a provisional checksum aid when the official UNSW download is unavailable.
    "UNSW_NB15_training-set.csv": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    "UNSW_NB15_testing-set.csv": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
}


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
        paths.extend(sorted(matches) or [Path(pattern)])
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_csv(path: Path, spec: DatasetSpec, reference_columns: list[str] | None) -> dict[str, object]:
    raw_columns = list(pd.read_csv(path, nrows=0).columns)
    used_reference_header = spec.label_column not in raw_columns and reference_columns is not None
    frame = read_csv_with_optional_header(path, spec=spec, reference_columns=reference_columns, nrows=5000)
    if spec.label_column not in frame.columns:
        raise KeyError(spec.label_column)
    labels = frame[spec.label_column].astype(str).value_counts(dropna=False).head(20).to_dict()
    return {
        "rows_checked": len(frame),
        "columns": len(frame.columns),
        "label_column_found": spec.label_column in frame.columns,
        "used_reference_header": used_reference_header,
        "derived_label_from_path": spec.derive_label_from_path,
        "label_preview": labels,
    }


def verify_config(config_path: Path) -> tuple[list[str], list[dict[str, object]]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    records: list[dict[str, object]] = []

    for dataset in config["datasets"]:
        spec = DatasetSpec(
            name=dataset["name"],
            label_column=dataset["label_column"],
            positive_labels=dataset["positive_labels"],
            derive_label_from_path=bool(dataset.get("derive_label_from_path", False)),
            benign_path_markers=tuple(dataset.get("benign_path_markers", ["Benign"])),
        )
        path_patterns: list[str] = []
        if "official_split" in dataset:
            path_patterns.extend(dataset["official_split"]["train"])
            path_patterns.extend(dataset["official_split"]["test"])
        else:
            path_patterns.extend(dataset["raw_paths"])

        expanded_paths = _expand(path_patterns)
        reference_columns = infer_reference_columns(expanded_paths, spec.label_column)
        for path in expanded_paths:
            record: dict[str, object] = {
                "dataset": spec.name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            if not path.exists():
                failures.append(f"missing raw data file: {path}")
            elif path.stat().st_size == 0:
                failures.append(f"empty raw data file: {path}")
            else:
                digest = _sha256(path)
                record["sha256"] = digest
                expected = KNOWN_SHA256.get(path.name)
                if expected:
                    record["known_sha256_match"] = digest == expected
                try:
                    record.update(_inspect_csv(path, spec, reference_columns))
                    if not record["label_column_found"]:
                        failures.append(f"{path} is missing label column {spec.label_column!r}")
                except Exception as exc:  # pragma: no cover - exact parser errors vary by pandas version.
                    failures.append(f"could not read {path} as CSV: {exc}")
            records.append(record)
    return failures, records


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify raw IDS datasets before running experiments.")
    parser.add_argument("--config", required=True, help="Experiment JSON config.")
    parser.add_argument(
        "--manifest",
        default="results/raw_data_manifest.csv",
        help="Path where the raw-data manifest CSV should be written.",
    )
    args = parser.parse_args()

    failures, records = verify_config(Path(args.config))
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(f"wrote raw data manifest: {manifest_path}")

    if failures:
        print("Raw data verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Raw data verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
