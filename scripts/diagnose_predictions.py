from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.reporting import confidence_diagnostics, coverage_by_class, set_size_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate class-wise coverage diagnostics from raw predictions.")
    parser.add_argument("--raw", required=True, help="Path to raw_predictions.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostic CSV files.")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw, dtype={"prediction_set": str}, low_memory=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_path = output_dir / "coverage_by_class.csv"
    set_size_path = output_dir / "set_size_distribution.csv"
    confidence_path = output_dir / "confidence_diagnostics.csv"
    coverage_by_class(raw).to_csv(coverage_path, index=False)
    set_size_distribution(raw).to_csv(set_size_path, index=False)
    confidence_diagnostics(raw).to_csv(confidence_path, index=False)

    print(f"wrote {coverage_path}")
    print(f"wrote {set_size_path}")
    print(f"wrote {confidence_path}")


if __name__ == "__main__":
    main()
