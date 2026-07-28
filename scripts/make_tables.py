from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.reporting import write_metric_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript LaTeX tables from summary CSV files.")
    parser.add_argument("--summary", default="results/combined_summary.csv")
    parser.add_argument("--output-dir", default="paper/tables")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["macro_f1", "mcc", "balanced_accuracy", "ece", "conformal_coverage"],
    )
    parser.add_argument("--caption", default="Machine-generated aggregate experiment metrics.")
    parser.add_argument("--label", default="tab:main-metrics")
    args = parser.parse_args()

    for path in write_metric_tables(
        args.summary,
        args.output_dir,
        metrics=args.metrics,
        caption=args.caption,
        label=args.label,
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
