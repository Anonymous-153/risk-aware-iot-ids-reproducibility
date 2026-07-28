from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.audit import audit_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit IDS experiment artifacts before manuscript use.")
    parser.add_argument("output_dir", help="Directory containing experiment artifacts.")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-conformal-coverage", type=float, default=0.8)
    parser.add_argument(
        "--allow-missing-raw-predictions",
        action="store_true",
        help="Audit included metrics and summaries when large raw_predictions.csv files are intentionally omitted.",
    )
    args = parser.parse_args()

    report = audit_results(
        args.output_dir,
        min_seeds=args.min_seeds,
        min_conformal_coverage=args.min_conformal_coverage,
        require_raw_predictions=not args.allow_missing_raw_predictions,
    )
    print("\n".join(report.to_lines()))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
