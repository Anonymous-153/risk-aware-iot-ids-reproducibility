from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _datasets_from_config(config_path: str | Path | None) -> list[str] | None:
    if config_path is None:
        return None
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return [dataset["name"] for dataset in config["datasets"]]


def summarize(results_dir: Path, include_datasets: list[str] | None = None) -> pd.DataFrame:
    metric_files = sorted(results_dir.glob("*/metrics_by_run.csv"))
    if include_datasets is not None:
        allowed = set(include_datasets)
        metric_files = [path for path in metric_files if path.parent.name in allowed]
    if not metric_files:
        raise FileNotFoundError(f"No metrics_by_run.csv files found under {results_dir}")
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    numeric_columns = metrics.select_dtypes(include="number").columns.difference(["seed"])
    summary = metrics.groupby(["dataset", "model", "ablation"])[list(numeric_columns)].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-run metrics into a manuscript-ready summary.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/combined_summary.csv")
    parser.add_argument("--config", help="Experiment config whose dataset list should be included in the summary.")
    parser.add_argument("--include-datasets", nargs="+", help="Dataset names to include in the summary.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    include_datasets = args.include_datasets or _datasets_from_config(args.config)
    summarize(Path(args.results_dir), include_datasets=include_datasets).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
