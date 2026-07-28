from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from iot_ids.pipeline import ExperimentRequest, run_experiment


def _read_splits(prepared_dir: Path, dataset_name: str) -> dict[str, pd.DataFrame]:
    dataset_dir = prepared_dir / dataset_name
    return {
        "train": pd.read_csv(dataset_dir / "train.csv"),
        "calibration": pd.read_csv(dataset_dir / "calibration.csv"),
        "test": pd.read_csv(dataset_dir / "test.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDS model and conformal ablation experiments.")
    parser.add_argument("--config", required=True, help="Experiment JSON config.")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    prepared_dir = Path(config.get("prepared_dir", "data/processed"))
    results_dir = Path(config.get("results_dir", "results"))
    model_names = config.get("models", ["logistic_regression", "random_forest", "xgboost", "mlp"])
    ablations = config.get("ablations", ["baseline", "cost_sensitive", "calibration", "full"])
    seeds = [int(seed) for seed in config.get("seeds", [11, 23, 37, 53, 71])]
    alpha = float(config.get("alpha", 0.1))
    xgboost_device = config.get("xgboost_device")
    latency_repeats = int(config.get("latency_repeats", 3))

    for dataset in config["datasets"]:
        dataset_name = dataset["name"]
        output_dir = results_dir / dataset_name
        request = ExperimentRequest(
            dataset_name=dataset_name,
            splits=_read_splits(prepared_dir, dataset_name),
            output_dir=output_dir,
            model_names=model_names,
            ablations=ablations,
            seeds=seeds,
            alpha=alpha,
            xgboost_device=xgboost_device,
            latency_repeats=latency_repeats,
        )
        run_experiment(request)
        print(f"trained {dataset_name} -> {output_dir}")


if __name__ == "__main__":
    main()
