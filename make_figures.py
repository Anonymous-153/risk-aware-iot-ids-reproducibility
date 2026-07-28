from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib before generating figures: python -m pip install -r requirements.txt") from exc
    return plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript figures from combined summary metrics.")
    parser.add_argument("--summary", default="results/combined_summary.csv")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    plt = _require_matplotlib()
    summary = pd.read_csv(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["macro_f1_mean", "mcc_mean", "ece_mean", "conformal_coverage_mean"]:
        if metric not in summary.columns:
            continue
        figure_data = summary.pivot_table(index="model", columns="ablation", values=metric, aggfunc="mean")
        ax = figure_data.plot(kind="bar", figsize=(7, 4))
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.replace("_mean", "").replace("_", " ").title())
        ax.legend(title="Ablation", frameon=False)
        ax.figure.tight_layout()
        path = output_dir / f"{metric}.pdf"
        ax.figure.savefig(path)
        plt.close(ax.figure)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
