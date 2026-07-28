from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.reporting import DISPLAY_LABELS


STRICT_RISK_ABLATION = "risk_aware_full_b0.01_a0.001"
RAW_PROP_ABLATION = "risk_aware_full_b0.05_a0.005"
STRICT_PLATT_ABLATION = "risk_aware_platt_full_b0.01_a0.001"
MODEL_ORDER = ["logistic_regression", "mlp", "random_forest", "xgboost"]


def _display(value: object) -> str:
    return DISPLAY_LABELS.get(str(value), str(value)).replace("_", r"\_")


def _fmt_mean_std(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _fmt_latency(row: pd.Series) -> str:
    return _fmt_mean_std(row, "end_to_end_latency_ms_per_sample", digits=5)


def _fmt_model_latency(row: pd.Series) -> str:
    return _fmt_mean_std(row, "model_only_latency_ms_per_sample", digits=5)


def _model_sort_key(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def _selected_operating_points(risk_summary: pd.DataFrame, raw_summary: pd.DataFrame) -> pd.DataFrame:
    risk_rows = risk_summary[
        risk_summary["ablation"].eq(STRICT_RISK_ABLATION)
        & risk_summary["dataset"].isin(["cic_iot_diad_2024", "unsw_nb15"])
    ].copy()
    risk_rows["operating_point"] = "strict 0.01/0.001"

    raw_rows = raw_summary[
        raw_summary["ablation"].eq(RAW_PROP_ABLATION)
        & raw_summary["dataset"].eq("cic_iot_diad_2024_raw_proportion")
    ].copy()
    raw_rows["operating_point"] = "raw-prop. 0.05/0.005"

    combined = pd.concat([risk_rows, raw_rows], ignore_index=True)
    combined["dataset_order"] = combined["dataset"].map(
        {"cic_iot_diad_2024": 0, "unsw_nb15": 1, "cic_iot_diad_2024_raw_proportion": 2}
    )
    combined["model_order"] = combined["model"].map(_model_sort_key)
    return combined.sort_values(["dataset_order", "model_order"]).reset_index(drop=True)


def deployment_cost_table_to_latex(risk_summary: pd.DataFrame, raw_summary: pd.DataFrame) -> str:
    rows = _selected_operating_points(risk_summary, raw_summary)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lllrrrr@{}}",
        r"\toprule",
        r"Dataset & Model & Operating Point & Model-only & End-to-end & Set Size & Singleton \\",
        r"\midrule",
    ]
    for _, row in rows.iterrows():
        values = [
            _display(row["dataset"]),
            _display(row["model"]),
            str(row["operating_point"]),
            _fmt_model_latency(row),
            _fmt_latency(row),
            _fmt_mean_std(row, "average_set_size"),
            _fmt_mean_std(row, "singleton_rate"),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Machine-generated deployment-cost indicators for selected risk-aware operating points. "
                r"Latency is measured in milliseconds per sample on the local experiment workstation; "
                r"end-to-end timing starts from a test flow-feature row and includes preprocessing, model inference, "
                r"probability calibration, and conformal set generation.}"
            ),
            r"\label{tab:deployment-cost}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def unsw_failure_diagnostic_table_to_latex(
    risk_summary: pd.DataFrame,
    confidence_diagnostics: pd.DataFrame,
) -> str:
    summary_rows = risk_summary[
        risk_summary["dataset"].eq("unsw_nb15") & risk_summary["ablation"].eq(STRICT_RISK_ABLATION)
    ].copy()
    summary_rows["model_order"] = summary_rows["model"].map(_model_sort_key)
    summary_rows = summary_rows.sort_values("model_order")

    uncovered = confidence_diagnostics[
        confidence_diagnostics["dataset"].eq("unsw_nb15")
        & confidence_diagnostics["ablation"].eq(STRICT_RISK_ABLATION)
        & confidence_diagnostics["covered"].eq(0)
    ].set_index("model")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        (
            r"Model & Coverage & Attack Cov. & Singleton & Uncovered $n$ & $p_y$ Uncov. "
            r"& Max Prob. & Margin & Unique Max \\"
        ),
        r"\midrule",
    ]
    for _, row in summary_rows.iterrows():
        model = str(row["model"])
        diagnostic = uncovered.loc[model]
        values = [
            _display(model),
            _fmt_mean_std(row, "conformal_coverage"),
            _fmt_mean_std(row, "attack_coverage"),
            _fmt_mean_std(row, "singleton_rate"),
            str(int(diagnostic["sample_count"])),
            f"{float(diagnostic['mean_true_label_probability']):.3f}",
            f"{float(diagnostic['mean_max_probability']):.3f}",
            f"{float(diagnostic['mean_probability_margin']):.3f}",
            str(int(diagnostic["unique_max_probability_6dp"])),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Machine-generated UNSW-NB15 diagnostics for the strict attack-risk setting. "
                r"Uncovered probabilities are aggregated over all five seeds.}"
            ),
            r"\label{tab:unsw-failure-diagnostics}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def calibration_baseline_table_to_latex(summary: pd.DataFrame) -> str:
    rows = summary[
        summary["dataset"].isin(["cic_iot_diad_2024", "unsw_nb15"])
        & summary["ablation"].isin([STRICT_RISK_ABLATION, STRICT_PLATT_ABLATION])
    ].copy()
    rows["dataset_order"] = rows["dataset"].map({"cic_iot_diad_2024": 0, "unsw_nb15": 1})
    rows["model_order"] = rows["model"].map(_model_sort_key)
    rows["calibrator"] = rows["ablation"].map(
        {
            STRICT_RISK_ABLATION: "Binned risk-aware",
            STRICT_PLATT_ABLATION: "Risk-aware Platt",
        }
    )
    rows["calibrator_order"] = rows["ablation"].map({STRICT_RISK_ABLATION: 0, STRICT_PLATT_ABLATION: 1})
    rows = rows.sort_values(["dataset_order", "model_order", "calibrator_order"]).reset_index(drop=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lllrrrr@{}}",
        r"\toprule",
        r"Dataset & Model & Calibrator & ECE & Brier & Attack Cov. & Set Size \\",
        r"\midrule",
    ]
    for _, row in rows.iterrows():
        values = [
            _display(row["dataset"]),
            _display(row["model"]),
            str(row["calibrator"]),
            _fmt_mean_std(row, "ece"),
            _fmt_mean_std(row, "brier_score"),
            _fmt_mean_std(row, "attack_coverage"),
            _fmt_mean_std(row, "average_set_size"),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Machine-generated calibration baseline comparison under the strict attack-risk setting. "
                r"The binned and Platt rows use the same train, calibration, and test splits.}"
            ),
            r"\label{tab:calibration-baseline}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_manuscript_tables(
    risk_summary_path: str | Path,
    raw_summary_path: str | Path,
    confidence_diagnostics_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    risk_summary = pd.read_csv(risk_summary_path)
    raw_summary = pd.read_csv(raw_summary_path)
    confidence_diagnostics = pd.read_csv(confidence_diagnostics_path)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    deployment_path = destination / "deployment_cost_metrics.tex"
    diagnostics_path = destination / "unsw_failure_diagnostics_metrics.tex"
    calibration_path = destination / "calibration_baseline_metrics.tex"
    deployment_path.write_text(deployment_cost_table_to_latex(risk_summary, raw_summary), encoding="utf-8")
    diagnostics_path.write_text(
        unsw_failure_diagnostic_table_to_latex(risk_summary, confidence_diagnostics),
        encoding="utf-8",
    )
    calibration_path.write_text(calibration_baseline_table_to_latex(risk_summary), encoding="utf-8")
    return [deployment_path, diagnostics_path, calibration_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript tables from existing CSV artifacts.")
    parser.add_argument("--risk-summary", default="results/cybersecurity_core_latency_calibration/combined_summary.csv")
    parser.add_argument("--raw-summary", default="results/cybersecurity_raw_prop_latency_refresh/combined_summary.csv")
    parser.add_argument(
        "--confidence-diagnostics",
        default="results/cybersecurity_core_latency_calibration/unsw_nb15/diagnostics/confidence_diagnostics.csv",
    )
    parser.add_argument("--output-dir", default="paper/tables")
    args = parser.parse_args()

    for path in write_manuscript_tables(
        risk_summary_path=args.risk_summary,
        raw_summary_path=args.raw_summary,
        confidence_diagnostics_path=args.confidence_diagnostics,
        output_dir=args.output_dir,
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
