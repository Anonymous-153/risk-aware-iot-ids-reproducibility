# Supplementary Reproducibility Notes

This package accompanies the manuscript **"Risk-Aware Conformal Prediction for IoT Intrusion Detection: Coverage, Calibration, and Deployment Cost"**.

It includes source code, experiment configurations, tests, leakage reports, sampling manifests, generated LaTeX tables, environment manifests, per-run metrics, diagnostics, and aggregate summaries.

The package does not redistribute raw public datasets, processed split CSV files, fitted model files, or large raw prediction CSV files. Obtain the raw datasets from the providers cited in the manuscript. Processed splits and large prediction files can be regenerated with the commands in `README.md`.

Useful entry points:

- `python -m unittest discover -s tests`
- `python scripts/make_manuscript_tables.py`
- `python train.py --config experiments/cybersecurity_core_latency_calibration_config.json`
- `python train.py --config experiments/cybersecurity_raw_prop_latency_refresh_config.json`
- `python scripts/audit_results.py results/cybersecurity_core_latency_calibration/cic_iot_diad_2024 --allow-missing-raw-predictions`
- `python scripts/audit_results.py results/cybersecurity_core_latency_calibration/unsw_nb15 --min-conformal-coverage 0.5 --allow-missing-raw-predictions`
- `python scripts/audit_results.py results/cybersecurity_raw_prop_latency_refresh/cic_iot_diad_2024_raw_proportion --min-conformal-coverage 0.5 --allow-missing-raw-predictions`
