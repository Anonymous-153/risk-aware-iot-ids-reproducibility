Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python environment not found at $Python. Create .venv and install requirements first."
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: ${Executable} $($Arguments -join ' ')"
    }
}

$PrepareConfig = "experiments\manuscript_prepare_config.json"
$RawPrepareConfig = "experiments\cybersecurity_cic_raw_proportion_prepare_config.json"
$CoreConfig = "experiments\cybersecurity_core_latency_calibration_config.json"
$RawConfig = "experiments\cybersecurity_raw_prop_latency_refresh_config.json"

Invoke-Step $Python -B -X utf8 scripts\verify_raw_data.py --config $PrepareConfig --manifest results\raw_data_manifest_full.csv
Invoke-Step $Python -B -X utf8 scripts\verify_raw_data.py --config $RawPrepareConfig --manifest results\raw_data_manifest_raw_proportion.csv
Invoke-Step $Python -B -X utf8 prepare_data.py --config $PrepareConfig
Invoke-Step $Python -B -X utf8 prepare_data.py --config $RawPrepareConfig

Invoke-Step $Python -B -X utf8 scripts\record_environment.py --config $CoreConfig --output results\cybersecurity_core_latency_calibration_environment_manifest.json --command train.py --config $CoreConfig
Invoke-Step $Python -B -X utf8 train.py --config $CoreConfig
Invoke-Step $Python -B -X utf8 evaluate.py --results-dir results\cybersecurity_core_latency_calibration --output results\cybersecurity_core_latency_calibration\combined_summary.csv --config $CoreConfig

Invoke-Step $Python -B -X utf8 scripts\record_environment.py --config $RawConfig --output results\cybersecurity_raw_prop_latency_refresh_environment_manifest.json --command train.py --config $RawConfig
Invoke-Step $Python -B -X utf8 train.py --config $RawConfig
Invoke-Step $Python -B -X utf8 evaluate.py --results-dir results\cybersecurity_raw_prop_latency_refresh --output results\cybersecurity_raw_prop_latency_refresh\combined_summary.csv --config $RawConfig

Invoke-Step $Python -B -X utf8 scripts\make_manuscript_tables.py
Invoke-Step $Python -B -X utf8 scripts\audit_results.py results\cybersecurity_core_latency_calibration\cic_iot_diad_2024 --allow-missing-raw-predictions
Invoke-Step $Python -B -X utf8 scripts\audit_results.py results\cybersecurity_core_latency_calibration\unsw_nb15 --min-conformal-coverage 0.5 --allow-missing-raw-predictions
Invoke-Step $Python -B -X utf8 scripts\audit_results.py results\cybersecurity_raw_prop_latency_refresh\cic_iot_diad_2024_raw_proportion --min-conformal-coverage 0.5 --allow-missing-raw-predictions

Write-Host "Manuscript experiment matrix completed."
