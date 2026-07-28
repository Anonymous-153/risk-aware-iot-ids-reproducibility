param(
    [switch]$ListOnly,
    [string]$Proxy = "http://127.0.0.1:7890",
    [string[]]$IncludePrefix = @(),
    [switch]$NoProxy,
    [switch]$SkipInsecureTls
)

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

$DownloadArgs = @("-B", "-X", "utf8", "scripts\download_cic_iot_diad.py")
if (-not $NoProxy) {
    $DownloadArgs += @("--proxy", $Proxy)
}
if (-not $SkipInsecureTls) {
    $DownloadArgs += "--insecure-tls"
}
if ($ListOnly) {
    $DownloadArgs += "--list-only"
}
foreach ($Prefix in $IncludePrefix) {
    $DownloadArgs += @("--include-prefix", $Prefix)
}

Invoke-Step $Python @DownloadArgs
