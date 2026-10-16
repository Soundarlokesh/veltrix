# Windows equivalent of `make test` (GNU make is not installed by default on Windows).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\test.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m pytest
exit $LASTEXITCODE
