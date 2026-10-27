$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PIP_RETRIES = "15"
$env:PIP_TIMEOUT = "120"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    if (Test-Path -LiteralPath $BundledPython) {
        & $BundledPython -m venv (Join-Path $ProjectRoot ".venv")
    } else {
        $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $PyLauncher) { throw "Install 64-bit Python 3.12 or later, then rerun this script." }
        & py -3.12 -m venv (Join-Path $ProjectRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "A 64-bit Python 3.12+ interpreter was not found. Install one and rerun this script." }
    }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip in the project virtual environment." }
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Review the error above, then retry." }
& $VenvPython -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'); raise SystemExit(0 if torch.cuda.is_available() else 'CUDA is unavailable; review the NVIDIA driver and CUDA wheel install.')"
if ($LASTEXITCODE -ne 0) { throw "PyTorch could not access CUDA. Review the NVIDIA driver and CUDA wheel install." }
