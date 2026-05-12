param(
    [string]$PythonBin = "python",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

# Bootstrap environment baru untuk project phenology-fuzzy (Windows PowerShell)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Invoke-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

try {
    $pyVersion = & $PythonBin --version 2>&1
} catch {
    Write-Error "Python tidak ditemukan: $PythonBin"
    exit 1
}

Invoke-Step "Python: $pyVersion"
Invoke-Step "Project: $ProjectRoot"
Invoke-Step "Virtualenv: $VenvDir"

if (-not (Test-Path $VenvDir)) {
    Invoke-Step "Membuat virtual environment..."
    & $PythonBin -m venv $VenvDir
} else {
    Invoke-Step "Virtual environment sudah ada, lanjut pakai yang existing."
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

if (-not (Test-Path $VenvPython) -or -not (Test-Path $VenvPip)) {
    Write-Error "Virtual environment tidak valid di $VenvDir"
    exit 1
}

Invoke-Step "Upgrade pip/setuptools/wheel..."
& $VenvPip install --upgrade pip setuptools wheel

Invoke-Step "Install dependency dari requirements.txt..."
& $VenvPip install -r requirements.txt

Invoke-Step "Smoke test import dependency utama..."
& $VenvPython -c "import numpy,pandas,matplotlib,scipy,seaborn,aquacrop; print('OK: semua dependency utama berhasil diimport')"

Write-Host ""
Write-Host "Bootstrap selesai."
Write-Host ""
Write-Host "Aktifkan environment:"
Write-Host "  .\$VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Jalankan simulasi:"
Write-Host "  python main.py"
Write-Host ""

