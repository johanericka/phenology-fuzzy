#!/usr/bin/env bash
set -euo pipefail

# Bootstrap environment baru untuk project phenology-fuzzy.
# Default: membuat virtualenv ".venv" dan menginstal dependency dari requirements.txt.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

cd "$PROJECT_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python tidak ditemukan: $PYTHON_BIN" >&2
  exit 1
fi

echo "==> Python: $("$PYTHON_BIN" --version)"
echo "==> Project: $PROJECT_ROOT"
echo "==> Virtualenv: $VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Membuat virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "==> Virtual environment sudah ada, lanjut pakai yang existing."
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ ! -x "$VENV_PY" ] || [ ! -x "$VENV_PIP" ]; then
  echo "ERROR: Virtual environment tidak valid di $VENV_DIR" >&2
  exit 1
fi

echo "==> Upgrade pip/setuptools/wheel..."
"$VENV_PIP" install --upgrade pip setuptools wheel

echo "==> Install dependency dari requirements.txt..."
"$VENV_PIP" install -r requirements.txt

echo "==> Smoke test import dependency utama..."
"$VENV_PY" - <<'PY'
mods = ["numpy", "pandas", "matplotlib", "scipy", "seaborn", "aquacrop"]
for m in mods:
    __import__(m)
print("OK: semua dependency utama berhasil diimport")
PY

cat <<EOF

Bootstrap selesai.

Aktifkan environment:
  source $VENV_DIR/bin/activate

Jalankan simulasi:
  python main.py

EOF

