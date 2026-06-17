#!/usr/bin/env bash
set -euo pipefail

# Recreate the local TensorFlow/Jupyter environment used for the SPIE U-Net runs.
# Usage: bash scripts/setup_spie_unet_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv-tf312"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

cd "$PROJECT_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
	echo "Could not find $PYTHON_BIN."
	echo "Install Python 3.12 or rerun with PYTHON_BIN=/path/to/python."
	exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
	"$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r requirements-spie-unet.txt
"$VENV_DIR/bin/python" -m ipykernel install --user --name spie-unet-tf312 --display-name "SPIE U-Net TF 3.12"

echo
echo "Environment ready."
echo "For terminal runs:"
echo "  source .venv-tf312/bin/activate"
echo "  source scripts/set_unet_env.sh"
echo
echo "For VS Code notebooks, select kernel: SPIE U-Net TF 3.12"
echo "Then source scripts/set_unet_env.sh in the shell that launches VS Code/Jupyter if GPU paths are missing."
