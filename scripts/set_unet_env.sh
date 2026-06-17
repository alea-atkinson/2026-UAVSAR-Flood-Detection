#!/usr/bin/env bash
# Usage: source scripts/set_unet_env.sh
# Automatically find `Preprocessed-128` under the repository and export `UNET_128_DATA_DIR`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try to find the git top-level as project root, otherwise use parent of script
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
	PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
	PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Find the first `Preprocessed-128` directory under project root (limit depth to avoid long searches)
CANDIDATE="$(find "$PROJECT_ROOT" -maxdepth 6 -type d -name 'Preprocessed-128' -print -quit 2>/dev/null)"

if [ -n "$CANDIDATE" ]; then
	UNET_128_DATA_DIR="$CANDIDATE"
else
	# Fallback to common location
	UNET_128_DATA_DIR="$PROJECT_ROOT/spie/spie/Preprocessed-128"
fi

export UNET_128_DATA_DIR
echo "Exported UNET_128_DATA_DIR=$UNET_128_DATA_DIR"

PROJECT_TMP="$PROJECT_ROOT/.tmp"
mkdir -p "$PROJECT_TMP"
export TMPDIR="$PROJECT_TMP"
echo "Exported TMPDIR=$TMPDIR"

MPLCONFIGDIR="$PROJECT_TMP/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR
echo "Exported MPLCONFIGDIR=$MPLCONFIGDIR"

JUPYTER_CONFIG_DIR="$PROJECT_TMP/jupyter-config"
JUPYTER_DATA_DIR="$PROJECT_TMP/jupyter-data"
JUPYTER_RUNTIME_DIR="$PROJECT_TMP/jupyter-runtime"
IPYTHONDIR="$PROJECT_TMP/ipython"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"
export JUPYTER_CONFIG_DIR JUPYTER_DATA_DIR JUPYTER_RUNTIME_DIR IPYTHONDIR
echo "Exported Jupyter config/data/runtime dirs under $PROJECT_TMP"

PROJECT_VENV="$PROJECT_ROOT/.venv-tf312"
NVIDIA_SITE_PACKAGES="$PROJECT_VENV/lib64/python3.12/site-packages/nvidia"

if [ -d "$NVIDIA_SITE_PACKAGES" ]; then
	CUDA_LIBRARY_DIRS="$(find "$NVIDIA_SITE_PACKAGES" -maxdepth 3 -type d -name lib | paste -sd: -)"
	if [ -n "$CUDA_LIBRARY_DIRS" ]; then
		if [ -n "${LD_LIBRARY_PATH:-}" ]; then
			export LD_LIBRARY_PATH="$CUDA_LIBRARY_DIRS:$LD_LIBRARY_PATH"
		else
			export LD_LIBRARY_PATH="$CUDA_LIBRARY_DIRS"
		fi
		echo "Prepended TensorFlow CUDA library paths to LD_LIBRARY_PATH"
	fi

	CUDA_NVCC_DIR="$NVIDIA_SITE_PACKAGES/cuda_nvcc"
	if [ -d "$CUDA_NVCC_DIR" ]; then
		export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_NVCC_DIR ${XLA_FLAGS:-}"
		echo "Exported XLA_FLAGS for CUDA libdevice discovery"
	fi
fi

echo "Source this script in the shell you use to launch Jupyter/VS Code so the kernel inherits these variables."
