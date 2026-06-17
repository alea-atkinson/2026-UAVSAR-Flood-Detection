# Setup and Recovery Guide

This guide is for recreating the environment and rerunning the SPIE U-Net experiments on another computer.

The repo should store code, setup instructions, and small metadata/results. It should not store the virtual environment, TIFF datasets, generated symlink datasets, or trained model weights in normal git.

## What To Commit

Commit these files because they help recover quickly:

- `requirements-spie-unet.txt`
- `scripts/setup_spie_unet_env.sh`
- `scripts/set_unet_env.sh`
- `scripts/filter_preprocessed_by_water_ratio.py`
- `scripts/report_paper_style_dataset.py`
- `scripts/train_paper_style_model.py`
- `scripts/train_paper_style_unet.py`
- `scripts/train_paper_style_attention_unet.py`
- `scripts/train_paper_style_unetpp.py`
- `scripts/write_paper_style_comparison.py`
- `spie/paper_style_rerun/*.md`
- `spie/paper_style_rerun/*.json`
- `spie/paper_style_rerun/artifacts/*/config.json`

Do not commit these unless you intentionally use Git LFS or another data store:

- `.venv-tf312/`
- `spie/Preprocessed-128/`
- `spie/Preprocessed-128-paper-filtered/`
- `*.h5`
- `*.keras`
- notebook checkpoint directories
- cache directories

## Fresh Setup

From a fresh clone:

```bash
cd 2026-UAVSAR-Flood-Detection
bash scripts/setup_spie_unet_env.sh
source .venv-tf312/bin/activate
source scripts/set_unet_env.sh
```

Verify TensorFlow and GPU visibility:

```bash
.venv-tf312/bin/python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
PY
```

If the GPU list is empty, check:

```bash
nvidia-smi
source scripts/set_unet_env.sh
```

## Dataset Placement

The original TIFF dataset is not committed. Put it back at:

```text
spie/Preprocessed-128/
```

Expected structure:

```text
spie/Preprocessed-128/
  train/sar/
  train/flood/
  val/sar/
  val/flood/
  test/sar/
  test/flood/
  metadata/
```

The available repo dataset used here had:

- 25,168 original SAR/flood pairs
- train: 17,465
- validation: 3,850
- test: 3,853

## Recreate The Water-Filtered Dataset

The filtered dataset is a symlink view, so it is small but depends on `spie/Preprocessed-128/` existing.

```bash
.venv-tf312/bin/python scripts/filter_preprocessed_by_water_ratio.py
.venv-tf312/bin/python scripts/report_paper_style_dataset.py
```

Expected filtered report:

- 15,714 filtered pairs
- train: 11,000
- validation: 2,357
- test: 2,357
- water-ratio rule: `0.05 <= water_ratio <= 0.85`

## Run Models

Full paper-style setting:

```bash
.venv-tf312/bin/python scripts/train_paper_style_model.py --model unet --epochs 80
.venv-tf312/bin/python scripts/train_paper_style_model.py --model attention_unet --epochs 80
.venv-tf312/bin/python scripts/train_paper_style_model.py --model unetpp --epochs 80
```

Faster preliminary run:

```bash
.venv-tf312/bin/python scripts/train_paper_style_model.py --model unet --epochs 20
.venv-tf312/bin/python scripts/train_paper_style_model.py --model attention_unet --epochs 20
.venv-tf312/bin/python scripts/train_paper_style_model.py --model unetpp --epochs 20
```

Update the comparison table:

```bash
.venv-tf312/bin/python scripts/write_paper_style_comparison.py
```

## Label Results Carefully

The exact curated SPIE 3,873-tile dataset is unavailable in this repo. Results from this workflow should be labeled:

```text
Paper-style rerun on the available Preprocessed-128 repo dataset after water filtering.
```

Do not describe it as an exact reproduction of the SPIE dataset.
