#!/usr/bin/env bash
# Train and evaluate two U-Net models on the IEEE PNG-filtered random
# tile-name group split (clean random baseline with no tile_name overlap).
#
# Models trained:
#   1. Original U-Net  (BCEWithLogitsLoss, Adam, lr=1e-3, batch=8)
#   2. Alea-tuned U-Net (FocalDiceLoss, AdamW, lr=9.327e-5, wd=6.088e-6, batch=16)
#
# Prerequisites:
#   - Run from the repository root directory
#   - Activate your conda/virtualenv BEFORE calling this script
#   - GPU must be available (--device cuda)
#
# Usage:
#   bash scripts/run_random_group_baselines.sh

set -euo pipefail

SPLIT_DIR="csv_splits/flood_splits_ieee_png_filtered_random_train_val_test/random_tile_name_group_split"
TRAIN_CSV="${SPLIT_DIR}/train.csv"
VAL_CSV="${SPLIT_DIR}/validation.csv"
TEST_CSV="${SPLIT_DIR}/test.csv"

LOG_DIR="outputs/random_group_baseline/logs"
SWEEP_DIR="outputs/random_group_baseline/threshold_sweeps"

mkdir -p "${LOG_DIR}" "${SWEEP_DIR}"

# ---------------------------------------------------------------------------
# 1. Original U-Net baseline
# ---------------------------------------------------------------------------
ORIG_RUN="random_group_original_unet_20epochs"
ORIG_CHECKPOINT="models/${ORIG_RUN}_best.pt"
ORIG_SWEEP="${SWEEP_DIR}/original_unet_threshold_sweep.csv"
ORIG_LOG="${LOG_DIR}/original_unet.log"

echo "============================================================"
echo " Training: Original U-Net (random group split)"
echo "   run name   : ${ORIG_RUN}"
echo "   train CSV  : ${TRAIN_CSV}"
echo "   val CSV    : ${VAL_CSV}"
echo "   test CSV   : ${TEST_CSV}"
echo "   checkpoint : ${ORIG_CHECKPOINT}"
echo "   log        : ${ORIG_LOG}"
echo "============================================================"

python3 scripts/train_unet_baseline.py \
    --train-csv      "${TRAIN_CSV}" \
    --val-csv        "${VAL_CSV}" \
    --test-csv       "${TEST_CSV}" \
    --run-name       "${ORIG_RUN}" \
    --epochs         20 \
    --batch-size     8 \
    --learning-rate  1e-3 \
    --base-channels  32 \
    --device         cuda \
    2>&1 | tee "${ORIG_LOG}"

echo ""
echo "------------------------------------------------------------"
echo " Threshold sweep: Original U-Net"
echo "   checkpoint : ${ORIG_CHECKPOINT}"
echo "   sweep out  : ${ORIG_SWEEP}"
echo "------------------------------------------------------------"

python3 scripts/threshold_sweep_single_checkpoint.py \
    --checkpoint  "${ORIG_CHECKPOINT}" \
    --val-csv     "${VAL_CSV}" \
    --test-csv    "${TEST_CSV}" \
    --output-csv  "${ORIG_SWEEP}" \
    --device      cuda \
    2>&1 | tee -a "${ORIG_LOG}"

echo ""
echo "Original U-Net complete. Sweep written to: ${ORIG_SWEEP}"
echo ""

# ---------------------------------------------------------------------------
# 2. Alea-tuned U-Net
# ---------------------------------------------------------------------------
ALEA_RUN="random_group_alea_tuned_unet_20epochs"
ALEA_CHECKPOINT="models/${ALEA_RUN}_best.pt"
ALEA_SWEEP="${SWEEP_DIR}/alea_tuned_unet_threshold_sweep.csv"
ALEA_LOG="${LOG_DIR}/alea_tuned_unet.log"

echo "============================================================"
echo " Training: Alea-tuned U-Net (random group split)"
echo "   run name   : ${ALEA_RUN}"
echo "   train CSV  : ${TRAIN_CSV}"
echo "   val CSV    : ${VAL_CSV}"
echo "   test CSV   : ${TEST_CSV}"
echo "   checkpoint : ${ALEA_CHECKPOINT}"
echo "   log        : ${ALEA_LOG}"
echo "============================================================"

python3 scripts/train_unet_baseline_tuned.py \
    --train-csv     "${TRAIN_CSV}" \
    --val-csv       "${VAL_CSV}" \
    --test-csv      "${TEST_CSV}" \
    --run-name      "${ALEA_RUN}" \
    --epochs        20 \
    --batch-size    16 \
    --learning-rate 9.327106954111342e-05 \
    --weight-decay  6.088353841746043e-06 \
    --base-channels 32 \
    --device        cuda \
    2>&1 | tee "${ALEA_LOG}"

echo ""
echo "------------------------------------------------------------"
echo " Threshold sweep: Alea-tuned U-Net"
echo "   checkpoint : ${ALEA_CHECKPOINT}"
echo "   sweep out  : ${ALEA_SWEEP}"
echo "------------------------------------------------------------"

python3 scripts/threshold_sweep_single_checkpoint.py \
    --checkpoint  "${ALEA_CHECKPOINT}" \
    --val-csv     "${VAL_CSV}" \
    --test-csv    "${TEST_CSV}" \
    --output-csv  "${ALEA_SWEEP}" \
    --device      cuda \
    2>&1 | tee -a "${ALEA_LOG}"

echo ""
echo "Alea-tuned U-Net complete. Sweep written to: ${ALEA_SWEEP}"
echo ""

echo "============================================================"
echo " All random group baseline runs complete."
echo " To summarize results run:"
echo "   python3 scripts/summarize_random_group_baselines.py"
echo "============================================================"
