#!/usr/bin/env bash
# Train Alea's tuned plain U-Net on the IEEE PNG-filtered strict no-overlap
# leave-one-flight-path-out splits (fp1–fp7), then run a threshold sweep on
# each saved checkpoint.
#
# Prerequisites:
#   - Run from the repository root directory
#   - Activate your conda/virtualenv BEFORE calling this script
#   - GPU must be available (--device cuda)
#
# Usage:
#   bash scripts/run_alea_tuned_filtered_strict.sh
#
# To run a single flight path only (e.g. fp3):
#   FP=3 bash -c 'source scripts/run_alea_tuned_filtered_strict.sh'
# Or edit the FPS array below.

set -euo pipefail

FPS=(1 2 3 4 5 6 7)

SPLIT_DIR="csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/strict_no_overlap"
LOG_DIR="outputs/alea_tuned_filtered_strict/logs"
SWEEP_DIR="outputs/alea_tuned_filtered_strict/threshold_sweeps"

mkdir -p "${LOG_DIR}" "${SWEEP_DIR}"

for FP in "${FPS[@]}"; do
    RUN_NAME="alea_tuned_filtered_strict_fp${FP}_focaldice_adamw_20epochs"
    TRAIN_CSV="${SPLIT_DIR}/heldout_fp${FP}_train.csv"
    VAL_CSV="${SPLIT_DIR}/heldout_fp${FP}_validation.csv"
    TEST_CSV="${SPLIT_DIR}/heldout_fp${FP}_test.csv"
    CHECKPOINT="models/${RUN_NAME}_best.pt"
    SWEEP_CSV="${SWEEP_DIR}/fp${FP}_threshold_sweep.csv"
    LOG="${LOG_DIR}/fp${FP}.log"

    echo "============================================================"
    echo " Training heldout fp${FP}"
    echo "   run name  : ${RUN_NAME}"
    echo "   train CSV : ${TRAIN_CSV}"
    echo "   val CSV   : ${VAL_CSV}"
    echo "   test CSV  : ${TEST_CSV}"
    echo "   log       : ${LOG}"
    echo "============================================================"

    python3 scripts/train_unet_baseline_tuned.py \
        --train-csv    "${TRAIN_CSV}" \
        --val-csv      "${VAL_CSV}" \
        --test-csv     "${TEST_CSV}" \
        --run-name     "${RUN_NAME}" \
        --epochs       20 \
        --batch-size   16 \
        --device       cuda \
        2>&1 | tee "${LOG}"

    echo ""
    echo "------------------------------------------------------------"
    echo " Threshold sweep for fp${FP}"
    echo "   checkpoint : ${CHECKPOINT}"
    echo "   sweep out  : ${SWEEP_CSV}"
    echo "------------------------------------------------------------"

    python3 scripts/threshold_sweep_single_checkpoint.py \
        --checkpoint  "${CHECKPOINT}" \
        --val-csv     "${VAL_CSV}" \
        --test-csv    "${TEST_CSV}" \
        --output-csv  "${SWEEP_CSV}" \
        --device      cuda \
        2>&1 | tee -a "${LOG}"

    echo ""
    echo "fp${FP} complete. Sweep written to: ${SWEEP_CSV}"
    echo ""
done

echo "============================================================"
echo " All fp1–fp7 runs complete."
echo " To summarize results run:"
echo "   python3 scripts/summarize_alea_tuned_filtered_strict.py"
echo "============================================================"
