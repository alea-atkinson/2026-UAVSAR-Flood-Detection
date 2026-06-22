#!/usr/bin/env bash
# run_original_unet_filtered_strict_threshold_sweeps.sh
#
# Generates per-fold threshold sweep CSVs for the original U-Net baseline
# (filtered-strict, 20-epoch checkpoints, fp1-fp7).
#
# Output format matches the Alea per-fold sweep CSVs:
#   columns: threshold, val_dice, val_iou, val_precision, val_recall,
#            test_dice, test_iou, test_precision, test_recall, selected
#   one file per held-out flight path
#
# These CSVs are consumed by compare_global_thresholds_original_vs_alea.py.
# If they already exist the comparison script will use them to avoid reloading
# checkpoints; run this script once to generate them.
#
# Usage:
#   bash scripts/run_original_unet_filtered_strict_threshold_sweeps.sh [device]
#
# Arguments:
#   device  cuda (default) or cpu
#
# Must be run from the project root:
#   cd /mnt/linuxlab/home/reujsalter/reu_flood_project/2026-UAVSAR-Flood-Detection

set -euo pipefail

DEVICE="${1:-cuda}"
SPLIT_DIR="csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/strict_no_overlap"
MODELS_DIR="models"
OUTPUT_DIR="outputs/original_unet_filtered_strict/threshold_sweeps"

echo "============================================================"
echo "  Original U-Net filtered-strict threshold sweeps"
echo "============================================================"
echo "  Device     : ${DEVICE}"
echo "  Split dir  : ${SPLIT_DIR}"
echo "  Models dir : ${MODELS_DIR}"
echo "  Output dir : ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

for FP in 1 2 3 4 5 6 7; do
    CHECKPOINT="${MODELS_DIR}/filtered_strict_fp${FP}_unet_20epochs_best.pt"
    VAL_CSV="${SPLIT_DIR}/heldout_fp${FP}_validation.csv"
    TEST_CSV="${SPLIT_DIR}/heldout_fp${FP}_test.csv"
    OUTPUT_CSV="${OUTPUT_DIR}/fp${FP}_threshold_sweep.csv"

    if [ ! -f "${CHECKPOINT}" ]; then
        echo "[WARN] Checkpoint not found, skipping fp${FP}: ${CHECKPOINT}"
        continue
    fi
    if [ ! -f "${VAL_CSV}" ]; then
        echo "[WARN] Val CSV not found, skipping fp${FP}: ${VAL_CSV}"
        continue
    fi
    if [ ! -f "${TEST_CSV}" ]; then
        echo "[WARN] Test CSV not found, skipping fp${FP}: ${TEST_CSV}"
        continue
    fi

    echo "[fp${FP}] Running threshold sweep ..."
    python scripts/threshold_sweep_single_checkpoint.py \
        --checkpoint "${CHECKPOINT}" \
        --val-csv    "${VAL_CSV}" \
        --test-csv   "${TEST_CSV}" \
        --output-csv "${OUTPUT_CSV}" \
        --device     "${DEVICE}"

    echo "[fp${FP}] Done -> ${OUTPUT_CSV}"
    echo ""
done

echo "============================================================"
echo "  All sweeps complete."
echo "  CSVs written to: ${OUTPUT_DIR}"
echo ""
echo "  Now run the comparison:"
echo "    python scripts/compare_global_thresholds_original_vs_alea.py"
echo "============================================================"
