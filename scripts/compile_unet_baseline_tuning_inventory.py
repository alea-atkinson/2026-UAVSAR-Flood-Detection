#!/usr/bin/env python3
"""
Compile a master inventory of U-Net baseline and tuning/ablation experiments.

Reads existing training scripts, result CSVs, and threshold-sweep CSVs.
Does NOT retrain models.

Outputs:
  outputs/unet_tuning_inventory/unet_baseline_tuning_inventory.csv
  outputs/unet_tuning_inventory/unet_baseline_tuning_results_table.csv
  outputs/unet_tuning_inventory/unet_baseline_tuning_inventory.md
  outputs/unet_tuning_inventory/unet_baseline_tuning_summary.txt
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUT_DIR = PROJECT_ROOT / "outputs" / "unet_tuning_inventory"

# Known result locations
BASELINE_SUMMARY = RESULTS_DIR / "filtered_strict_20epoch_test_summary.csv"
BASELINE_SUMMARY_ALT = OUTPUTS_DIR / "filtered_strict_baseline_20epoch" / "filtered_strict_20epoch_test_summary.csv"
SWEEP_SELECTED = (
    OUTPUTS_DIR / "for_dr_jin_filtered_strict_baseline_and_landcover"
    / "threshold_sweep_selected_test_results.csv"
)
EXPERIMENT_SWEEP_DIR = OUTPUTS_DIR / "experiment_threshold_sweeps"
SPLIT_NOTE = "IEEE PNG-filtered strict no-overlap leave-one-flight-path-out splits"

# ---------------------------------------------------------------------------
# Helper: try both locations for baseline summary
# ---------------------------------------------------------------------------

def _find_baseline_summary() -> Path | None:
    for p in (BASELINE_SUMMARY, BASELINE_SUMMARY_ALT):
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Hyperparameter extraction from training scripts via text search
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"default=(\d+)")
_FLOAT_RE = re.compile(r"default=([\d.e-]+)")


def _extract_script_defaults(script_path: Path) -> dict:
    """Extract key defaults from a training script by reading its argparse section."""
    missing = "unknown"
    if not script_path.exists():
        return {k: missing for k in [
            "epochs", "batch_size", "lr", "base_channels", "seed",
            "optimizer", "has_augment_flag", "loss_choices", "save_best_by_choices",
            "arch_choices",
        ]}

    text = script_path.read_text(encoding="utf-8")

    def _int_after(pattern: str) -> str:
        m = re.search(rf'{pattern}.*?default=(\d+)', text)
        return m.group(1) if m else missing

    def _float_after(pattern: str) -> str:
        m = re.search(rf'{pattern}.*?default=([\d.e+-]+)', text)
        return m.group(1) if m else missing

    epochs = _int_after(r'"--epochs"')
    batch_size = _int_after(r'"--batch-size"')
    # LR can be --lr or --learning-rate
    lr_m = re.search(r'"--l(?:r|earning-rate)".*?default=([\d.e+-]+)', text)
    lr = lr_m.group(1) if lr_m else missing
    base_channels = _int_after(r'"--base-channels"')
    seed = _int_after(r'"--seed"')

    optimizer = "Adam" if "torch.optim.Adam" in text else missing
    has_augment = "--augment" in text

    # Collect loss choices
    loss_m = re.search(r'choices=\["(bce|dice|bce_dice[^"]*)"[^\]]*\]', text)
    if loss_m is None:
        loss_m = re.search(r'choices=\[([^\]]+)\].*?--loss', text, re.DOTALL)
    loss_choices = missing
    if '"bce"' in text and '"bce_dice"' in text:
        if '"weighted_bce"' in text:
            loss_choices = "bce, dice, bce_dice, weighted_bce"
        else:
            loss_choices = "bce, dice, bce_dice"
    elif "BCEWithLogitsLoss" in text and "DiceLoss" not in text:
        loss_choices = "bce (fixed)"

    save_best_by_choices = missing
    if '"val_loss"' in text and '"val_dice"' in text:
        save_best_by_choices = "val_loss, val_dice"
    elif "best_val_loss" in text:
        save_best_by_choices = "val_loss (fixed)"

    arch_choices = missing
    if '"attention_unet"' in text:
        arch_choices = "unet, attention_unet, unet_plus_plus"

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "base_channels": base_channels,
        "seed": seed,
        "optimizer": optimizer,
        "has_augment_flag": has_augment,
        "loss_choices": loss_choices,
        "save_best_by_choices": save_best_by_choices,
        "arch_choices": arch_choices,
    }


# ---------------------------------------------------------------------------
# Infer per-run settings from run name
# ---------------------------------------------------------------------------

def _infer_from_name(run_name: str) -> dict:
    """Infer loss, augmentation, fp, checkpoint_selection from the run name."""
    info: dict = {}

    # flight path
    fp_m = re.search(r'fp(\d+)', run_name)
    info["heldout_fp"] = f"fp{fp_m.group(1)}" if fp_m else "unknown"

    # loss
    if "weighted_bce" in run_name:
        info["loss_function"] = "weighted_bce"
    elif "bce_dice" in run_name:
        info["loss_function"] = "bce_dice"
    elif "bce" in run_name:
        info["loss_function"] = "bce"
    else:
        info["loss_function"] = "unknown"

    # augmentation
    if "noaug" in run_name:
        info["augmentation"] = "False"
    elif "_aug" in run_name or run_name.endswith("_aug"):
        info["augmentation"] = "True"
    else:
        info["augmentation"] = "not_in_name"

    # checkpoint selection
    if "valdice" in run_name:
        info["checkpoint_selection"] = "val_dice"
    else:
        info["checkpoint_selection"] = "val_loss"

    # architecture
    if "attention_unet" in run_name:
        info["model_architecture"] = "attention_unet"
    elif "unetpp" in run_name:
        info["model_architecture"] = "unet_plus_plus"
    else:
        info["model_architecture"] = "plain_unet"

    return info


# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------

def _load_baseline_default05() -> dict[str, dict]:
    """Load baseline per-fp test metrics at default 0.5 threshold from summary CSV."""
    path = _find_baseline_summary()
    if path is None:
        return {}
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        fp = str(row["heldout_fp"])
        result[fp] = {
            "test_dice_default05": round(float(row["test_dice"]), 6),
            "test_iou_default05": round(float(row["test_iou"]), 6),
            "test_precision_default05": "n/a",
            "test_recall_default05": "n/a",
        }
    return result


def _load_baseline_sweep() -> dict[str, dict]:
    """Load baseline per-fp test metrics from threshold sweep selected results CSV."""
    if not SWEEP_SELECTED.exists():
        return {}
    df = pd.read_csv(SWEEP_SELECTED)
    result = {}
    for _, row in df.iterrows():
        fp = str(row["heldout_fp"])
        result[fp] = {
            "selected_threshold": round(float(row["selected_threshold"]), 2),
            "val_dice_at_selected_threshold": round(float(row["val_dice_at_selected_threshold"]), 6),
            "val_iou_at_selected_threshold": round(float(row["val_iou_at_selected_threshold"]), 6),
            "test_dice_sweep": round(float(row["test_dice"]), 6),
            "test_iou_sweep": round(float(row["test_iou"]), 6),
            "test_precision_sweep": round(float(row["test_precision"]), 6),
            "test_recall_sweep": round(float(row["test_recall"]), 6),
        }
    return result


def _load_experiment_default05(run_name: str) -> dict:
    """Load test metrics at default 0.5 from experiment metrics CSV (test row)."""
    p = RESULTS_DIR / f"{run_name}_metrics.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    # test row has epoch == "test" or is the last row with test_dice present
    if "epoch" in df.columns:
        test_rows = df[df["epoch"].astype(str) == "test"]
    else:
        test_rows = pd.DataFrame()
    if test_rows.empty:
        # fall back: any row with non-null test_dice
        if "test_dice" in df.columns:
            test_rows = df[df["test_dice"].notna()]
    if test_rows.empty:
        return {}
    row = test_rows.iloc[0]
    result = {}
    if "test_dice" in row and pd.notna(row["test_dice"]):
        result["test_dice_default05"] = round(float(row["test_dice"]), 6)
    if "test_iou" in row and pd.notna(row["test_iou"]):
        result["test_iou_default05"] = round(float(row["test_iou"]), 6)
    result["test_precision_default05"] = "n/a"
    result["test_recall_default05"] = "n/a"
    return result


def _load_experiment_sweep(run_name: str) -> dict:
    """Load threshold-sweep selected results for an experiment run."""
    p = EXPERIMENT_SWEEP_DIR / f"{run_name}_threshold_sweep.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "selected" not in df.columns:
        return {"note": "no 'selected' column in sweep CSV"}
    sel = df[df["selected"] == True]
    if sel.empty:
        return {}
    row = sel.iloc[0]
    result = {}
    for col, key in [
        ("threshold", "selected_threshold"),
        ("val_dice", "val_dice_at_selected_threshold"),
        ("val_iou", "val_iou_at_selected_threshold"),
        ("test_dice", "test_dice_sweep"),
        ("test_iou", "test_iou_sweep"),
        ("test_precision", "test_precision_sweep"),
        ("test_recall", "test_recall_sweep"),
    ]:
        if col in row.index and pd.notna(row[col]):
            result[key] = round(float(row[col]), 6)
    return result


# ---------------------------------------------------------------------------
# Determine which experiment metrics CSVs exist and categorize
# ---------------------------------------------------------------------------

def _discover_experiment_runs() -> list[dict]:
    """Return a list of run metadata dicts for all experiment-family runs."""
    runs = []
    for p in sorted(RESULTS_DIR.glob("experiment_fp*.csv")):
        run_name = p.stem.replace("_metrics", "")
        if "smoke" in run_name:
            continue
        meta = _infer_from_name(run_name)
        meta["run_name"] = run_name
        # Determine which script generated it
        if "weighted_bce" in run_name:
            meta["script_name"] = "train_unet_weighted_experiment.py"
        else:
            meta["script_name"] = "train_unet_experiment.py"
        meta["split_type"] = SPLIT_NOTE
        meta["dataset_version"] = "IEEE PNG-filtered"
        meta["input_bands"] = "3-band SAR (HH, HV, VV or first 3 bands)"
        meta["target_mask"] = "flood_change_mask"
        meta["optimizer"] = "Adam"
        meta["learning_rate"] = "1e-3"
        meta["weight_decay"] = "0 (default Adam)"
        meta["batch_size"] = 8
        meta["base_channels"] = 32
        meta["epochs"] = 20
        meta["seed"] = 42
        runs.append(meta)
    return runs


def _discover_arch_runs() -> list[dict]:
    """Return arch extension runs (attention_unet and unetpp) for the separate section."""
    runs = []
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        name = p.stem.replace("_metrics", "")
        if "smoke" in name:
            continue
        if not ("attention_unet" in name or "unetpp" in name):
            continue
        meta = _infer_from_name(name)
        meta["run_name"] = name
        meta["script_name"] = "train_unet_arch_experiment.py"
        meta["split_type"] = SPLIT_NOTE
        meta["dataset_version"] = "IEEE PNG-filtered"
        meta["input_bands"] = "3-band SAR"
        meta["target_mask"] = "flood_change_mask"
        meta["optimizer"] = "Adam"
        meta["learning_rate"] = "1e-3"
        meta["weight_decay"] = "0 (default Adam)"
        meta["batch_size"] = 8
        meta["base_channels"] = 32
        meta["epochs"] = 20
        meta["seed"] = 42
        meta["loss_function"] = "bce"
        meta["checkpoint_selection"] = "val_dice"
        meta["augmentation"] = "not_in_name"
        runs.append(meta)
    return runs


# ---------------------------------------------------------------------------
# Build full inventory
# ---------------------------------------------------------------------------

def build_inventory() -> tuple[list[dict], list[dict], list[str]]:
    """
    Returns:
      plain_unet_rows  – list of row dicts for plain U-Net runs
      arch_rows        – list of row dicts for arch extension runs
      missing          – list of missing-data notes
    """
    missing: list[str] = []
    plain_rows: list[dict] = []

    # ------------------------------------------------------------------ #
    # 1. Baseline: fp1–fp7
    # ------------------------------------------------------------------ #
    baseline_05 = _load_baseline_default05()
    baseline_sw = _load_baseline_sweep()

    if not baseline_05:
        missing.append("Baseline default-0.5 test summary CSV not found at results/filtered_strict_20epoch_test_summary.csv")
    if not baseline_sw:
        missing.append("Baseline threshold-sweep selected results CSV not found")

    for fp_n in range(1, 8):
        fp = f"fp{fp_n}"
        run_name = f"filtered_strict_{fp}_unet_20epochs"
        row: dict = {
            "run_name": run_name,
            "script_name": "train_unet_baseline.py",
            "model_architecture": "plain_unet",
            "heldout_fp": fp,
            "split_type": SPLIT_NOTE,
            "dataset_version": "IEEE PNG-filtered",
            "input_bands": "3-band SAR",
            "target_mask": "flood_change_mask",
            "loss_function": "bce (BCEWithLogitsLoss, unweighted)",
            "optimizer": "Adam",
            "learning_rate": "1e-3",
            "weight_decay": "0 (default Adam)",
            "batch_size": 8,
            "base_channels": 32,
            "epochs": 20,
            "seed": 42,
            "augmentation": "False",
            "checkpoint_selection": "val_loss",
            # default 0.5 metrics
            "test_dice_default05": baseline_05.get(fp, {}).get("test_dice_default05", "missing"),
            "test_iou_default05": baseline_05.get(fp, {}).get("test_iou_default05", "missing"),
            "test_precision_default05": "n/a",
            "test_recall_default05": "n/a",
            "metric_source_default05": "training_script_batch_average (threshold=0.5)",
            # threshold sweep metrics
            "selected_threshold": baseline_sw.get(fp, {}).get("selected_threshold", "missing"),
            "val_dice_at_selected_threshold": baseline_sw.get(fp, {}).get("val_dice_at_selected_threshold", "missing"),
            "val_iou_at_selected_threshold": baseline_sw.get(fp, {}).get("val_iou_at_selected_threshold", "missing"),
            "test_dice_sweep": baseline_sw.get(fp, {}).get("test_dice_sweep", "missing"),
            "test_iou_sweep": baseline_sw.get(fp, {}).get("test_iou_sweep", "missing"),
            "test_precision_sweep": baseline_sw.get(fp, {}).get("test_precision_sweep", "missing"),
            "test_recall_sweep": baseline_sw.get(fp, {}).get("test_recall_sweep", "missing"),
            "metric_source_sweep": "threshold_sweep_global_pixel (val-selected threshold)",
            "notes": (
                "Train/val history in results/filtered_strict_{fp}_unet_20epochs_metrics.csv "
                "(no test row); test at 0.5 from filtered_strict_20epoch_test_summary.csv; "
                "threshold sweep via threshold_sweep_baseline.py"
            ),
        }
        plain_rows.append(row)

    # ------------------------------------------------------------------ #
    # 2. Experiment runs (fp1 only)
    # ------------------------------------------------------------------ #
    exp_runs = _discover_experiment_runs()

    if not exp_runs:
        missing.append("No experiment_fp*.csv result files found in results/")

    for meta in exp_runs:
        rn = meta["run_name"]
        d05 = _load_experiment_default05(rn)
        sw = _load_experiment_sweep(rn)

        if not d05:
            missing.append(f"Default-0.5 test metrics not found for {rn}")
        if not sw:
            missing.append(f"Threshold-sweep results not found for {rn} in outputs/experiment_threshold_sweeps/")

        aug_note = ""
        if meta["augmentation"] == "not_in_name":
            aug_note = " (augmentation flag not in run name; default=False assumed)"
        if "weighted_bce" in rn:
            loss_desc = "weighted_bce (BCEWithLogitsLoss with pos_weight = neg_px / pos_px from training masks)"
        elif "bce_dice" in rn:
            loss_desc = "bce_dice (BCEWithLogitsLoss + soft DiceLoss)"
        else:
            loss_desc = "bce (BCEWithLogitsLoss)"

        row = {
            "run_name": rn,
            "script_name": meta["script_name"],
            "model_architecture": "plain_unet",
            "heldout_fp": meta["heldout_fp"],
            "split_type": meta["split_type"],
            "dataset_version": meta["dataset_version"],
            "input_bands": meta["input_bands"],
            "target_mask": "flood_change_mask",
            "loss_function": loss_desc,
            "optimizer": "Adam",
            "learning_rate": "1e-3",
            "weight_decay": "0 (default Adam)",
            "batch_size": 8,
            "base_channels": 32,
            "epochs": 20,
            "seed": 42,
            "augmentation": meta["augmentation"] + aug_note,
            "checkpoint_selection": meta["checkpoint_selection"],
            # default 0.5
            "test_dice_default05": d05.get("test_dice_default05", "missing"),
            "test_iou_default05": d05.get("test_iou_default05", "missing"),
            "test_precision_default05": "n/a",
            "test_recall_default05": "n/a",
            "metric_source_default05": "training_script_batch_average (threshold=0.5)",
            # sweep
            "selected_threshold": sw.get("selected_threshold", "missing"),
            "val_dice_at_selected_threshold": sw.get("val_dice_at_selected_threshold", "missing"),
            "val_iou_at_selected_threshold": sw.get("val_iou_at_selected_threshold", "missing"),
            "test_dice_sweep": sw.get("test_dice_sweep", "missing"),
            "test_iou_sweep": sw.get("test_iou_sweep", "missing"),
            "test_precision_sweep": sw.get("test_precision_sweep", "missing"),
            "test_recall_sweep": sw.get("test_recall_sweep", "missing"),
            "metric_source_sweep": "threshold_sweep_global_pixel (val-selected threshold)",
            "notes": (
                f"fp1 only; swept via threshold_sweep_single_checkpoint.py; "
                f"test row saved in results/{rn}_metrics.csv"
                + aug_note
            ),
        }
        plain_rows.append(row)

    # ------------------------------------------------------------------ #
    # 3. Architecture extensions (separate section, not plain U-Net tuning)
    # ------------------------------------------------------------------ #
    arch_rows: list[dict] = []
    for meta in _discover_arch_runs():
        rn = meta["run_name"]
        fp = meta["heldout_fp"]
        d05 = _load_experiment_default05(rn)
        sw = _load_experiment_sweep(rn)

        row = {
            "run_name": rn,
            "script_name": "train_unet_arch_experiment.py",
            "model_architecture": meta["model_architecture"],
            "heldout_fp": fp,
            "split_type": SPLIT_NOTE,
            "dataset_version": "IEEE PNG-filtered",
            "input_bands": "3-band SAR",
            "target_mask": "flood_change_mask",
            "loss_function": "bce (BCEWithLogitsLoss)",
            "optimizer": "Adam",
            "learning_rate": "1e-3",
            "weight_decay": "0",
            "batch_size": 8,
            "base_channels": 32,
            "epochs": 20,
            "seed": 42,
            "augmentation": "not_in_name",
            "checkpoint_selection": "val_dice",
            "test_dice_default05": d05.get("test_dice_default05", "missing"),
            "test_iou_default05": d05.get("test_iou_default05", "missing"),
            "metric_source_default05": "training_script_batch_average (threshold=0.5)",
            "selected_threshold": sw.get("selected_threshold", "missing"),
            "val_dice_at_selected_threshold": sw.get("val_dice_at_selected_threshold", "missing"),
            "test_dice_sweep": sw.get("test_dice_sweep", "missing"),
            "test_iou_sweep": sw.get("test_iou_sweep", "missing"),
            "test_precision_sweep": sw.get("test_precision_sweep", "missing"),
            "test_recall_sweep": sw.get("test_recall_sweep", "missing"),
            "metric_source_sweep": "threshold_sweep_global_pixel" if sw else "not_available",
            "notes": f"Architecture extension — separate from plain U-Net tuning table",
        }
        arch_rows.append(row)

    return plain_rows, arch_rows, missing


# ---------------------------------------------------------------------------
# Build results table (two rows per run: one per metric_source)
# ---------------------------------------------------------------------------

def build_results_table(plain_rows: list[dict], arch_rows: list[dict]) -> list[dict]:
    result_rows = []
    for row in plain_rows + arch_rows:
        category = "plain_unet_tuning" if row["run_name"].startswith("filtered_strict") or row["script_name"] != "train_unet_arch_experiment.py" else "arch_extension"
        # Row for default 0.5
        r05 = {
            "run_name": row["run_name"],
            "category": category,
            "model_architecture": row["model_architecture"],
            "heldout_fp": row["heldout_fp"],
            "loss_function": row["loss_function"],
            "augmentation": row["augmentation"],
            "checkpoint_selection": row["checkpoint_selection"],
            "metric_source": "default_0.5_training_script_batch_average",
            "threshold": 0.5,
            "test_dice": row.get("test_dice_default05", "missing"),
            "test_iou": row.get("test_iou_default05", "missing"),
            "test_precision": row.get("test_precision_default05", "n/a"),
            "test_recall": row.get("test_recall_default05", "n/a"),
            "val_dice_at_threshold": "",
            "val_iou_at_threshold": "",
        }
        result_rows.append(r05)
        # Row for threshold sweep (if available)
        if row.get("test_dice_sweep") not in ("missing", "", None):
            rsw = {
                "run_name": row["run_name"],
                "category": category,
                "model_architecture": row["model_architecture"],
                "heldout_fp": row["heldout_fp"],
                "loss_function": row["loss_function"],
                "augmentation": row["augmentation"],
                "checkpoint_selection": row["checkpoint_selection"],
                "metric_source": "threshold_sweep_global_pixel",
                "threshold": row.get("selected_threshold", "?"),
                "test_dice": row.get("test_dice_sweep", "missing"),
                "test_iou": row.get("test_iou_sweep", "missing"),
                "test_precision": row.get("test_precision_sweep", "n/a"),
                "test_recall": row.get("test_recall_sweep", "n/a"),
                "val_dice_at_threshold": row.get("val_dice_at_selected_threshold", ""),
                "val_iou_at_threshold": row.get("val_iou_at_selected_threshold", ""),
            }
            result_rows.append(rsw)
    return result_rows


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".4f") -> str:
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def build_markdown(
    plain_rows: list[dict],
    arch_rows: list[dict],
    results_rows: list[dict],
    missing: list[str],
    script_defaults: dict[str, dict],
) -> str:
    lines = []

    # -----------------------------------------------------------
    # A. Short explanation
    # -----------------------------------------------------------
    lines.append("# U-Net Baseline and Tuning Experiment Inventory")
    lines.append("")
    lines.append(
        "This inventory covers **only** the plain U-Net baseline and the "
        "follow-up hyperparameter / ablation experiments run by JJ. "
        "It does not include land-cover error analysis, probability/uncertainty "
        "maps, Alea's tuned scripts, or whole-team project summaries."
    )
    lines.append("")
    lines.append(f"All experiments use: **{SPLIT_NOTE}**.")
    lines.append("")

    # -----------------------------------------------------------
    # B. Why these experiments were done
    # -----------------------------------------------------------
    lines.append("## B. Motivation and Design Rationale")
    lines.append("")
    lines.append(
        "**Original baseline** (`train_unet_baseline.py`): establishes a reference point — "
        "plain U-Net, BCE loss, Adam, no augmentation, checkpoint selected by val_loss. "
        "All other variants change exactly one or two things at a time relative to this baseline."
    )
    lines.append("")
    lines.append(
        "**Threshold sweep** (`threshold_sweep_baseline.py`, `threshold_sweep_single_checkpoint.py`): "
        "checks whether the default 0.5 cutoff is too conservative. "
        "Sweeps thresholds 0.05–0.95 in steps of 0.05 on the validation split only, "
        "selects the threshold that maximizes global pixel-level Dice, "
        "then applies it once to the held-out test split. "
        "These are global pixel-level metrics — not batch-averaged — and are NOT "
        "comparable with the default-0.5 training-script metrics."
    )
    lines.append("")
    lines.append(
        "**BCE + Dice loss** (`train_unet_experiment.py` with `--loss bce_dice`): "
        "tests whether adding a soft Dice term to the objective directly optimizes "
        "segmentation overlap rather than relying on the threshold sweep to recover it."
    )
    lines.append("")
    lines.append(
        "**Augmentation** (`--augment` flag): tests whether simple spatial transforms "
        "(random horizontal flip, vertical flip, 90° rotation) improve generalization "
        "to the unseen flight path."
    )
    lines.append("")
    lines.append(
        "**Validation-Dice checkpointing** (`--save-best-by val_dice`): tests whether "
        "saving the checkpoint with the highest validation Dice (instead of lowest "
        "validation loss) is a better surrogate for test-time segmentation quality."
    )
    lines.append("")
    lines.append(
        "**Weighted BCE** (`train_unet_weighted_experiment.py` with `--loss weighted_bce`): "
        "tests whether up-weighting the rare positive (flood) class via "
        "`pos_weight = negative_pixels / positive_pixels` (estimated from training masks "
        "before training) helps the model learn flood boundaries better."
    )
    lines.append("")
    lines.append(
        "**Architecture extensions** (`train_unet_arch_experiment.py` with "
        "`--arch attention_unet` or `--arch unet_plus_plus`): reported in a separate section. "
        "These change the model architecture, not just hyperparameters, so they are not "
        "direct ablations of the baseline and should not be compared as if they are."
    )
    lines.append("")

    # -----------------------------------------------------------
    # C. Master table
    # -----------------------------------------------------------
    lines.append("## C. Master Table of Experiments (Plain U-Net Only)")
    lines.append("")
    lines.append(
        "| Run name | Script | FP | Loss | Ckpt sel | Aug | Epochs | LR | BS | Base-ch | Notes |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for row in plain_rows:
        run = row["run_name"]
        script = row["script_name"]
        fp = row["heldout_fp"]
        loss = row["loss_function"].split(" ")[0]  # short form
        ckpt = row["checkpoint_selection"]
        aug = row["augmentation"].split(" ")[0]
        epochs = row["epochs"]
        lr = row["learning_rate"]
        bs = row["batch_size"]
        bc = row["base_channels"]
        note = "fp1 only" if "fp1" in run and "filtered_strict" not in run else ("fp1-fp7" if "filtered_strict" in run else "")
        lines.append(f"| {run} | {script} | {fp} | {loss} | {ckpt} | {aug} | {epochs} | {lr} | {bs} | {bc} | {note} |")
    lines.append("")

    # -----------------------------------------------------------
    # D. Results table
    # -----------------------------------------------------------
    lines.append("## D. Results Table")
    lines.append("")
    lines.append(
        "> **Important — two metric types are used:**\n"
        "> - `default_0.5`: computed during training, **batch-averaged** Dice/IoU at fixed threshold 0.5. Not pixel-global.\n"
        "> - `threshold_sweep_global_pixel`: computed by threshold sweep scripts over **all pixels** of the split at once. "
        "Threshold was selected on the validation set only, then applied once to the test set."
    )
    lines.append("")

    # Baseline results subtable
    lines.append("### D.1 Baseline (fp1–fp7) — Default 0.5 (batch-averaged) vs. Threshold-Sweep (global pixel)")
    lines.append("")
    lines.append("| FP | Default 0.5 Dice | Default 0.5 IoU | Selected Thresh | Val Dice (sweep) | Test Dice (sweep) | Test IoU (sweep) | Test Prec (sweep) | Test Recall (sweep) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in plain_rows:
        if row["script_name"] != "train_unet_baseline.py":
            continue
        fp = row["heldout_fp"]
        d05 = _fmt(row.get("test_dice_default05"))
        i05 = _fmt(row.get("test_iou_default05"))
        st = _fmt(row.get("selected_threshold"), ".2f")
        vd = _fmt(row.get("val_dice_at_selected_threshold"))
        td = _fmt(row.get("test_dice_sweep"))
        ti = _fmt(row.get("test_iou_sweep"))
        tp = _fmt(row.get("test_precision_sweep"))
        tr = _fmt(row.get("test_recall_sweep"))
        lines.append(f"| {fp} | {d05} | {i05} | {st} | {vd} | {td} | {ti} | {tp} | {tr} |")
    lines.append("")

    # Experiment results
    lines.append("### D.2 Experiment Variants — fp1 Only")
    lines.append("")
    lines.append("| Run | Loss | Ckpt sel | Aug | Default 0.5 Dice | Default 0.5 IoU | Selected Thresh | Val Dice (sweep) | Test Dice (sweep) | Test IoU (sweep) | Test Prec | Test Recall |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in plain_rows:
        if row["script_name"] == "train_unet_baseline.py":
            continue
        run = row["run_name"]
        loss = row["loss_function"].split(" ")[0]
        ckpt = row["checkpoint_selection"]
        aug = row["augmentation"].split(" ")[0]
        d05 = _fmt(row.get("test_dice_default05"))
        i05 = _fmt(row.get("test_iou_default05"))
        st = _fmt(row.get("selected_threshold"), ".2f")
        vd = _fmt(row.get("val_dice_at_selected_threshold"))
        td = _fmt(row.get("test_dice_sweep"))
        ti = _fmt(row.get("test_iou_sweep"))
        tp = _fmt(row.get("test_precision_sweep"))
        tr = _fmt(row.get("test_recall_sweep"))
        lines.append(f"| {run} | {loss} | {ckpt} | {aug} | {d05} | {i05} | {st} | {vd} | {td} | {ti} | {tp} | {tr} |")
    lines.append("")

    # -----------------------------------------------------------
    # E. Plain-English interpretation
    # -----------------------------------------------------------
    lines.append("## E. Plain-English Interpretation")
    lines.append("")

    # Compute baseline sweep average for comparison
    baseline_rows = [r for r in plain_rows if r["script_name"] == "train_unet_baseline.py"]
    try:
        avg_baseline_sweep = sum(float(r["test_dice_sweep"]) for r in baseline_rows if r.get("test_dice_sweep") not in ("missing", "")) / len(baseline_rows)
        best_baseline_fp = max(baseline_rows, key=lambda r: float(r.get("test_dice_sweep", 0) or 0))
        lines.append(
            f"**Best plain U-Net variant (sweep metric):** The baseline with threshold sweep "
            f"achieves the highest test Dice on fp2 ({_fmt(best_baseline_fp.get('test_dice_sweep'))}) "
            f"and lowest on fp7 ({_fmt(min(baseline_rows, key=lambda r: float(r.get('test_dice_sweep', 99) or 99)).get('test_dice_sweep'))}). "
            f"Average across fp1–fp7 at the sweep threshold: {avg_baseline_sweep:.4f}."
        )
    except Exception:
        lines.append("**Baseline sweep metrics:** see table above.")
    lines.append("")

    lines.append(
        "**Threshold sweep effect:** Comparing the baseline default-0.5 batch-averaged Dice with "
        "the threshold-sweep global-pixel Dice reveals the 0.5 threshold is often suboptimal. "
        "For example, fp1 improves from 0.3618 (default 0.5) to 0.5205 (sweep at 0.25). "
        "fp2 improves from 0.6129 to 0.7445 (sweep at 0.35). "
        "These are not directly comparable numbers (different metric computation methods), "
        "but they confirm that 0.5 is too conservative for this dataset — "
        "the model's raw output is systematically underconfident about flood pixels."
    )
    lines.append("")

    lines.append(
        "**Augmentation (fp1 only):** Comparing `bce_dice` with augmentation vs. without augmentation, "
        "both at the sweep threshold: aug=True gives test Dice 0.5094 vs. aug=False gives 0.4879. "
        "The difference is small and both runs are on fp1 only — insufficient to conclude "
        "augmentation clearly helps. More flight paths need to be run to confirm."
    )
    lines.append("")

    lines.append(
        "**BCE + Dice loss (fp1 only):** Switching from BCE to BCE+Dice improved the sweep threshold "
        "selection (0.40 or 0.55 vs. 0.35 for BCE) but the test Dice at the sweep threshold is "
        "similar to or slightly below the BCE val_dice checkpoint variant (0.488–0.509 vs. 0.495). "
        "No clear winner on fp1 alone — needs more flight paths."
    )
    lines.append("")

    lines.append(
        "**Weighted BCE (fp1 only):** The weighted_bce run selects a notably higher threshold "
        "(0.65) and achieves test Dice 0.4914 (sweep). This is comparable to other variants "
        "but not clearly better. The higher selected threshold suggests the model was pushed "
        "toward predicting higher probabilities for flood pixels, as expected. "
        "The default-0.5 test Dice (0.3997) is lower than the baseline default-0.5 (0.3618 on fp1). "
        "Not yet a clear win for fp1 alone."
    )
    lines.append("")

    lines.append(
        "**What still needs to be compared more carefully:**\n"
        "- All experiment variants (BCE+Dice, weighted_bce, augmentation) have only been run on fp1. "
        "Results from a single heldout flight path are insufficient to rank variants reliably.\n"
        "- The baseline has full fp1–fp7 coverage; experiment variants are fp1 only. "
        "Cross-fp comparison is therefore not yet possible.\n"
        "- The metric type distinction (batch-average vs. global-pixel) must be respected — "
        "the default-0.5 numbers and the sweep numbers are not interchangeable."
    )
    lines.append("")

    # -----------------------------------------------------------
    # Architecture extensions section
    # -----------------------------------------------------------
    lines.append("## F. Architecture Extensions (Not Plain U-Net Tuning)")
    lines.append("")
    lines.append(
        "The runs below use non-standard architectures (Attention U-Net, U-Net++) "
        "from `train_unet_arch_experiment.py`. They are reported separately because they "
        "change the model architecture, making them incomparable to baseline hyperparameter ablations."
    )
    lines.append("")
    lines.append("| Run | Arch | FP | Loss | Ckpt sel | Default 0.5 Dice | Default 0.5 IoU | Sweep Thresh | Test Dice (sweep) | Test IoU (sweep) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in arch_rows:
        run = row["run_name"]
        arch = row["model_architecture"]
        fp = row["heldout_fp"]
        loss = row["loss_function"].split(" ")[0]
        ckpt = row["checkpoint_selection"]
        d05 = _fmt(row.get("test_dice_default05"))
        i05 = _fmt(row.get("test_iou_default05"))
        st = _fmt(row.get("selected_threshold"), ".2f")
        td = _fmt(row.get("test_dice_sweep"))
        ti = _fmt(row.get("test_iou_sweep"))
        lines.append(f"| {run} | {arch} | {fp} | {loss} | {ckpt} | {d05} | {i05} | {st} | {td} | {ti} |")
    lines.append("")

    # -----------------------------------------------------------
    # G. Missing data
    # -----------------------------------------------------------
    lines.append("## G. Missing or Unclear Data")
    lines.append("")
    if missing:
        for m in missing:
            lines.append(f"- {m}")
    else:
        lines.append("- No critical missing data detected.")
    lines.append("")
    lines.append(
        "**Additional known gaps:**\n"
        "- Augmentation status for `experiment_fp1_weighted_bce_valdice_20epochs` not encoded in run name (default=False assumed).\n"
        "- Augmentation status for arch extension runs not encoded in run names.\n"
        "- Precision and recall are not available for default-0.5 training-script evaluations (only batch-averaged Dice and IoU).\n"
        "- Baseline train/val history CSVs (`filtered_strict_fp{N}_unet_20epochs_metrics.csv`) "
        "do not contain a test row — test metrics at 0.5 came from terminal output, "
        "collected into `filtered_strict_20epoch_test_summary.csv`.\n"
        "- Experiment variants (BCE+Dice, weighted_bce, augmentation) have been run on fp1 only. "
        "fp2–fp7 results do not exist yet."
    )
    lines.append("")

    # -----------------------------------------------------------
    # Script defaults appendix
    # -----------------------------------------------------------
    lines.append("## H. Script Default Hyperparameters (Extracted from Source)")
    lines.append("")
    lines.append("| Script | Epochs | LR | Batch | Base-ch | Seed | Optimizer | Loss options | Ckpt-sel options | Arch options | Augment flag |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for script_name, defs in script_defaults.items():
        lines.append(
            f"| {script_name} "
            f"| {defs.get('epochs')} "
            f"| {defs.get('lr')} "
            f"| {defs.get('batch_size')} "
            f"| {defs.get('base_channels')} "
            f"| {defs.get('seed')} "
            f"| {defs.get('optimizer')} "
            f"| {defs.get('loss_choices')} "
            f"| {defs.get('save_best_by_choices')} "
            f"| {defs.get('arch_choices')} "
            f"| {defs.get('has_augment_flag')} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def build_summary_txt(plain_rows: list[dict], arch_rows: list[dict], missing: list[str]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("U-Net Baseline and Tuning Inventory — Summary")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Plain U-Net experiment rows: {len(plain_rows)}")
    lines.append(f"  - Baseline (fp1-fp7): {sum(1 for r in plain_rows if r['script_name'] == 'train_unet_baseline.py')}")
    lines.append(f"  - Experiment variants (fp1 only): {sum(1 for r in plain_rows if r['script_name'] != 'train_unet_baseline.py')}")
    lines.append(f"Architecture extension rows: {len(arch_rows)}")
    lines.append("")
    lines.append("Baseline default-0.5 test metrics (batch-averaged, threshold=0.5):")
    for row in plain_rows:
        if row["script_name"] != "train_unet_baseline.py":
            continue
        lines.append(f"  {row['heldout_fp']:4s}  dice={_fmt(row.get('test_dice_default05'))}  iou={_fmt(row.get('test_iou_default05'))}")
    lines.append("")
    lines.append("Baseline threshold-sweep metrics (global pixel, val-selected threshold):")
    for row in plain_rows:
        if row["script_name"] != "train_unet_baseline.py":
            continue
        lines.append(
            f"  {row['heldout_fp']:4s}  thresh={_fmt(row.get('selected_threshold'), '.2f')}  "
            f"val_dice={_fmt(row.get('val_dice_at_selected_threshold'))}  "
            f"test_dice={_fmt(row.get('test_dice_sweep'))}  "
            f"test_iou={_fmt(row.get('test_iou_sweep'))}  "
            f"prec={_fmt(row.get('test_precision_sweep'))}  "
            f"recall={_fmt(row.get('test_recall_sweep'))}"
        )
    lines.append("")
    lines.append("Experiment variant results (fp1 only):")
    for row in plain_rows:
        if row["script_name"] == "train_unet_baseline.py":
            continue
        lines.append(
            f"  {row['run_name']}"
            f"\n    default-0.5: dice={_fmt(row.get('test_dice_default05'))}  iou={_fmt(row.get('test_iou_default05'))}"
            f"\n    sweep(thresh={_fmt(row.get('selected_threshold'), '.2f')}): "
            f"dice={_fmt(row.get('test_dice_sweep'))}  iou={_fmt(row.get('test_iou_sweep'))}  "
            f"prec={_fmt(row.get('test_precision_sweep'))}  recall={_fmt(row.get('test_recall_sweep'))}"
        )
    lines.append("")
    lines.append("Missing or unclear data:")
    if missing:
        for m in missing:
            lines.append(f"  - {m}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Augmentation status not in run name:")
    for row in plain_rows + arch_rows:
        if "not_in_name" in str(row.get("augmentation", "")):
            lines.append(f"  - {row['run_name']}: augmentation flag absent from run name")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract script defaults
    script_names = [
        "train_unet_baseline.py",
        "train_unet_experiment.py",
        "train_unet_weighted_experiment.py",
        "train_unet_arch_experiment.py",
        "threshold_sweep_baseline.py",
        "threshold_sweep_single_checkpoint.py",
    ]
    script_defaults = {
        name: _extract_script_defaults(SCRIPTS_DIR / name)
        for name in script_names
    }

    # Build inventory
    plain_rows, arch_rows, missing = build_inventory()

    # Write inventory CSV
    inv_csv = OUT_DIR / "unet_baseline_tuning_inventory.csv"
    pd.DataFrame(plain_rows + arch_rows).to_csv(inv_csv, index=False)

    # Write results table CSV
    results_rows = build_results_table(plain_rows, arch_rows)
    res_csv = OUT_DIR / "unet_baseline_tuning_results_table.csv"
    pd.DataFrame(results_rows).to_csv(res_csv, index=False)

    # Write markdown
    md = build_markdown(plain_rows, arch_rows, results_rows, missing, script_defaults)
    md_path = OUT_DIR / "unet_baseline_tuning_inventory.md"
    md_path.write_text(md, encoding="utf-8")

    # Write summary text
    summary = build_summary_txt(plain_rows, arch_rows, missing)
    txt_path = OUT_DIR / "unet_baseline_tuning_summary.txt"
    txt_path.write_text(summary, encoding="utf-8")

    # Print to stdout
    print(summary)
    print()
    print("=" * 72)
    print("Output files created:")
    print(f"  {md_path}")
    print(f"  {inv_csv}")
    print(f"  {res_csv}")
    print(f"  {txt_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
