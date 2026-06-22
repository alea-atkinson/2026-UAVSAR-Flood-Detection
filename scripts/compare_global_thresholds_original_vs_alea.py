#!/usr/bin/env python3
"""Global-threshold comparison: Original U-Net vs Alea-tuned U-Net.

Reads existing per-fold threshold sweep CSVs to find the best *single* global
threshold for each model (maximising mean validation Dice across fp1-fp7).
Then evaluates each model at that global threshold on every held-out test set.

Data sources (in priority order for test metrics at a given threshold):
  1. Per-fold sweep CSV already has test data at that row  → use directly.
  2. Original baseline fallback: threshold_sweep_selected_test_results.csv
     has test at the per-fold selected threshold → use if it matches.
  3. Checkpoint inference: load the checkpoint, run one forward pass over the
     test set, compute metrics.  Model is discarded after each fold so only
     one checkpoint lives in GPU memory at a time.

Outputs
-------
  outputs/global_threshold_comparison/global_threshold_summary.csv
  outputs/global_threshold_comparison/global_threshold_per_fp.csv
  outputs/global_threshold_comparison/global_threshold_comparison.md

Usage
-----
    python scripts/compare_global_thresholds_original_vs_alea.py [--device cuda]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import threshold_sweep_baseline as _tsb  # collect_probs, pixel_metrics
import train_unet_baseline as _tb        # UNet, FloodTileDataset (same arch for both models)

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FPS = list(range(1, 8))
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 8
NUM_WORKERS = 2

SPLIT_DIR = Path(
    "csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/strict_no_overlap"
)
DATA_ROOT = Path("2025_Tile_Data")
OUTPUT_DIR = Path("outputs/global_threshold_comparison")

# Reference numbers from the per-fold validation-selected threshold comparison
ORIGINAL_PERFOLD_MEAN_DICE = 0.5982
ALEA_PERFOLD_MEAN_DICE     = 0.6068

# Original U-Net: existing fallback CSVs written by threshold_sweep_baseline.py
ORIGINAL_FALLBACK_VAL_CSV = Path(
    "outputs/threshold_sweep_baseline/threshold_sweep_validation_by_fp.csv"
)
ORIGINAL_FALLBACK_SELECTED_CSV = Path(
    "outputs/threshold_sweep_baseline/threshold_sweep_selected_test_results.csv"
)

MODEL_CONFIGS = {
    "original_unet": {
        "label": "Original U-Net",
        "sweep_dir": Path("outputs/original_unet_filtered_strict/threshold_sweeps"),
        "fallback_val_csv": ORIGINAL_FALLBACK_VAL_CSV,
        "fallback_selected_csv": ORIGINAL_FALLBACK_SELECTED_CSV,
        "checkpoint_template": "models/filtered_strict_fp{fp}_unet_20epochs_best.pt",
        "perfold_mean_dice": ORIGINAL_PERFOLD_MEAN_DICE,
    },
    "alea_tuned": {
        "label": "Alea-tuned U-Net",
        "sweep_dir": Path("outputs/alea_tuned_filtered_strict/threshold_sweeps"),
        "fallback_val_csv": None,
        "fallback_selected_csv": None,
        "checkpoint_template": (
            "models/alea_tuned_filtered_strict_fp{fp}_focaldice_adamw_20epochs_best.pt"
        ),
        "perfold_mean_dice": ALEA_PERFOLD_MEAN_DICE,
    },
}


# ---------------------------------------------------------------------------
# Helpers: model loading and inference
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_channels = ck.get("args", {}).get("base_channels", 32)
    model = _tb.UNet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(ck["model_state_dict"])
    return model


def run_inference(
    checkpoint_path: Path,
    split_csv: Path,
    data_root: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Load checkpoint, run inference over split_csv, return (probs, targets)."""
    print(f"      Loading checkpoint: {checkpoint_path.name}")
    model = load_model(checkpoint_path, device)
    pin_memory = device.type == "cuda"
    dataset = _tb.FloodTileDataset(split_csv, data_root)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    print(f"      Inference over {len(dataset)} tiles ...")
    probs, targets = _tsb.collect_probs(model, loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, targets


def metrics_from_tsb(m: dict) -> dict:
    """Normalise pixel_metrics output: replace empty string with None."""
    return {
        "dice": float(m["dice"]),
        "iou": float(m["iou"]),
        "precision": float(m["precision"]) if m.get("precision") not in ("", None) else None,
        "recall": float(m["recall"]) if m.get("recall") not in ("", None) else None,
    }


# ---------------------------------------------------------------------------
# Helpers: reading val/test data from existing CSVs
# ---------------------------------------------------------------------------

def read_val_data(cfg: dict, fp: int) -> pd.DataFrame:
    """Return DataFrame with columns: threshold, val_dice, val_precision, val_recall."""
    per_fold = cfg["sweep_dir"] / f"fp{fp}_threshold_sweep.csv"
    if per_fold.exists():
        df = pd.read_csv(per_fold)
        keep = [c for c in ["threshold", "val_dice", "val_iou", "val_precision", "val_recall"] if c in df.columns]
        return df[keep].copy()

    fallback = cfg.get("fallback_val_csv")
    if fallback and Path(fallback).exists():
        df = pd.read_csv(fallback)
        fp_df = df[df["heldout_fp"] == f"fp{fp}"].copy()
        keep = [c for c in ["threshold", "val_dice", "val_iou", "val_precision", "val_recall"] if c in fp_df.columns]
        return fp_df[keep].reset_index(drop=True)

    raise FileNotFoundError(
        f"No val sweep data for fp{fp}.\n"
        f"  Looked for: {per_fold}\n"
        f"  Fallback:   {fallback}"
    )


def try_csv_test_lookup(cfg: dict, fp: int, threshold: float) -> dict | None:
    """
    Try to find test metrics at threshold from existing CSVs (no inference).
    Returns a metrics dict or None if not found.
    """
    # 1. Per-fold sweep CSV
    per_fold = cfg["sweep_dir"] / f"fp{fp}_threshold_sweep.csv"
    if per_fold.exists():
        df = pd.read_csv(per_fold)
        mask = (df["threshold"] - threshold).abs() < 1e-6
        if mask.any():
            r = df[mask].iloc[0]
            if pd.notna(r.get("test_dice")):
                return {
                    "dice": float(r["test_dice"]),
                    "iou": float(r["test_iou"]),
                    "precision": float(r["test_precision"]) if pd.notna(r.get("test_precision")) else None,
                    "recall": float(r["test_recall"]) if pd.notna(r.get("test_recall")) else None,
                }

    # 2. Original baseline selected CSV (test only at per-fold selected threshold)
    fallback_sel = cfg.get("fallback_selected_csv")
    if fallback_sel and Path(fallback_sel).exists():
        df = pd.read_csv(fallback_sel)
        mask = (df["heldout_fp"] == f"fp{fp}") & (
            (df["selected_threshold"] - threshold).abs() < 1e-6
        )
        if mask.any():
            r = df[mask].iloc[0]
            return {
                "dice": float(r["test_dice"]),
                "iou": float(r["test_iou"]),
                "precision": float(r["test_precision"]) if pd.notna(r.get("test_precision")) else None,
                "recall": float(r["test_recall"]) if pd.notna(r.get("test_recall")) else None,
            }

    return None


# ---------------------------------------------------------------------------
# Global threshold selection
# ---------------------------------------------------------------------------

def select_global_threshold(
    val_dfs: dict[int, pd.DataFrame],
) -> tuple[float, pd.DataFrame]:
    """
    Average val_dice per threshold across all folds.
    Return (best_threshold, summary_df with columns threshold, mean_val_dice, fp*_val_dice).
    """
    merged = None
    for fp, df in val_dfs.items():
        tmp = df[["threshold", "val_dice"]].rename(columns={"val_dice": f"fp{fp}_val_dice"})
        merged = tmp if merged is None else pd.merge(merged, tmp, on="threshold", how="inner")

    dice_cols = [f"fp{fp}_val_dice" for fp in FPS if f"fp{fp}_val_dice" in merged.columns]
    merged["mean_val_dice"] = merged[dice_cols].mean(axis=1)
    best_idx = merged["mean_val_dice"].idxmax()
    best_t = float(merged.loc[best_idx, "threshold"])
    return best_t, merged


# ---------------------------------------------------------------------------
# Per-fold analysis
# ---------------------------------------------------------------------------

def analyze_fold(
    cfg: dict,
    fp: int,
    global_threshold: float,
    device: torch.device,
) -> dict:
    """
    Get test metrics for fold fp at global_threshold and at DEFAULT_THRESHOLD (0.5).
    Tries CSV first; runs inference only when needed.
    """
    checkpoint = Path(cfg["checkpoint_template"].format(fp=fp))
    test_csv = SPLIT_DIR / f"heldout_fp{fp}_test.csv"

    thresholds = sorted({global_threshold, DEFAULT_THRESHOLD})
    results: dict[float, dict] = {}
    sources: dict[float, str] = {}

    need_inference: list[float] = []
    for t in thresholds:
        m = try_csv_test_lookup(cfg, fp, t)
        if m is not None:
            results[t] = m
            sources[t] = "csv"
        else:
            need_inference.append(t)

    if need_inference:
        probs, targets = run_inference(checkpoint, test_csv, DATA_ROOT, device)
        for t in need_inference:
            m = metrics_from_tsb(_tsb.pixel_metrics(probs, targets, t))
            results[t] = m
            sources[t] = "inference"
        del probs, targets

    def _r(t: float) -> dict:
        m = results[t]
        return {
            "dice": m["dice"],
            "iou": m["iou"],
            "precision": m["precision"],
            "recall": m["recall"],
            "source": sources[t],
        }

    return {"global": _r(global_threshold), "default": _r(DEFAULT_THRESHOLD)}


# ---------------------------------------------------------------------------
# Model analysis
# ---------------------------------------------------------------------------

def analyze_model(cfg: dict, device: torch.device) -> dict:
    label = cfg["label"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # --- Validation data ---
    print("  Reading validation sweep data ...")
    val_dfs: dict[int, pd.DataFrame] = {}
    for fp in FPS:
        print(f"    fp{fp} ...", end=" ", flush=True)
        val_dfs[fp] = read_val_data(cfg, fp)
        print(f"{len(val_dfs[fp])} threshold rows")

    global_t, val_summary = select_global_threshold(val_dfs)
    best_mean_val = float(val_summary.loc[
        (val_summary["threshold"] - global_t).abs() < 1e-6, "mean_val_dice"
    ].iloc[0])
    print(f"\n  Selected global threshold: {global_t:.2f}  "
          f"(mean val Dice = {best_mean_val:.4f})")

    # --- Test metrics ---
    per_fold: list[dict] = []
    for fp in FPS:
        print(f"\n  [fp{fp}] Test metrics at global={global_t:.2f} and default=0.50 ...")
        fold_res = analyze_fold(cfg, fp, global_t, device)
        g = fold_res["global"]
        d = fold_res["default"]
        print(
            f"    global({global_t:.2f}): dice={g['dice']:.4f}  iou={g['iou']:.4f}  "
            f"prec={_fmt(g['precision'])}  rec={_fmt(g['recall'])}  [{g['source']}]"
        )
        print(
            f"    default(0.50): dice={d['dice']:.4f}  iou={d['iou']:.4f}  "
            f"prec={_fmt(d['precision'])}  rec={_fmt(d['recall'])}  [{d['source']}]"
        )
        per_fold.append({
            "heldout_fp": f"fp{fp}",
            "global_threshold": global_t,
            "test_dice_global": g["dice"],
            "test_iou_global": g["iou"],
            "test_precision_global": g["precision"],
            "test_recall_global": g["recall"],
            "source_global": g["source"],
            "test_dice_0.5": d["dice"],
            "test_iou_0.5": d["iou"],
            "test_precision_0.5": d["precision"],
            "test_recall_0.5": d["recall"],
            "source_0.5": d["source"],
        })

    def _nanmean(vals: list) -> float | None:
        clean = [v for v in vals if v is not None]
        return float(np.mean(clean)) if clean else None

    g_dices = [r["test_dice_global"] for r in per_fold]
    g_ious = [r["test_iou_global"] for r in per_fold]
    g_precs = [r["test_precision_global"] for r in per_fold]
    g_recs  = [r["test_recall_global"]  for r in per_fold]
    d_dices = [r["test_dice_0.5"] for r in per_fold]
    d_ious  = [r["test_iou_0.5"] for r in per_fold]
    d_precs = [r["test_precision_0.5"] for r in per_fold]
    d_recs  = [r["test_recall_0.5"]  for r in per_fold]

    return {
        "label": label,
        "perfold_mean_dice": cfg["perfold_mean_dice"],
        "global_threshold": global_t,
        "mean_val_dice_at_global": best_mean_val,
        "mean_test_dice_global": float(np.mean(g_dices)),
        "mean_test_iou_global": float(np.mean(g_ious)),
        "mean_test_precision_global": _nanmean(g_precs),
        "mean_test_recall_global": _nanmean(g_recs),
        "mean_test_dice_0.5": float(np.mean(d_dices)),
        "mean_test_iou_0.5": float(np.mean(d_ious)),
        "mean_test_precision_0.5": _nanmean(d_precs),
        "mean_test_recall_0.5": _nanmean(d_recs),
        "per_fold": per_fold,
        "val_summary": val_summary,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _fmt(v: float | None, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else ""


def write_summary_csv(results: list[dict], path: Path) -> None:
    fieldnames = [
        "model",
        "global_threshold",
        "mean_val_dice_at_global_threshold",
        "mean_test_dice_global", "mean_test_iou_global",
        "mean_test_precision_global", "mean_test_recall_global",
        "mean_test_dice_at_0.5", "mean_test_iou_at_0.5",
        "mean_test_precision_at_0.5", "mean_test_recall_at_0.5",
        "per_fold_mean_test_dice_reference",
        "global_vs_perfold_dice_delta",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            delta = r["mean_test_dice_global"] - r["perfold_mean_dice"]
            writer.writerow({
                "model": r["label"],
                "global_threshold": r["global_threshold"],
                "mean_val_dice_at_global_threshold": _fmt(r["mean_val_dice_at_global"]),
                "mean_test_dice_global": _fmt(r["mean_test_dice_global"]),
                "mean_test_iou_global": _fmt(r["mean_test_iou_global"]),
                "mean_test_precision_global": _fmt(r["mean_test_precision_global"]),
                "mean_test_recall_global": _fmt(r["mean_test_recall_global"]),
                "mean_test_dice_at_0.5": _fmt(r["mean_test_dice_0.5"]),
                "mean_test_iou_at_0.5": _fmt(r["mean_test_iou_0.5"]),
                "mean_test_precision_at_0.5": _fmt(r["mean_test_precision_0.5"]),
                "mean_test_recall_at_0.5": _fmt(r["mean_test_recall_0.5"]),
                "per_fold_mean_test_dice_reference": r["perfold_mean_dice"],
                "global_vs_perfold_dice_delta": _fmt(delta),
            })


def write_per_fp_csv(results: list[dict], path: Path) -> None:
    fieldnames = [
        "model", "heldout_fp", "global_threshold",
        "test_dice", "test_iou", "test_precision", "test_recall",
        "test_dice_at_0.5", "test_iou_at_0.5",
        "test_precision_at_0.5", "test_recall_at_0.5",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            for fold in res["per_fold"]:
                writer.writerow({
                    "model": res["label"],
                    "heldout_fp": fold["heldout_fp"],
                    "global_threshold": fold["global_threshold"],
                    "test_dice": _fmt(fold["test_dice_global"]),
                    "test_iou": _fmt(fold["test_iou_global"]),
                    "test_precision": _fmt(fold["test_precision_global"]),
                    "test_recall": _fmt(fold["test_recall_global"]),
                    "test_dice_at_0.5": _fmt(fold["test_dice_0.5"]),
                    "test_iou_at_0.5": _fmt(fold["test_iou_0.5"]),
                    "test_precision_at_0.5": _fmt(fold["test_precision_0.5"]),
                    "test_recall_at_0.5": _fmt(fold["test_recall_0.5"]),
                    "source": fold["source_global"],
                })


def write_markdown(results: list[dict], path: Path) -> None:
    orig = results[0]
    alea = results[1]
    o_gt = orig["global_threshold"]
    a_gt = alea["global_threshold"]
    winner_global = (
        alea["label"] if alea["mean_test_dice_global"] >= orig["mean_test_dice_global"]
        else orig["label"]
    )
    alea_still_wins = alea["mean_test_dice_global"] >= orig["mean_test_dice_global"]
    orig_delta = orig["mean_test_dice_global"] - orig["perfold_mean_dice"]
    alea_delta  = alea["mean_test_dice_global"]  - alea["perfold_mean_dice"]

    def _05_note(r: dict) -> str:
        d = abs(r["mean_test_dice_0.5"] - r["mean_test_dice_global"])
        sign = "+" if r["mean_test_dice_0.5"] >= r["mean_test_dice_global"] else "-"
        return f"{sign}{d:.4f} vs global threshold"

    lines = [
        "# Global Threshold Comparison: Original U-Net vs Alea-tuned U-Net",
        "",
        "Splits: IEEE PNG-filtered strict no-overlap leave-one-flight-path-out (fp1–fp7).",
        "Global threshold = the single threshold with the highest **mean validation Dice**",
        "across all 7 held-out flight paths.",
        "",
        "---",
        "",
        "## Selected Global Thresholds",
        "",
        f"| Model | Global Threshold | Mean Val Dice at Global Threshold |",
        f"|-------|-----------------|-----------------------------------|",
        f"| {orig['label']} | **{o_gt:.2f}** | {orig['mean_val_dice_at_global']:.4f} |",
        f"| {alea['label']} | **{a_gt:.2f}** | {alea['mean_val_dice_at_global']:.4f} |",
        "",
        "---",
        "",
        "## Test Results at Global Threshold",
        "",
        "| Model | Mean Dice | Mean IoU | Mean Precision | Mean Recall |",
        "|-------|-----------|----------|----------------|-------------|",
        f"| {orig['label']} (threshold={o_gt:.2f}) "
        f"| {orig['mean_test_dice_global']:.4f} "
        f"| {orig['mean_test_iou_global']:.4f} "
        f"| {_fmt(orig['mean_test_precision_global'])} "
        f"| {_fmt(orig['mean_test_recall_global'])} |",
        f"| {alea['label']} (threshold={a_gt:.2f}) "
        f"| {alea['mean_test_dice_global']:.4f} "
        f"| {alea['mean_test_iou_global']:.4f} "
        f"| {_fmt(alea['mean_test_precision_global'])} "
        f"| {_fmt(alea['mean_test_recall_global'])} |",
        "",
        "---",
        "",
        "## Per-Fold Test Dice at Global Threshold",
        "",
        "| Flight Path | Original U-Net | Alea-tuned U-Net |",
        "|-------------|----------------|------------------|",
    ]
    orig_folds = {r["heldout_fp"]: r for r in orig["per_fold"]}
    alea_folds  = {r["heldout_fp"]: r for r in alea["per_fold"]}
    for fp in [f"fp{i}" for i in FPS]:
        o = orig_folds[fp]
        a = alea_folds[fp]
        lines.append(
            f"| {fp} "
            f"| {o['test_dice_global']:.4f} (prec={_fmt(o['test_precision_global'])}, rec={_fmt(o['test_recall_global'])}) "
            f"| {a['test_dice_global']:.4f} (prec={_fmt(a['test_precision_global'])}, rec={_fmt(a['test_recall_global'])}) |"
        )

    lines += [
        "",
        "---",
        "",
        "## Default Threshold 0.5 Results",
        "",
        "| Model | Mean Dice | Mean IoU | Mean Precision | Mean Recall |",
        "|-------|-----------|----------|----------------|-------------|",
        f"| {orig['label']} (threshold=0.50) "
        f"| {orig['mean_test_dice_0.5']:.4f} "
        f"| {orig['mean_test_iou_0.5']:.4f} "
        f"| {_fmt(orig['mean_test_precision_0.5'])} "
        f"| {_fmt(orig['mean_test_recall_0.5'])} |",
        f"| {alea['label']} (threshold=0.50) "
        f"| {alea['mean_test_dice_0.5']:.4f} "
        f"| {alea['mean_test_iou_0.5']:.4f} "
        f"| {_fmt(alea['mean_test_precision_0.5'])} "
        f"| {_fmt(alea['mean_test_recall_0.5'])} |",
        "",
        "---",
        "",
        "## Summary: Key Questions",
        "",
        f"**Q1: What global threshold was selected for Original U-Net?**",
        f"> {o_gt:.2f}",
        "",
        f"**Q2: What global threshold was selected for Alea-tuned U-Net?**",
        f"> {a_gt:.2f}",
        "",
        f"**Q3: Which model wins under global-threshold evaluation by mean test Dice?**",
        f"> **{winner_global}**  "
        f"({orig['mean_test_dice_global']:.4f} original vs {alea['mean_test_dice_global']:.4f} Alea)",
        "",
        f"**Q4: Does Alea still improve over Original under global threshold?**",
        f"> {'Yes' if alea_still_wins else 'No'}. "
        f"Alea mean Dice = {alea['mean_test_dice_global']:.4f}, "
        f"Original = {orig['mean_test_dice_global']:.4f} "
        f"(delta = {alea['mean_test_dice_global'] - orig['mean_test_dice_global']:+.4f}).",
        "",
        f"**Q5: How much performance is lost vs per-fold threshold selection?**",
        f"> Original: global ({o_gt:.2f}) vs per-fold → "
        f"mean Dice {orig['mean_test_dice_global']:.4f} vs {orig['perfold_mean_dice']:.4f} "
        f"({orig_delta:+.4f}).",
        f"> Alea: global ({a_gt:.2f}) vs per-fold → "
        f"mean Dice {alea['mean_test_dice_global']:.4f} vs {alea['perfold_mean_dice']:.4f} "
        f"({alea_delta:+.4f}).",
        "",
        f"**Q6: Is default threshold 0.5 close to the global optimum?**",
        f"> Original: 0.5 gives mean Dice {orig['mean_test_dice_0.5']:.4f} ({_05_note(orig)}). "
        f"Global optimum is {o_gt:.2f}.{'  Default is at global optimum.' if o_gt == 0.5 else ''}",
        f"> Alea: 0.5 gives mean Dice {alea['mean_test_dice_0.5']:.4f} ({_05_note(alea)}). "
        f"Global optimum is {a_gt:.2f}.{'  Default is at global optimum.' if a_gt == 0.5 else ''}",
        "",
        "---",
        "",
        "_Generated by `scripts/compare_global_thresholds_original_vs_alea.py`_",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global-threshold comparison: Original U-Net vs Alea-tuned U-Net."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root directory for tile data (default: 2025_Tile_Data)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Check Alea sweep CSVs exist
    alea_sweep_dir = MODEL_CONFIGS["alea_tuned"]["sweep_dir"]
    missing_alea = [
        alea_sweep_dir / f"fp{fp}_threshold_sweep.csv"
        for fp in FPS
        if not (alea_sweep_dir / f"fp{fp}_threshold_sweep.csv").exists()
    ]
    if missing_alea:
        print("[ERROR] Missing Alea threshold sweep CSVs:")
        for p in missing_alea:
            print(f"  {p}")
        sys.exit(1)

    # Check original val data
    orig_sweep_dir = MODEL_CONFIGS["original_unet"]["sweep_dir"]
    orig_has_per_fold = all(
        (orig_sweep_dir / f"fp{fp}_threshold_sweep.csv").exists() for fp in FPS
    )
    if not orig_has_per_fold and not ORIGINAL_FALLBACK_VAL_CSV.exists():
        print("[ERROR] No original U-Net val sweep data found.")
        print(f"  Neither {orig_sweep_dir} per-fold CSVs nor {ORIGINAL_FALLBACK_VAL_CSV} exist.")
        print("  Run the shell script first:")
        print("    bash scripts/run_original_unet_filtered_strict_threshold_sweeps.sh")
        sys.exit(1)

    if not orig_has_per_fold:
        print(
            f"[INFO] Original per-fold sweep CSVs not found at {orig_sweep_dir}.\n"
            f"       Falling back to {ORIGINAL_FALLBACK_VAL_CSV} for val data.\n"
            f"       Test metrics at the global threshold will be computed via inference.\n"
        )

    results: list[dict] = []
    for key in ["original_unet", "alea_tuned"]:
        cfg = MODEL_CONFIGS[key]
        result = analyze_model(cfg, device)
        results.append(result)

    # Write outputs
    print(f"\n\nWriting outputs to {OUTPUT_DIR} ...")
    summary_path  = OUTPUT_DIR / "global_threshold_summary.csv"
    per_fp_path   = OUTPUT_DIR / "global_threshold_per_fp.csv"
    markdown_path = OUTPUT_DIR / "global_threshold_comparison.md"

    write_summary_csv(results, summary_path)
    write_per_fp_csv(results, per_fp_path)
    write_markdown(results, markdown_path)

    print(f"  Wrote: {summary_path}")
    print(f"  Wrote: {per_fp_path}")
    print(f"  Wrote: {markdown_path}")

    # --- Terminal summary ---
    print("\n" + "=" * 70)
    print("GLOBAL THRESHOLD COMPARISON SUMMARY")
    print("=" * 70)

    orig = results[0]
    alea = results[1]

    print(f"\n{'Model':<30} {'Global T':>8} {'Mean Dice':>10} {'Mean IoU':>10} "
          f"{'Mean Prec':>10} {'Mean Rec':>10}")
    print("-" * 82)
    for r in results:
        gt = r["global_threshold"]
        print(
            f"  {r['label']:<28} {gt:>8.2f}"
            f" {r['mean_test_dice_global']:>10.4f}"
            f" {r['mean_test_iou_global']:>10.4f}"
            f" {_fmt(r['mean_test_precision_global']):>10}"
            f" {_fmt(r['mean_test_recall_global']):>10}"
        )

    print(f"\n  Default threshold 0.5:")
    for r in results:
        print(
            f"    {r['label']:<28}"
            f"  dice={r['mean_test_dice_0.5']:.4f}"
            f"  iou={r['mean_test_iou_0.5']:.4f}"
            f"  prec={_fmt(r['mean_test_precision_0.5'])}"
            f"  rec={_fmt(r['mean_test_recall_0.5'])}"
        )

    print(f"\n  Per-fold threshold selection (reference):")
    for r in results:
        print(f"    {r['label']:<28}  mean dice={r['perfold_mean_dice']:.4f}")

    alea_global_wins = alea["mean_test_dice_global"] >= orig["mean_test_dice_global"]
    orig_delta = orig["mean_test_dice_global"] - orig["perfold_mean_dice"]
    alea_delta = alea["mean_test_dice_global"] - alea["perfold_mean_dice"]

    print(f"\n  Alea vs Original under global threshold: "
          f"{'Alea wins' if alea_global_wins else 'Original wins'} "
          f"({alea['mean_test_dice_global']:.4f} vs {orig['mean_test_dice_global']:.4f})")
    print(f"  Performance vs per-fold:  Original {orig_delta:+.4f},  Alea {alea_delta:+.4f}")

    # --- Bash commands ---
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    if not orig_has_per_fold:
        print("""
If you want standalone per-fold sweep CSVs for the original U-Net
(test metrics at every threshold, reproducible without reloading checkpoints):

    cd /mnt/linuxlab/home/reujsalter/reu_flood_project/2026-UAVSAR-Flood-Detection
    bash scripts/run_original_unet_filtered_strict_threshold_sweeps.sh

Then re-run this comparison (it will use the generated CSVs instead of
running inference):

    python scripts/compare_global_thresholds_original_vs_alea.py
""")
    else:
        print("""
All sweep CSVs were found. To re-run this comparison at any time:

    cd /mnt/linuxlab/home/reujsalter/reu_flood_project/2026-UAVSAR-Flood-Detection
    python scripts/compare_global_thresholds_original_vs_alea.py
""")

    print(f"""View the markdown report:

    cat {markdown_path}

Or the summary CSV:

    column -t -s, {summary_path}
""")


if __name__ == "__main__":
    main()
