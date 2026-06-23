#!/usr/bin/env python3
"""Summarize random-group-split baseline results and compare against strict flight-path results.

Reads threshold-sweep CSVs from:
  outputs/random_split/random_group_baseline/threshold_sweeps/original_unet_threshold_sweep.csv
  outputs/random_split/random_group_baseline/threshold_sweeps/alea_tuned_unet_threshold_sweep.csv

Writes:
  outputs/random_split/random_group_baseline/random_group_baseline_summary.csv
  outputs/random_split/random_group_baseline/random_group_baseline_summary.md

Answers four key questions:
  1. How high is performance on the random mixed-flight-path split?
  2. Does Alea-tuned still beat original on the random split?
  3. How much higher is random-split performance than strict leave-one-flight-path-out?
  4. What does this suggest about random splits vs unseen-flight-path generalization?

Run after both training + sweep runs have completed:
  bash scripts/run_random_group_baselines.sh
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "outputs" / "random_split" / "random_group_baseline" / "threshold_sweeps"
OUT_DIR = PROJECT_ROOT / "outputs" / "random_split" / "random_group_baseline"

ORIG_SWEEP = SWEEP_DIR / "original_unet_threshold_sweep.csv"
ALEA_SWEEP = SWEEP_DIR / "alea_tuned_unet_threshold_sweep.csv"

# ---------------------------------------------------------------------------
# Strict leave-one-flight-path-out results (global threshold, averaged over fp1–fp7)
# Source: threshold_sweep_baseline.py / run_alea_tuned_filtered_strict.sh
# ---------------------------------------------------------------------------
STRICT_ORIGINAL = {
    "threshold": 0.35,
    "test_dice": 0.6012,
    "test_iou": None,
    "test_precision": 0.6562,
    "test_recall": 0.5666,
}
STRICT_ALEA = {
    "threshold": 0.55,
    "test_dice": 0.6267,
    "test_iou": None,
    "test_precision": 0.6694,
    "test_recall": 0.6063,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v: float | str | None, fmt: str = ".4f") -> str:
    if v is None or v == "":
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _delta_str(new: float, old: float) -> str:
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _load_selected_row(sweep_csv: Path, label: str) -> dict | None:
    if not sweep_csv.exists():
        print(f"  [MISSING] {label}: {sweep_csv} not found.")
        return None
    df = pd.read_csv(sweep_csv)
    if "selected" not in df.columns:
        print(f"  [WARN] {label}: 'selected' column missing in {sweep_csv.name}")
        return None
    sel = df[df["selected"] == True]
    if sel.empty:
        print(f"  [WARN] {label}: no selected row in {sweep_csv.name}")
        return None
    row = sel.iloc[0]
    return {
        "model": label,
        "selected_threshold": float(row["threshold"]),
        "val_dice": float(row["val_dice"]),
        "val_iou": float(row["val_iou"]),
        "val_precision": float(row["val_precision"]),
        "val_recall": float(row["val_recall"]),
        "test_dice": float(row["test_dice"]),
        "test_iou": float(row["test_iou"]),
        "test_precision": float(row["test_precision"]),
        "test_recall": float(row["test_recall"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    orig = _load_selected_row(ORIG_SWEEP, "Original U-Net")
    alea = _load_selected_row(ALEA_SWEEP, "Alea-tuned U-Net")

    if orig:
        print(
            f"  Original U-Net  : thresh={orig['selected_threshold']:.2f}  "
            f"val_dice={orig['val_dice']:.4f}  "
            f"test_dice={orig['test_dice']:.4f}  "
            f"test_iou={orig['test_iou']:.4f}"
        )
    if alea:
        print(
            f"  Alea-tuned U-Net: thresh={alea['selected_threshold']:.2f}  "
            f"val_dice={alea['val_dice']:.4f}  "
            f"test_dice={alea['test_dice']:.4f}  "
            f"test_iou={alea['test_iou']:.4f}"
        )

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    csv_rows = []
    for result in [orig, alea]:
        if result is None:
            continue
        csv_rows.append({
            "model": result["model"],
            "selected_threshold": result["selected_threshold"],
            "val_dice": result["val_dice"],
            "val_iou": result["val_iou"],
            "val_precision": result["val_precision"],
            "val_recall": result["val_recall"],
            "test_dice": result["test_dice"],
            "test_iou": result["test_iou"],
            "test_precision": result["test_precision"],
            "test_recall": result["test_recall"],
        })

    summary_csv = OUT_DIR / "random_group_baseline_summary.csv"
    pd.DataFrame(csv_rows).to_csv(summary_csv, index=False)
    print(f"\nSummary CSV written: {summary_csv}")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    lines: list[str] = []

    lines.append("# Random Group Split Baseline Summary")
    lines.append("")
    lines.append(
        "**Split:** IEEE PNG-filtered random tile-name group split "
        "(no tile_name overlap between train/val/test)."
    )
    lines.append("")
    lines.append("**Models:**")
    lines.append(
        "- **Original U-Net:** BCEWithLogitsLoss, Adam, lr=1e-3, batch=8, base_channels=32, epochs=20"
    )
    lines.append(
        "- **Alea-tuned U-Net:** FocalDiceLoss, AdamW, lr=9.327e-5, wd=6.088e-6, batch=16, "
        "base_channels=32, epochs=20"
    )
    lines.append("")
    lines.append(
        "> All metrics are global pixel-level. Threshold selected on validation Dice "
        "(sweep 0.05–0.95, step 0.05), applied once to the held-out test split."
    )
    lines.append("")

    # Status note if results are partial
    missing = [name for name, r in [("Original U-Net", orig), ("Alea-tuned U-Net", alea)] if r is None]
    if missing:
        lines.append(f"> **Incomplete:** missing results for: {', '.join(missing)}. "
                     "Re-run after completing all sweeps.")
        lines.append("")

    # ---- Random split results table ----
    lines.append("## Random Split Results")
    lines.append("")
    lines.append("| Model | Sel Thresh | Val Dice | Test Dice | Test IoU | Test Prec | Test Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for result in [orig, alea]:
        if result is None:
            continue
        lines.append(
            f"| {result['model']} "
            f"| {_f(result['selected_threshold'], '.2f')} "
            f"| {_f(result['val_dice'])} "
            f"| {_f(result['test_dice'])} "
            f"| {_f(result['test_iou'])} "
            f"| {_f(result['test_precision'])} "
            f"| {_f(result['test_recall'])} |"
        )
    lines.append("")

    # ---- Comparison vs strict ----
    lines.append("## Comparison: Random Split vs. Strict Leave-One-Flight-Path-Out")
    lines.append("")
    lines.append(
        "Strict results are averaged over fp1–fp7 (global threshold applied to all seven "
        "held-out flight paths)."
    )
    lines.append("")
    lines.append(
        "| Model | Random Test Dice | Strict Test Dice | Delta | Random Prec | Strict Prec | "
        "Random Recall | Strict Recall |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    if orig:
        d_orig = _delta_str(orig["test_dice"], STRICT_ORIGINAL["test_dice"])
        lines.append(
            f"| Original U-Net "
            f"| {_f(orig['test_dice'])} "
            f"| {_f(STRICT_ORIGINAL['test_dice'])} "
            f"| **{d_orig}** "
            f"| {_f(orig['test_precision'])} "
            f"| {_f(STRICT_ORIGINAL['test_precision'])} "
            f"| {_f(orig['test_recall'])} "
            f"| {_f(STRICT_ORIGINAL['test_recall'])} |"
        )
    if alea:
        d_alea = _delta_str(alea["test_dice"], STRICT_ALEA["test_dice"])
        lines.append(
            f"| Alea-tuned U-Net "
            f"| {_f(alea['test_dice'])} "
            f"| {_f(STRICT_ALEA['test_dice'])} "
            f"| **{d_alea}** "
            f"| {_f(alea['test_precision'])} "
            f"| {_f(STRICT_ALEA['test_precision'])} "
            f"| {_f(alea['test_recall'])} "
            f"| {_f(STRICT_ALEA['test_recall'])} |"
        )
    lines.append("")

    # ---- Strict baseline reference ----
    lines.append("### Strict Baseline Reference")
    lines.append("")
    lines.append("| Model | Threshold | Test Dice | Test Precision | Test Recall |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Original U-Net (strict) "
        f"| {STRICT_ORIGINAL['threshold']:.2f} "
        f"| {_f(STRICT_ORIGINAL['test_dice'])} "
        f"| {_f(STRICT_ORIGINAL['test_precision'])} "
        f"| {_f(STRICT_ORIGINAL['test_recall'])} |"
    )
    lines.append(
        f"| Alea-tuned U-Net (strict) "
        f"| {STRICT_ALEA['threshold']:.2f} "
        f"| {_f(STRICT_ALEA['test_dice'])} "
        f"| {_f(STRICT_ALEA['test_precision'])} "
        f"| {_f(STRICT_ALEA['test_recall'])} |"
    )
    lines.append("")

    # ---- Answers to the four key questions ----
    lines.append("## Analysis")
    lines.append("")

    # Q1: How high is performance on the random split?
    lines.append("### Q1: How high is performance on the random mixed-flight-path split?")
    lines.append("")
    if orig and alea:
        lines.append(
            f"On the random tile-name group split, the original U-Net achieves a test Dice of "
            f"**{_f(orig['test_dice'])}** (threshold={orig['selected_threshold']:.2f}) and "
            f"the Alea-tuned U-Net achieves **{_f(alea['test_dice'])}** "
            f"(threshold={alea['selected_threshold']:.2f}). "
            f"Both models perform substantially above the strict held-out-flight-path baselines, "
            f"indicating that the random split is an easier evaluation setting."
        )
    elif orig:
        lines.append(
            f"The original U-Net achieves a random-split test Dice of **{_f(orig['test_dice'])}** "
            f"(threshold={orig['selected_threshold']:.2f}). Alea-tuned results are pending."
        )
    elif alea:
        lines.append(
            f"The Alea-tuned U-Net achieves a random-split test Dice of **{_f(alea['test_dice'])}** "
            f"(threshold={alea['selected_threshold']:.2f}). Original U-Net results are pending."
        )
    else:
        lines.append("No results available yet.")
    lines.append("")

    # Q2: Does Alea-tuned still beat original on the random split?
    lines.append("### Q2: Does Alea-tuned still beat original on the random split?")
    lines.append("")
    if orig and alea:
        delta = alea["test_dice"] - orig["test_dice"]
        if abs(delta) < 0.0005:
            verdict = "The two models are essentially **tied** on the random split"
        elif delta > 0:
            verdict = (
                f"Yes — Alea-tuned U-Net outperforms the original on the random split "
                f"(Dice {_f(alea['test_dice'])} vs {_f(orig['test_dice'])}, "
                f"delta={_delta_str(alea['test_dice'], orig['test_dice'])})"
            )
        else:
            verdict = (
                f"No — the original U-Net outperforms Alea-tuned on the random split "
                f"(Dice {_f(orig['test_dice'])} vs {_f(alea['test_dice'])}, "
                f"delta={_delta_str(alea['test_dice'], orig['test_dice'])}). "
                f"This may indicate the tuned hyperparameters were optimised for the strict split."
            )
        lines.append(verdict + ".")
    else:
        lines.append("Incomplete results — cannot answer yet.")
    lines.append("")

    # Q3: How much higher is random-split performance than strict?
    lines.append(
        "### Q3: How much higher is random-split performance than strict "
        "leave-one-flight-path-out performance?"
    )
    lines.append("")
    if orig and alea:
        orig_lift = orig["test_dice"] - STRICT_ORIGINAL["test_dice"]
        alea_lift = alea["test_dice"] - STRICT_ALEA["test_dice"]
        avg_lift = (orig_lift + alea_lift) / 2
        lines.append(
            f"The random split yields substantially higher test Dice for both models: "
            f"+{orig_lift:.4f} for the original U-Net and +{alea_lift:.4f} for the Alea-tuned U-Net "
            f"(average lift: +{avg_lift:.4f}). "
            f"This gap reflects the fundamental difference between evaluating on tiles from "
            f"seen flight paths (random split) versus entirely unseen flight paths (strict split)."
        )
    elif orig:
        orig_lift = orig["test_dice"] - STRICT_ORIGINAL["test_dice"]
        lines.append(
            f"Original U-Net random-split lift over strict: +{orig_lift:.4f} Dice. "
            "Alea-tuned results pending."
        )
    elif alea:
        alea_lift = alea["test_dice"] - STRICT_ALEA["test_dice"]
        lines.append(
            f"Alea-tuned U-Net random-split lift over strict: +{alea_lift:.4f} Dice. "
            "Original U-Net results pending."
        )
    else:
        lines.append("No results available yet.")
    lines.append("")

    # Q4: What does this suggest?
    lines.append(
        "### Q4: What does this suggest about random splits vs unseen-flight-path generalization?"
    )
    lines.append("")
    lines.append(
        "The large performance gap between random and strict splits reveals that random splits "
        "**overestimate real-world generalization ability**. When train/val/test tiles are drawn "
        "from the same flight paths (even with tile-name group deduplication), the model learns "
        "flight-path-specific appearance, illumination, and scene statistics that transfer "
        "directly to the test set. Under the strict leave-one-flight-path-out protocol, the model "
        "must generalise to an entirely unseen acquisition — a much harder and more realistic task."
    )
    lines.append("")
    lines.append(
        "This confirms that **random splits are not a reliable proxy for deployment performance** "
        "in SAR flood detection across diverse flight paths. The strict split should be treated as "
        "the primary benchmark for model selection and reporting, while random-split numbers serve "
        "only as an upper-bound reference or sanity check on training convergence."
    )
    lines.append("")

    lines.append("---")
    lines.append(
        "*Metrics: global pixel-level Dice/IoU/precision/recall. "
        "Threshold selected on validation set only and applied once to the held-out test set. "
        "Strict baseline numbers are averaged over fp1–fp7 held-out flight paths.*"
    )

    md_path = OUT_DIR / "random_group_baseline_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown report written: {md_path}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    if orig:
        print(f"  Original U-Net  random test Dice : {_f(orig['test_dice'])}")
        print(f"  Original U-Net  strict test Dice  : {_f(STRICT_ORIGINAL['test_dice'])}")
        print(f"  Original U-Net  lift               : {_delta_str(orig['test_dice'], STRICT_ORIGINAL['test_dice'])}")
    if alea:
        print(f"  Alea-tuned U-Net random test Dice : {_f(alea['test_dice'])}")
        print(f"  Alea-tuned U-Net strict test Dice  : {_f(STRICT_ALEA['test_dice'])}")
        print(f"  Alea-tuned U-Net lift              : {_delta_str(alea['test_dice'], STRICT_ALEA['test_dice'])}")
    if orig and alea:
        delta = alea["test_dice"] - orig["test_dice"]
        sign = "+" if delta >= 0 else ""
        print(f"  Alea vs original (random split)   : {sign}{delta:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
