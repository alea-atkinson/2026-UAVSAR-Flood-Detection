#!/usr/bin/env python3
"""Summarize Alea-tuned U-Net results and compare against the original baseline.

Reads per-fp threshold-sweep CSVs from:
  outputs/alea_tuned_filtered_strict/threshold_sweeps/fp{N}_threshold_sweep.csv

Writes:
  outputs/alea_tuned_filtered_strict/alea_tuned_summary.csv
  outputs/alea_tuned_filtered_strict/alea_tuned_vs_original_unet.csv
  outputs/alea_tuned_filtered_strict/alea_tuned_summary.md

Run after all fp1–fp7 training + sweep runs have completed.
Partial results (only some fp's done) are handled gracefully.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "outputs" / "alea_tuned_filtered_strict" / "threshold_sweeps"
OUT_DIR = PROJECT_ROOT / "outputs" / "alea_tuned_filtered_strict"

FLIGHT_PATHS = list(range(1, 8))

# ---------------------------------------------------------------------------
# Original U-Net threshold-sweep baseline (global pixel, val-selected threshold)
# Source: threshold_sweep_baseline.py results, all-fp run
# ---------------------------------------------------------------------------
ORIGINAL_BASELINE: dict[int, dict] = {
    1: {"threshold": 0.25, "test_dice": 0.5205, "test_iou": 0.3518, "test_precision": 0.4597, "test_recall": 0.5998},
    2: {"threshold": 0.35, "test_dice": 0.7445, "test_iou": 0.5930, "test_precision": 0.7447, "test_recall": 0.7444},
    3: {"threshold": 0.35, "test_dice": 0.6264, "test_iou": 0.4560, "test_precision": 0.7483, "test_recall": 0.5387},
    4: {"threshold": 0.50, "test_dice": 0.4657, "test_iou": 0.3035, "test_precision": 0.5288, "test_recall": 0.4160},
    5: {"threshold": 0.25, "test_dice": 0.6725, "test_iou": 0.5066, "test_precision": 0.6109, "test_recall": 0.7480},
    6: {"threshold": 0.30, "test_dice": 0.6008, "test_iou": 0.4294, "test_precision": 0.6940, "test_recall": 0.5297},
    7: {"threshold": 0.30, "test_dice": 0.5570, "test_iou": 0.3860, "test_precision": 0.6799, "test_recall": 0.4718},
}
ORIGINAL_AVG_DICE = 0.5982


# ---------------------------------------------------------------------------
# Load one sweep CSV and extract the selected-threshold row
# ---------------------------------------------------------------------------

def _load_sweep(fp_n: int) -> dict | None:
    path = SWEEP_DIR / f"fp{fp_n}_threshold_sweep.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    if "selected" not in df.columns:
        print(f"  [WARN] fp{fp_n}: 'selected' column missing in {path.name} — skipping")
        return None

    sel = df[df["selected"] == True]
    if sel.empty:
        print(f"  [WARN] fp{fp_n}: no selected row found in {path.name} — skipping")
        return None

    row = sel.iloc[0]
    return {
        "heldout_fp": f"fp{fp_n}",
        "selected_threshold": float(row["threshold"]),
        "val_dice": float(row["val_dice"]),
        "val_iou": float(row["val_iou"]),
        "test_dice": float(row["test_dice"]),
        "test_iou": float(row["test_iou"]),
        "test_precision": float(row["test_precision"]),
        "test_recall": float(row["test_recall"]),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _f(v: float | str, fmt: str = ".4f") -> str:
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _win(alea: float, orig: float) -> str:
    delta = alea - orig
    if abs(delta) < 0.0005:
        return "tied"
    return "better" if delta > 0 else "worse"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load results
    # ------------------------------------------------------------------
    alea_rows: list[dict] = []
    missing_fps: list[int] = []

    for fp_n in FLIGHT_PATHS:
        result = _load_sweep(fp_n)
        if result is None:
            missing_fps.append(fp_n)
            print(f"  [INFO] fp{fp_n}: sweep CSV not found — will be marked as missing")
        else:
            alea_rows.append(result)
            print(
                f"  fp{fp_n}: thresh={result['selected_threshold']:.2f}  "
                f"val_dice={result['val_dice']:.4f}  "
                f"test_dice={result['test_dice']:.4f}  "
                f"test_iou={result['test_iou']:.4f}"
            )

    if not alea_rows:
        print("\nNo sweep results found yet. Run the training shell script first:")
        print("  bash scripts/run_alea_tuned_filtered_strict.sh")
        return

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(alea_rows)

    # Compute stats over available fps only
    dice_vals = [r["test_dice"] for r in alea_rows]
    iou_vals = [r["test_iou"] for r in alea_rows]
    prec_vals = [r["test_precision"] for r in alea_rows]
    rec_vals = [r["test_recall"] for r in alea_rows]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": "n/a", "std": "n/a"}
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 6), "std": round(std, 6)}

    stats_row = {
        "heldout_fp": f"mean (n={len(alea_rows)})",
        "selected_threshold": "",
        "val_dice": _stats([r["val_dice"] for r in alea_rows])["mean"],
        "val_iou": _stats([r["val_iou"] for r in alea_rows])["mean"],
        "test_dice": _stats(dice_vals)["mean"],
        "test_iou": _stats(iou_vals)["mean"],
        "test_precision": _stats(prec_vals)["mean"],
        "test_recall": _stats(rec_vals)["mean"],
    }
    std_row = {
        "heldout_fp": f"std  (n={len(alea_rows)})",
        "selected_threshold": "",
        "val_dice": _stats([r["val_dice"] for r in alea_rows])["std"],
        "val_iou": _stats([r["val_iou"] for r in alea_rows])["std"],
        "test_dice": _stats(dice_vals)["std"],
        "test_iou": _stats(iou_vals)["std"],
        "test_precision": _stats(prec_vals)["std"],
        "test_recall": _stats(rec_vals)["std"],
    }

    full_df = pd.concat([summary_df, pd.DataFrame([stats_row, std_row])], ignore_index=True)
    summary_csv = OUT_DIR / "alea_tuned_summary.csv"
    full_df.to_csv(summary_csv, index=False)
    print(f"\nSummary CSV written: {summary_csv}")

    # ------------------------------------------------------------------
    # Comparison CSV
    # ------------------------------------------------------------------
    comp_rows = []
    for r in alea_rows:
        fp_n = int(r["heldout_fp"].replace("fp", ""))
        orig = ORIGINAL_BASELINE[fp_n]
        verdict = _win(r["test_dice"], orig["test_dice"])
        delta = r["test_dice"] - orig["test_dice"]
        comp_rows.append({
            "heldout_fp": r["heldout_fp"],
            # Alea tuned
            "alea_threshold": r["selected_threshold"],
            "alea_val_dice": round(r["val_dice"], 6),
            "alea_test_dice": round(r["test_dice"], 6),
            "alea_test_iou": round(r["test_iou"], 6),
            "alea_test_precision": round(r["test_precision"], 6),
            "alea_test_recall": round(r["test_recall"], 6),
            # Original
            "orig_threshold": orig["threshold"],
            "orig_test_dice": orig["test_dice"],
            "orig_test_iou": orig["test_iou"],
            "orig_test_precision": orig["test_precision"],
            "orig_test_recall": orig["test_recall"],
            # Verdict
            "dice_delta": round(delta, 6),
            "verdict": verdict,
        })

    # Overall verdict
    if len(alea_rows) == 7:
        alea_avg = statistics.mean(dice_vals)
        avg_delta = alea_avg - ORIGINAL_AVG_DICE
        overall_verdict = _win(alea_avg, ORIGINAL_AVG_DICE)
    else:
        alea_avg = statistics.mean(dice_vals)
        avg_delta = alea_avg - ORIGINAL_AVG_DICE
        overall_verdict = f"{_win(alea_avg, ORIGINAL_AVG_DICE)} (partial: {len(alea_rows)}/7 fps)"

    comp_rows.append({
        "heldout_fp": f"mean (n={len(alea_rows)})",
        "alea_threshold": "",
        "alea_val_dice": "",
        "alea_test_dice": round(alea_avg, 6),
        "alea_test_iou": round(statistics.mean(iou_vals), 6),
        "alea_test_precision": round(statistics.mean(prec_vals), 6),
        "alea_test_recall": round(statistics.mean(rec_vals), 6),
        "orig_threshold": "",
        "orig_test_dice": ORIGINAL_AVG_DICE,
        "orig_test_iou": "",
        "orig_test_precision": "",
        "orig_test_recall": "",
        "dice_delta": round(avg_delta, 6),
        "verdict": overall_verdict,
    })

    comp_csv = OUT_DIR / "alea_tuned_vs_original_unet.csv"
    pd.DataFrame(comp_rows).to_csv(comp_csv, index=False)
    print(f"Comparison CSV written: {comp_csv}")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    lines: list[str] = []
    lines.append("# Alea-Tuned U-Net vs. Original U-Net Baseline")
    lines.append("")
    lines.append(
        "**Setup:** Alea's tuned plain U-Net (FocalDiceLoss, AdamW, "
        "lr=9.327e-05, wd=6.088e-06, batch_size=16, base_channels=32, epochs=20) "
        "trained on the IEEE PNG-filtered strict no-overlap leave-one-flight-path-out splits."
    )
    lines.append("")
    lines.append(
        "**Comparison baseline:** original plain U-Net (BCEWithLogitsLoss, Adam, "
        "lr=1e-3, wd=0, batch_size=8, base_channels=32, epochs=20, val_loss checkpoint) "
        "with validation-selected threshold sweep."
    )
    lines.append("")
    lines.append(
        "> All metrics are global pixel-level, computed by threshold sweep "
        "(threshold selected on validation Dice only, applied once to held-out test split)."
    )
    lines.append("")

    # Status
    if missing_fps:
        lines.append(f"> **Note:** Results are partial. Missing flight paths: {missing_fps}. "
                     f"Re-run summarizer after completing all sweeps.")
        lines.append("")

    # Per-fp results
    lines.append("## Alea-Tuned Results per Flight Path")
    lines.append("")
    lines.append("| FP | Sel thresh | Val Dice | Test Dice | Test IoU | Test Prec | Test Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in alea_rows:
        lines.append(
            f"| {r['heldout_fp']} "
            f"| {_f(r['selected_threshold'], '.2f')} "
            f"| {_f(r['val_dice'])} "
            f"| {_f(r['test_dice'])} "
            f"| {_f(r['test_iou'])} "
            f"| {_f(r['test_precision'])} "
            f"| {_f(r['test_recall'])} |"
        )
    # Stats row
    lines.append(
        f"| **mean (n={len(alea_rows)})** | — "
        f"| {_f(statistics.mean([r['val_dice'] for r in alea_rows]))} "
        f"| **{_f(alea_avg)}** "
        f"| {_f(statistics.mean(iou_vals))} "
        f"| {_f(statistics.mean(prec_vals))} "
        f"| {_f(statistics.mean(rec_vals))} |"
    )
    if len(alea_rows) > 1:
        lines.append(
            f"| std | — "
            f"| {_f(statistics.stdev([r['val_dice'] for r in alea_rows]))} "
            f"| {_f(statistics.stdev(dice_vals))} "
            f"| {_f(statistics.stdev(iou_vals))} "
            f"| {_f(statistics.stdev(prec_vals))} "
            f"| {_f(statistics.stdev(rec_vals))} |"
        )
    lines.append("")

    # Comparison table
    lines.append("## Head-to-Head Comparison: Alea Tuned vs. Original U-Net")
    lines.append("")
    lines.append(
        "| FP | Alea Dice | Orig Dice | Delta | Verdict | Alea Prec | Orig Prec | Alea Recall | Orig Recall |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in comp_rows:
        if r["heldout_fp"].startswith("mean"):
            continue
        fp_n = int(str(r["heldout_fp"]).replace("fp", ""))
        orig = ORIGINAL_BASELINE[fp_n]
        delta_str = f"+{_f(r['dice_delta'])}" if float(r["dice_delta"]) >= 0 else _f(r["dice_delta"])
        verdict_str = f"**{r['verdict']}**" if r["verdict"] == "better" else r["verdict"]
        lines.append(
            f"| {r['heldout_fp']} "
            f"| {_f(r['alea_test_dice'])} "
            f"| {_f(orig['test_dice'])} "
            f"| {delta_str} "
            f"| {verdict_str} "
            f"| {_f(r['alea_test_precision'])} "
            f"| {_f(orig['test_precision'])} "
            f"| {_f(r['alea_test_recall'])} "
            f"| {_f(orig['test_recall'])} |"
        )
    # Average row
    avg_delta_str = f"+{_f(avg_delta)}" if avg_delta >= 0 else _f(avg_delta)
    lines.append(
        f"| **mean** "
        f"| **{_f(alea_avg)}** "
        f"| **{_f(ORIGINAL_AVG_DICE)}** "
        f"| **{avg_delta_str}** "
        f"| **{overall_verdict}** "
        f"| — | — | — | — |"
    )
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    n_better = sum(1 for r in comp_rows if r["heldout_fp"] != f"mean (n={len(alea_rows)})" and r["verdict"] == "better")
    n_worse = sum(1 for r in comp_rows if r["heldout_fp"] != f"mean (n={len(alea_rows)})" and r["verdict"] == "worse")
    n_tied = sum(1 for r in comp_rows if r["heldout_fp"] != f"mean (n={len(alea_rows)})" and r["verdict"] == "tied")
    lines.append(
        f"Over {len(alea_rows)} completed flight path(s): "
        f"Alea-tuned is better on {n_better}, worse on {n_worse}, tied on {n_tied}."
    )
    lines.append("")
    if avg_delta > 0:
        lines.append(
            f"Average test Dice: Alea-tuned **{_f(alea_avg)}** vs. original **{_f(ORIGINAL_AVG_DICE)}** "
            f"(+{_f(avg_delta)} improvement)."
        )
    else:
        lines.append(
            f"Average test Dice: Alea-tuned **{_f(alea_avg)}** vs. original **{_f(ORIGINAL_AVG_DICE)}** "
            f"({_f(avg_delta)} — not yet an improvement)."
        )
    if missing_fps:
        lines.append("")
        lines.append(
            f"**Incomplete:** fp{missing_fps} not yet run. "
            "Re-run this summarizer after completing all flight paths for the full picture."
        )
    lines.append("")
    lines.append("---")
    lines.append(
        "*Metrics: global pixel-level Dice/IoU/precision/recall. "
        "Threshold selected on validation set only and applied once to the held-out test set. "
        "These are directly comparable to the original baseline threshold-sweep results.*"
    )

    md_path = OUT_DIR / "alea_tuned_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown report written: {md_path}")

    # ------------------------------------------------------------------
    # Print terminal summary
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print(f"  Alea-tuned mean test Dice  : {_f(alea_avg)}")
    print(f"  Original baseline mean Dice: {_f(ORIGINAL_AVG_DICE)}")
    delta_sign = "+" if avg_delta >= 0 else ""
    print(f"  Delta                      : {delta_sign}{_f(avg_delta)} ({overall_verdict})")
    print(f"  Flight paths completed     : {len(alea_rows)}/7")
    if missing_fps:
        print(f"  Still missing              : fp{missing_fps}")
    print("=" * 64)


if __name__ == "__main__":
    main()
