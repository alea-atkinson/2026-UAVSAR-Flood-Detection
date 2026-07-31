#!/usr/bin/env python3
"""
analyze_landcover_vs_lofpo_performance.py

Exploratory analysis: does held-out flight-path (leave-one-flight-path-out,
LOFPO) segmentation performance appear related to land-cover composition?

---- Inputs (read this first) --------------------------------------------
- Land-cover composition per flight path:
    outputs/dataset_analysis/landcover_by_flight_path_ieee_filtered/
    landcover_by_flight_path_summary.csv
  (produced by scripts/analyze_landcover_by_flight_path.py)
- LOFPO Dice/IoU per held-out flight path, TUNED model (the model used
  throughout this project's hardware-verification work --
  "alea_tuned_filtered_strict_fp{N}_focaldice_adamw_20epochs"):
    outputs/alea_tuned_filtered_strict/alea_tuned_summary.csv
  (test_dice / test_iou columns; the ORIGINAL baseline U-Net's equivalent
  numbers also exist at results/filtered_strict_20epoch_test_summary.csv
  and are referenced in the markdown summary as secondary context, but
  are NOT joined into the main table -- the tuned model is used as
  the primary result since it is the model this project's hardware
  feasibility work is built around.)

---- Claim boundary (IMPORTANT) -------------------------------------------
This is an EXPLORATORY analysis with n=7 flight paths. It computes simple
Pearson correlations between land-cover percentages and Dice/IoU. It does
NOT establish causation. Any relationship reported is phrased as "may
help explain" or "is associated with" -- never "causes" -- and the small
sample size (n=7) means every correlation here is fragile and should not
be treated as statistically robust.

---- Outputs -----------------------------------------------------------
outputs/dataset_analysis/landcover_vs_lofpo_performance/
    landcover_vs_lofpo_performance.csv
    landcover_vs_lofpo_correlations.csv
    landcover_vs_lofpo_summary.md

---- Usage ------------------------------------------------------------------
    python3 scripts/analyze_landcover_vs_lofpo_performance.py
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

LANDCOVER_CSV = (
    REPO_ROOT / "outputs" / "dataset_analysis" / "landcover_by_flight_path_ieee_filtered"
    / "landcover_by_flight_path_summary.csv"
)
TUNED_PERFORMANCE_CSV = (
    REPO_ROOT / "outputs" / "alea_tuned_filtered_strict" / "alea_tuned_summary.csv"
)
BASELINE_PERFORMANCE_CSV = REPO_ROOT / "results" / "filtered_strict_20epoch_test_summary.csv"

OUT_DIR = REPO_ROOT / "outputs" / "dataset_analysis" / "landcover_vs_lofpo_performance"

# (output column name, land-cover CSV column name)
CLASS_COLUMNS = [
    ("water_pct", "class_1_water_pct"),
    ("trees_pct", "class_2_trees_pct"),
    ("flooded_vegetation_pct", "class_4_flooded_vegetation_pct"),
    ("crops_pct", "class_5_crops_pct"),
    ("built_area_pct", "class_7_built_area_pct"),
    ("bare_ground_pct", "class_8_bare_ground_pct"),
    ("rangeland_pct", "class_11_rangeland_pct"),
]


def load_csv_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"ERROR: required input file not found: {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1] Loading land-cover composition: {LANDCOVER_CSV.relative_to(REPO_ROOT)}")
    lc_rows = load_csv_rows(LANDCOVER_CSV)
    lc_by_fp = {r["flight_path"]: r for r in lc_rows if r["flight_path"] != "ALL"}
    print(f"    {len(lc_by_fp)} flight paths found: {sorted(lc_by_fp.keys())}")

    print(f"\n[2] Loading tuned-model LOFPO performance: {TUNED_PERFORMANCE_CSV.relative_to(REPO_ROOT)}")
    perf_rows = load_csv_rows(TUNED_PERFORMANCE_CSV)
    perf_by_fp = {
        r["heldout_fp"]: r for r in perf_rows
        if r["heldout_fp"] and r["heldout_fp"].startswith("fp")
    }
    print(f"    {len(perf_by_fp)} flight paths found: {sorted(perf_by_fp.keys())}")

    missing_files = []
    if not BASELINE_PERFORMANCE_CSV.exists():
        missing_files.append(str(BASELINE_PERFORMANCE_CSV.relative_to(REPO_ROOT)))
    baseline_by_fp = {}
    if BASELINE_PERFORMANCE_CSV.exists():
        print(f"\n[2b] Loading baseline-model LOFPO performance (secondary reference): "
              f"{BASELINE_PERFORMANCE_CSV.relative_to(REPO_ROOT)}")
        baseline_rows = load_csv_rows(BASELINE_PERFORMANCE_CSV)
        baseline_by_fp = {r["heldout_fp"]: r for r in baseline_rows}
        print(f"    {len(baseline_by_fp)} flight paths found.")
    else:
        print(f"\n[2b] NOTE: baseline performance file not found "
              f"({BASELINE_PERFORMANCE_CSV.relative_to(REPO_ROOT)}); "
              f"baseline comparison will be omitted from the summary.")

    fps = sorted(set(lc_by_fp.keys()) & set(perf_by_fp.keys()), key=lambda s: int(s[2:]))
    missing_lc = sorted(set(perf_by_fp.keys()) - set(lc_by_fp.keys()))
    missing_perf = sorted(set(lc_by_fp.keys()) - set(perf_by_fp.keys()))
    if missing_lc:
        print(f"    WARNING: flight path(s) in performance CSV but not in land-cover CSV: {missing_lc}")
    if missing_perf:
        print(f"    WARNING: flight path(s) in land-cover CSV but not in performance CSV: {missing_perf}")
    if not fps:
        sys.exit("ERROR: no flight paths in common between the two input files -- cannot join.")

    print(f"\n[3] Joining {len(fps)} flight paths: {fps}")
    joined_rows = []
    for fp in fps:
        lc = lc_by_fp[fp]
        perf = perf_by_fp[fp]
        row = {
            "flight_path": fp,
            "num_tiles": int(lc["num_tiles"]),
            "dice": float(perf["test_dice"]),
            "iou": float(perf["test_iou"]),
        }
        for out_col, lc_col in CLASS_COLUMNS:
            row[out_col] = float(lc[lc_col])
        joined_rows.append(row)

    # -- Write joined performance CSV --
    perf_csv_path = OUT_DIR / "landcover_vs_lofpo_performance.csv"
    fieldnames = ["flight_path", "num_tiles", "dice", "iou"] + [c[0] for c in CLASS_COLUMNS]
    with open(perf_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(joined_rows)
    print(f"    Written: {perf_csv_path.relative_to(REPO_ROOT)}")

    # -- Correlations --
    print("\n[4] Computing Pearson correlations (n={} flight paths) ...".format(len(joined_rows)))
    dice_arr = np.array([r["dice"] for r in joined_rows])
    iou_arr = np.array([r["iou"] for r in joined_rows])

    corr_rows = []
    for out_col, _ in CLASS_COLUMNS:
        class_arr = np.array([r[out_col] for r in joined_rows])
        r_dice = pearson_r(class_arr, dice_arr)
        r_iou = pearson_r(class_arr, iou_arr)
        corr_rows.append({
            "land_cover_class": out_col.replace("_pct", ""),
            "pearson_r_vs_dice": r_dice,
            "pearson_r_vs_iou": r_iou,
            "n_flight_paths": len(joined_rows),
        })
        print(f"    {out_col:26s} r_vs_dice={r_dice:+.4f}  r_vs_iou={r_iou:+.4f}")

    corr_csv_path = OUT_DIR / "landcover_vs_lofpo_correlations.csv"
    with open(corr_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["land_cover_class", "pearson_r_vs_dice",
                                                "pearson_r_vs_iou", "n_flight_paths"])
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"    Written: {corr_csv_path.relative_to(REPO_ROOT)}")

    # -- Markdown summary --
    print("\n[5] Writing markdown summary ...")
    write_markdown_summary(joined_rows, corr_rows, baseline_by_fp, missing_files)
    print(f"    Written: {(OUT_DIR / 'landcover_vs_lofpo_summary.md').relative_to(REPO_ROOT)}")

    # -- Print tables to terminal --
    print("\n" + "=" * 100)
    print("LAND-COVER vs. LOFPO PERFORMANCE (tuned model, test set)")
    print("=" * 100)
    header = f"{'FP':<5}{'Dice':>8}{'IoU':>8}{'Water%':>9}{'Trees%':>9}{'FloodVeg%':>11}{'Crops%':>9}{'Built%':>9}{'Bare%':>8}{'Range%':>9}"
    print(header)
    for r in sorted(joined_rows, key=lambda x: -x["dice"]):
        print(f"{r['flight_path']:<5}{r['dice']:>8.4f}{r['iou']:>8.4f}"
              f"{r['water_pct']:>9.2f}{r['trees_pct']:>9.2f}{r['flooded_vegetation_pct']:>11.2f}"
              f"{r['crops_pct']:>9.2f}{r['built_area_pct']:>9.2f}{r['bare_ground_pct']:>8.2f}"
              f"{r['rangeland_pct']:>9.2f}")

    print("\n" + "=" * 60)
    print("CORRELATIONS (Pearson r, n={})".format(len(joined_rows)))
    print("=" * 60)
    print(f"{'Class':<22}{'r vs Dice':>12}{'r vs IoU':>12}")
    for c in sorted(corr_rows, key=lambda x: -abs(x["pearson_r_vs_dice"])):
        print(f"{c['land_cover_class']:<22}{c['pearson_r_vs_dice']:>+12.4f}{c['pearson_r_vs_iou']:>+12.4f}")


def write_markdown_summary(joined_rows, corr_rows, baseline_by_fp, missing_files) -> None:
    by_dice_desc = sorted(joined_rows, key=lambda r: -r["dice"])
    highest = by_dice_desc[0]
    lowest = by_dice_desc[-1]

    class_names = {
        "water": "Water", "trees": "Trees", "flooded_vegetation": "Flooded Vegetation",
        "crops": "Crops", "built_area": "Built Area", "bare_ground": "Bare Ground",
        "rangeland": "Rangeland",
    }

    # Which classes vary most across flight paths (population std of the pct columns)
    variability = []
    for out_col, _ in CLASS_COLUMNS:
        vals = np.array([r[out_col] for r in joined_rows])
        variability.append((out_col.replace("_pct", ""), float(vals.std()), float(vals.max() - vals.min())))
    variability.sort(key=lambda t: -t[1])

    corr_sorted = sorted(corr_rows, key=lambda c: -abs(c["pearson_r_vs_dice"]))

    perf_table_rows = "\n".join(
        f"| {r['flight_path']} | {r['num_tiles']} | {r['dice']:.4f} | {r['iou']:.4f} | "
        f"{r['water_pct']:.2f}% | {r['trees_pct']:.2f}% | {r['flooded_vegetation_pct']:.2f}% | "
        f"{r['crops_pct']:.2f}% | {r['built_area_pct']:.2f}% | {r['bare_ground_pct']:.2f}% | "
        f"{r['rangeland_pct']:.2f}% |"
        for r in by_dice_desc
    )
    corr_table_rows = "\n".join(
        f"| {class_names[c['land_cover_class']]} | {c['pearson_r_vs_dice']:+.4f} | "
        f"{c['pearson_r_vs_iou']:+.4f} | {c['n_flight_paths']} |"
        for c in corr_sorted
    )
    variability_lines = "\n".join(
        f"- **{class_names[name]}**: std = {std:.2f} percentage points, range = {rng:.2f} pp across the 7 flight paths"
        for name, std, rng in variability
    )

    strongest_positive = max(corr_rows, key=lambda c: c["pearson_r_vs_dice"])
    strongest_negative = min(corr_rows, key=lambda c: c["pearson_r_vs_dice"])

    baseline_note = ""
    if baseline_by_fp:
        baseline_lines = "\n".join(
            f"| {fp} | {baseline_by_fp[fp]['test_dice']} | {baseline_by_fp[fp]['test_iou']} |"
            for fp in sorted(baseline_by_fp.keys(), key=lambda s: int(s[2:]))
            if fp in {r["flight_path"] for r in joined_rows}
        )
        baseline_note = f"""
## Secondary reference: original (non-tuned) baseline U-Net

The main table above uses the **tuned** model's test Dice/IoU
(`outputs/alea_tuned_filtered_strict/alea_tuned_summary.csv`), since that
is the model used throughout this project's hardware-feasibility work.
For reference, the original (non-tuned) baseline U-Net's test Dice/IoU
per held-out flight path (`results/filtered_strict_20epoch_test_summary.csv`)
is:

| Flight path | Baseline test Dice | Baseline test IoU |
|---|---:|---:|
{baseline_lines}

The same overall pattern (fp2 strongest, fp1/fp7 weakest) is visible in
the baseline model too, suggesting this is more likely a property of the
held-out flight paths' data than an artifact specific to the tuning
procedure -- though this is still an exploratory, small-sample
observation, not a controlled comparison.
"""
    missing_note = (
        f"\n**Note:** the following expected input file(s) were not found and were "
        f"skipped: {missing_files}.\n" if missing_files else ""
    )

    md = f"""# Land-Cover Composition vs. LOFPO Segmentation Performance

Generated by `scripts/analyze_landcover_vs_lofpo_performance.py`.

## Claim boundary (read this first)

- **This is exploratory.** With only 7 flight paths (n=7), every
  correlation reported here is a small-sample observation, not a
  statistically robust result.
- **No causal claim is made anywhere in this document.** Any
  relationship described uses "may help explain," "is associated with,"
  or similar language -- never "causes."
- Dice/IoU figures are the **tuned** model's test-set results per
  held-out flight path (leave-one-flight-path-out, LOFPO), from the IEEE
  PNG-filtered strict no-overlap splits.
- Land-cover percentages are computed across **all land-cover pixels** in
  each flight path's tile set (not just flooded pixels), from
  `outputs/dataset_analysis/landcover_by_flight_path_ieee_filtered/landcover_by_flight_path_summary.csv`.
{missing_note}
## Joined table (sorted by Dice, descending)

| Flight path | Tiles | Dice | IoU | Water % | Trees % | Flooded Veg % | Crops % | Built Area % | Bare Ground % | Rangeland % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{perf_table_rows}
{baseline_note}
## Which held-out paths had the highest/lowest Dice?

- **Highest Dice**: **{highest['flight_path']}** (Dice = {highest['dice']:.4f}, IoU = {highest['iou']:.4f}).
- **Lowest Dice**: **{lowest['flight_path']}** (Dice = {lowest['dice']:.4f}, IoU = {lowest['iou']:.4f}).
- **{by_dice_desc[-2]['flight_path']}** (Dice = {by_dice_desc[-2]['dice']:.4f}) is a close second-lowest,
  meaning the two weakest held-out paths ({lowest['flight_path']} and {by_dice_desc[-2]['flight_path']})
  are nearly tied, while {highest['flight_path']} is a clear outlier at the top of the ranking.

## Which land-cover percentages vary most by flight path?

{variability_lines}

Crops, Trees, and Water vary the most across flight paths (tens of
percentage points of range); Bare Ground and Flooded Vegetation are
minor land-cover classes everywhere and vary the least in absolute terms.

## Are any land-cover classes visibly associated with lower Dice?

Pearson correlation between each land-cover percentage and Dice/IoU
across the 7 flight paths:

| Land-cover class | r vs. Dice | r vs. IoU | n |
|---|---:|---:|---:|
{corr_table_rows}

- **{class_names[strongest_positive['land_cover_class']]}** shows the
  strongest POSITIVE association with Dice (r = {strongest_positive['pearson_r_vs_dice']:+.4f}) --
  flight paths with a higher percentage of this class tend to have
  higher Dice in this sample.
- **{class_names[strongest_negative['land_cover_class']]}** shows the
  strongest NEGATIVE association with Dice (r = {strongest_negative['pearson_r_vs_dice']:+.4f}) --
  flight paths with a higher percentage of this class tend to have
  lower Dice in this sample. This is consistent with fp7 and fp1
  (the two lowest-Dice paths) among the higher-Water-percentage /
  lower-Built-Area-percentage flight paths in this dataset.
- With only 7 data points, none of these correlations should be read as
  a confirmed effect -- they describe a pattern in this specific sample,
  which **may help explain** part of the Dice variation across held-out
  flight paths, not a proven driver of it.

## Limitations

- **n=7.** Every correlation here is computed from exactly 7 flight
  paths -- far too few for any meaningful statistical significance
  testing, and a single unusual flight path (e.g. fp2's high Dice, or
  fp7's unusually high Water%) can dominate a correlation coefficient.
- **Confounded by other factors.** Flight-path Dice differences could
  also reflect flood extent/severity, SAR acquisition conditions, tile
  count, or other dataset properties not modeled here -- land-cover
  composition is only one candidate explanatory factor among several,
  and this analysis does not control for the others.
- **Correlation, not causation.** No claim is made that any land-cover
  class causally affects segmentation performance -- only that certain
  classes' percentages and Dice/IoU move together (positively or
  negatively) across this specific 7-flight-path sample.
- **Land-cover percentages describe the INPUT tiles' composition**, not
  the model's per-class accuracy -- this analysis does not measure
  whether the model performs worse ON pixels of any particular
  land-cover class (see `scripts/evaluate_landcover_errors.py` and
  `outputs/landcover_error_analysis/` for that separate, complementary
  analysis).
- **Tuned model only** in the main table; the baseline model shows a
  similar top/bottom pattern (see secondary reference above) but was not
  formally joined or correlated here.
"""
    (OUT_DIR / "landcover_vs_lofpo_summary.md").write_text(md)


if __name__ == "__main__":
    main()
