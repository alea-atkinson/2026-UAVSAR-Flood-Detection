#!/usr/bin/env python3
"""Create random train/validation/test splits from the IEEE PNG-filtered strict dataset.

The master dataset is reconstructed by concatenating the seven held-out test CSVs
from the strict no-overlap leave-one-flight-path-out splits.  Each held-out test
CSV is the complete test flight-path for that fold, so concatenating all seven
gives every tile exactly once.

Two split variants are produced (seed=42, ratio=70/15/15):

  A. random_record_split
       Shuffle individual rows, then split 70/15/15.

  B. random_tile_name_group_split
       Shuffle tile_name groups, then split 70/15/15 by group.  No tile_name
       can appear in more than one of train/validation/test.

Outputs
-------
  csv_splits/flood_splits_ieee_png_filtered_random_train_val_test/
    master_dataset_from_heldout_tests.csv
    random_record_split/train.csv
    random_record_split/validation.csv
    random_record_split/test.csv
    random_tile_name_group_split/train.csv
    random_tile_name_group_split/validation.csv
    random_tile_name_group_split/test.csv
    random_split_summary.txt
    random_split_audit.csv

Usage
-----
    python scripts/make_ieee_png_filtered_random_splits.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STRICT_DIR = Path(
    "csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/strict_no_overlap"
)
OUT_DIR = Path("csv_splits/flood_splits_ieee_png_filtered_random_train_val_test")

FPS = list(range(1, 8))
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test fraction = 1 - TRAIN_FRAC - VAL_FRAC

PATH_COLS = ["uavsar_path", "flood_mask_path", "land_cover_mask_path"]
REQUIRED_COLS = ["sample_id", "flight_path", "tile_name", "uavsar_path", "flood_mask_path"]

SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Step 1 – Build master dataset
# ---------------------------------------------------------------------------

def build_master(strict_dir: Path) -> pd.DataFrame:
    frames = []
    for fp in FPS:
        p = strict_dir / f"heldout_fp{fp}_test.csv"
        if not p.exists():
            print(f"[ERROR] Missing: {p}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(p)
        df["_source_file"] = p.name
        frames.append(df)
    master = pd.concat(frames, ignore_index=True)
    return master


# ---------------------------------------------------------------------------
# Step 2 – Audit master
# ---------------------------------------------------------------------------

def audit_master(df: pd.DataFrame, n_dupes_removed: int) -> list[str]:
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("MASTER DATASET AUDIT")
    lines.append("=" * 70)
    lines.append(f"  Source: seven heldout_fp*_test.csv files from strict no-overlap splits")
    lines.append(f"  Exact duplicate rows found and removed : {n_dupes_removed}")
    lines.append(f"  Total rows after deduplication         : {len(df)}")
    lines.append(f"  Unique tile_names                      : {df['tile_name'].nunique()}")
    lines.append("")

    # Required columns
    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        lines.append(f"  [WARN] Missing required columns: {missing_required}")
    else:
        lines.append(f"  Required columns present: {REQUIRED_COLS}")
    lines.append("")

    # Rows per flight path
    lines.append("  Rows per flight_path:")
    fp_counts = df["flight_path"].value_counts().sort_index()
    for fp_label, cnt in fp_counts.items():
        pct = 100 * cnt / len(df)
        lines.append(f"    {fp_label:>5}  {cnt:5d}  ({pct:.1f}%)")
    lines.append("")

    # Path columns
    for col in PATH_COLS:
        if col in df.columns:
            n_blank = df[col].isna().sum() + (df[col] == "").sum()
            lines.append(f"  {col}: {len(df) - n_blank} non-blank, {n_blank} blank/NaN")
        else:
            lines.append(f"  {col}: column not present")
    lines.append("")

    # Boolean availability columns
    for col in ["has_flood_mask", "has_land_cover_mask"]:
        if col in df.columns:
            vc = df[col].value_counts().to_dict()
            lines.append(f"  {col}: {vc}")
    lines.append("")

    # tile_name appearing in multiple flight paths
    cross = df.groupby("tile_name")["flight_path"].nunique()
    n_cross = (cross > 1).sum()
    n_cross_rows = df[df["tile_name"].isin(cross[cross > 1].index)].shape[0]
    lines.append(f"  tile_name in >1 flight_path: {n_cross} tile names, {n_cross_rows} rows")
    if "tile_name_appears_in_multiple_flight_paths" in df.columns:
        col_flag = df["tile_name_appears_in_multiple_flight_paths"].value_counts().to_dict()
        lines.append(f"  tile_name_appears_in_multiple_flight_paths (column): {col_flag}")
    lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Step 3 – Random record split
# ---------------------------------------------------------------------------

def record_split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(round(n * TRAIN_FRAC))
    n_val   = int(round(n * VAL_FRAC))
    return {
        "train":      shuffled.iloc[:n_train],
        "validation": shuffled.iloc[n_train : n_train + n_val],
        "test":       shuffled.iloc[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Step 4 – Random tile_name group split
# ---------------------------------------------------------------------------

def tile_group_split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    unique_tiles = df["tile_name"].unique()
    shuffled_tiles = rng.permutation(unique_tiles)

    n = len(shuffled_tiles)
    n_train = int(round(n * TRAIN_FRAC))
    n_val   = int(round(n * VAL_FRAC))

    train_tiles = set(shuffled_tiles[:n_train])
    val_tiles   = set(shuffled_tiles[n_train : n_train + n_val])
    test_tiles  = set(shuffled_tiles[n_train + n_val :])

    return {
        "train":      df[df["tile_name"].isin(train_tiles)].copy(),
        "validation": df[df["tile_name"].isin(val_tiles)].copy(),
        "test":       df[df["tile_name"].isin(test_tiles)].copy(),
    }


# ---------------------------------------------------------------------------
# Step 5 – Audit a split variant
# ---------------------------------------------------------------------------

def audit_split_variant(
    name: str,
    splits: dict[str, pd.DataFrame],
    total_rows: int,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"\n{'='*70}")
    lines.append(f"SPLIT AUDIT: {name}")
    lines.append(f"{'='*70}")

    # Row and tile counts
    lines.append(f"\n  {'Split':>12}  {'Rows':>6}  {'Rows%':>6}  {'Tiles':>6}  {'Tiles%':>7}")
    lines.append(f"  {'-'*50}")
    n_total_tiles = sum(len(sp["tile_name"].unique()) for sp in splits.values())
    for split_name in SPLITS:
        sp = splits[split_name]
        n_rows = len(sp)
        n_tiles = sp["tile_name"].nunique()
        pct_rows  = 100 * n_rows  / total_rows if total_rows else 0
        pct_tiles = 100 * n_tiles / n_total_tiles if n_total_tiles else 0
        lines.append(
            f"  {split_name:>12}  {n_rows:>6}  {pct_rows:>5.1f}%  {n_tiles:>6}  {pct_tiles:>6.1f}%"
        )
    lines.append("")

    # Rows per flight_path per split
    lines.append("  Rows per flight_path:")
    all_fps = sorted(
        set().union(*(sp["flight_path"].unique() for sp in splits.values()))
    )
    header = f"  {'flight_path':>14}" + "".join(f"  {s:>12}" for s in SPLITS)
    lines.append(header)
    for fp_label in all_fps:
        row_str = f"  {fp_label:>14}"
        for split_name in SPLITS:
            sp = splits[split_name]
            cnt = (sp["flight_path"] == fp_label).sum()
            pct = 100 * cnt / len(sp) if len(sp) else 0
            row_str += f"  {cnt:>6} ({pct:>4.1f}%)"
        lines.append(row_str)
    lines.append("")

    # tile_name overlap between splits
    train_tiles = set(splits["train"]["tile_name"])
    val_tiles   = set(splits["validation"]["tile_name"])
    test_tiles  = set(splits["test"]["tile_name"])

    tv_overlap  = len(train_tiles & val_tiles)
    tt_overlap  = len(train_tiles & test_tiles)
    vt_overlap  = len(val_tiles  & test_tiles)
    tvt_overlap = len(train_tiles & val_tiles & test_tiles)

    lines.append("  tile_name overlap between splits:")
    lines.append(f"    train ∩ val            : {tv_overlap}")
    lines.append(f"    train ∩ test           : {tt_overlap}")
    lines.append(f"    val   ∩ test           : {vt_overlap}")
    lines.append(f"    train ∩ val ∩ test     : {tvt_overlap}")
    lines.append("")

    # Missing paths
    for col in PATH_COLS:
        if col in next(iter(splits.values())).columns:
            missing = sum(
                splits[s][col].isna().sum() + (splits[s][col] == "").sum()
                for s in SPLITS
            )
            lines.append(f"  {col} missing across all splits: {missing}")
    lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Step 6 – Build audit CSV rows
# ---------------------------------------------------------------------------

def build_audit_rows(
    variant: str,
    splits: dict[str, pd.DataFrame],
    total_rows: int,
) -> list[dict]:
    rows = []
    train_tiles = set(splits["train"]["tile_name"])
    val_tiles   = set(splits["validation"]["tile_name"])
    test_tiles  = set(splits["test"]["tile_name"])

    for split_name in SPLITS:
        sp = splits[split_name]
        fp_counts = sp["flight_path"].value_counts().to_dict()

        # Missing paths per column
        missing: dict[str, int] = {}
        for col in PATH_COLS:
            if col in sp.columns:
                missing[col] = int(sp[col].isna().sum() + (sp[col] == "").sum())
            else:
                missing[col] = -1

        # tile_name overlaps
        this_tiles = set(sp["tile_name"])
        other_tiles = {
            "train":      train_tiles,
            "validation": val_tiles,
            "test":       test_tiles,
        }
        overlaps = {
            f"tile_overlap_with_{k}": len(this_tiles & other_tiles[k]) if k != split_name else "self"
            for k in SPLITS
        }

        row = {
            "variant": variant,
            "split": split_name,
            "n_rows": len(sp),
            "pct_rows": round(100 * len(sp) / total_rows, 2) if total_rows else 0,
            "n_unique_tile_names": sp["tile_name"].nunique(),
        }
        for fp_label in [f"fp{i}" for i in FPS]:
            row[f"rows_{fp_label}"] = int(fp_counts.get(fp_label, 0))
            row[f"pct_{fp_label}"] = round(
                100 * fp_counts.get(fp_label, 0) / len(sp), 2
            ) if len(sp) else 0

        row.update(overlaps)
        for col, val in missing.items():
            row[f"missing_{col}"] = val

        # has_flood_mask, has_land_cover_mask counts
        for col in ["has_flood_mask", "has_land_cover_mask"]:
            if col in sp.columns:
                row[f"{col}_true"]  = int((sp[col] == True).sum())   # noqa: E712
                row[f"{col}_false"] = int((sp[col] == False).sum())  # noqa: E712

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("IEEE PNG-FILTERED RANDOM SPLIT GENERATION")
    print("=" * 70)
    print(f"  Source dir : {STRICT_DIR}")
    print(f"  Output dir : {OUT_DIR}")
    print(f"  Seed       : {SEED}")
    print(f"  Ratio      : {int(TRAIN_FRAC*100)}/{int(VAL_FRAC*100)}/{int((1-TRAIN_FRAC-VAL_FRAC)*100)} (train/val/test)")
    print()

    # --- Build master ---
    print("Step 1: Building master dataset from held-out test CSVs ...")
    master_raw = build_master(STRICT_DIR)

    n_dupes = master_raw.duplicated().sum()
    if n_dupes > 0:
        print(f"  [WARN] Found {n_dupes} exact duplicate rows — removing them.")
        master = master_raw.drop_duplicates().reset_index(drop=True)
    else:
        print(f"  No exact duplicate rows found.")
        master = master_raw.reset_index(drop=True)

    total_rows = len(master)
    print(f"  Master dataset: {total_rows} rows, {master['tile_name'].nunique()} unique tile_names")
    print()

    # --- Audit master ---
    print("Step 2: Auditing master dataset ...")
    audit_lines = audit_master(master, n_dupes)
    for line in audit_lines:
        print(line)

    # --- Record split ---
    print("Step 3: Creating random_record_split (seed=42) ...")
    rec_splits = record_split(master, SEED)
    for s, df in rec_splits.items():
        print(f"  {s:>12}: {len(df):5d} rows ({100*len(df)/total_rows:.1f}%)")
    print()

    # --- Tile group split ---
    print("Step 4: Creating random_tile_name_group_split (seed=42) ...")
    grp_splits = tile_group_split(master, SEED)
    for s, df in grp_splits.items():
        n_tiles = df["tile_name"].nunique()
        print(
            f"  {s:>12}: {len(df):5d} rows ({100*len(df)/total_rows:.1f}%)  "
            f"| {n_tiles} unique tile_names"
        )
    print()

    # --- Save files ---
    print("Step 5: Saving output files ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # master
    master_drop = master.drop(columns=["_source_file"], errors="ignore")
    master_path = OUT_DIR / "master_dataset_from_heldout_tests.csv"
    master_drop.to_csv(master_path, index=False)
    print(f"  Wrote: {master_path}")

    # record split
    rec_dir = OUT_DIR / "random_record_split"
    rec_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in rec_splits.items():
        p = rec_dir / f"{split_name}.csv"
        df.to_csv(p, index=False)
        print(f"  Wrote: {p}")

    # group split
    grp_dir = OUT_DIR / "random_tile_name_group_split"
    grp_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in grp_splits.items():
        p = grp_dir / f"{split_name}.csv"
        df.to_csv(p, index=False)
        print(f"  Wrote: {p}")

    print()

    # --- Audit split variants ---
    print("Step 6: Auditing split variants ...")
    rec_audit = audit_split_variant("random_record_split", rec_splits, total_rows)
    grp_audit = audit_split_variant("random_tile_name_group_split", grp_splits, total_rows)

    for line in rec_audit:
        print(line)
    for line in grp_audit:
        print(line)

    # --- Build audit CSV ---
    audit_rows: list[dict] = []
    audit_rows += build_audit_rows("random_record_split", rec_splits, total_rows)
    audit_rows += build_audit_rows("random_tile_name_group_split", grp_splits, total_rows)

    audit_df = pd.DataFrame(audit_rows)
    audit_path = OUT_DIR / "random_split_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"  Wrote: {audit_path}")

    # --- Summary text ---
    summary_lines = audit_lines + rec_audit + grp_audit
    summary_path = OUT_DIR / "random_split_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"  Wrote: {summary_path}")

    # --- Final terminal summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Master dataset total rows : {total_rows}")
    print(f"  Unique tile_names         : {master['tile_name'].nunique()}")
    cross = master.groupby("tile_name")["flight_path"].nunique()
    n_cross_tiles = int((cross > 1).sum())
    n_cross_rows  = int(master[master["tile_name"].isin(cross[cross > 1].index)].shape[0])
    print(f"  Tile names in >1 fp       : {n_cross_tiles} tile names, {n_cross_rows} rows")
    print()

    print(f"  {'':30}  {'train':>8} {'validation':>10} {'test':>8}")
    print(f"  {'-'*60}")
    for variant, splits in [
        ("random_record_split", rec_splits),
        ("random_tile_name_group_split", grp_splits),
    ]:
        row_counts = {s: len(splits[s]) for s in SPLITS}
        tile_counts = {s: splits[s]["tile_name"].nunique() for s in SPLITS}
        tr, va, te = row_counts["train"], row_counts["validation"], row_counts["test"]
        print(f"  {variant:30}  {tr:>8} {va:>10} {te:>8}  (rows)")
        tr2, va2, te2 = tile_counts["train"], tile_counts["validation"], tile_counts["test"]
        print(f"  {'  → unique tile_names':30}  {tr2:>8} {va2:>10} {te2:>8}")

    # tile overlap check
    print()
    for variant, splits in [
        ("random_record_split", rec_splits),
        ("random_tile_name_group_split", grp_splits),
    ]:
        train_t = set(splits["train"]["tile_name"])
        val_t   = set(splits["validation"]["tile_name"])
        test_t  = set(splits["test"]["tile_name"])
        overlap = len(train_t & val_t) + len(train_t & test_t) + len(val_t & test_t)
        status = "OK (no overlap)" if overlap == 0 else f"[WARN] {overlap} tile_name overlaps"
        print(f"  {variant}: tile_name overlap = {status}")

    print()

    # --- Commands to inspect output ---
    print("=" * 70)
    print("COMMANDS TO INSPECT OUTPUTS")
    print("=" * 70)
    print(f"""
# View summary report
cat {OUT_DIR}/random_split_summary.txt

# View audit CSV (formatted)
column -t -s, {OUT_DIR}/random_split_audit.csv | less -S

# Row counts in each split file
wc -l {OUT_DIR}/random_record_split/*.csv
wc -l {OUT_DIR}/random_tile_name_group_split/*.csv

# Peek at first rows of each split
head -3 {OUT_DIR}/random_record_split/train.csv
head -3 {OUT_DIR}/random_tile_name_group_split/train.csv

# Flight-path distribution in each split
python3 -c "
import pandas as pd
from pathlib import Path
for variant in ['random_record_split', 'random_tile_name_group_split']:
    print(f'\\n=== {{variant}} ===')
    for split in ['train', 'validation', 'test']:
        df = pd.read_csv(Path('{OUT_DIR}') / variant / f'{{split}}.csv')
        print(f'  {{split}}: {{dict(df.flight_path.value_counts().sort_index())}}')
"
""")


if __name__ == "__main__":
    main()
