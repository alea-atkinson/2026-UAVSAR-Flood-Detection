"""
Audit script for IEEE PNG-filtered standard/strict leave-one-flight-path-out CSV splits.

Split root: csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val
Audits both: standard/ and strict_no_overlap/
"""

import os
import sys
import csv
import textwrap
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_ROOT = PROJECT_ROOT / "csv_splits" / "flood_splits_ieee_png_filtered_standard_strict_train_val"
DATA_ROOT = PROJECT_ROOT / "2025_Tile_Data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "split_audit"

SPLIT_TYPES = {
    "standard": SPLIT_ROOT / "standard",
    "strict_no_overlap": SPLIT_ROOT / "strict_no_overlap",
}

FLIGHT_PATHS = [f"fp{n}" for n in range(1, 8)]
REQUIRED_COLUMNS = {"tile_name", "flight_path", "uavsar_path", "flood_mask_path"}
OPTIONAL_COLUMNS = {"land_cover_mask_path"}
PATH_COLUMNS = ["uavsar_path", "flood_mask_path", "land_cover_mask_path"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hdr(msg: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{msg}\n{'=' * width}"


def _sub(msg: str) -> str:
    return f"  {msg}"


def _ok(msg: str) -> str:
    return f"  [PASS] {msg}"


def _fail(msg: str) -> str:
    return f"  [FAIL] {msg}"


def _info(msg: str) -> str:
    return f"  [INFO] {msg}"


def _warn(msg: str) -> str:
    return f"  [WARN] {msg}"


# ---------------------------------------------------------------------------
# Mask validity
# ---------------------------------------------------------------------------

def audit_masks(df: pd.DataFrame, split: str, fp_label: str, role: str) -> dict:
    """Open every flood mask in df and collect pixel statistics."""
    if "flood_mask_path" not in df.columns:
        return {}

    total = 0
    has_flood = 0
    no_flood = 0
    total_pos = 0
    total_neg = 0
    missing = 0

    for rel_path in df["flood_mask_path"].dropna():
        full = DATA_ROOT / rel_path
        if not full.exists():
            missing += 1
            continue
        try:
            with rasterio.open(full) as src:
                data = src.read(1)
        except Exception:
            missing += 1
            continue
        total += 1
        pos = int((data > 0).sum())
        neg = int((data == 0).sum())
        total_pos += pos
        total_neg += neg
        if pos > 0:
            has_flood += 1
        else:
            no_flood += 1

    return {
        "split_type": split,
        "heldout_fp": fp_label,
        "role": role,
        "total_masks": total,
        "masks_with_flood": has_flood,
        "masks_no_flood": no_flood,
        "masks_missing": missing,
        "total_positive_pixels": total_pos,
        "total_negative_pixels": total_neg,
    }


# ---------------------------------------------------------------------------
# Single heldout-FP audit
# ---------------------------------------------------------------------------

def audit_one_fp(
    split_name: str,
    split_dir: Path,
    fp: str,
    lines: list,
    rows: list,
) -> bool:
    """
    Audit one heldout flight path for one split type.
    Appends human-readable lines and CSV row dicts.
    Returns True if all FAIL-level checks passed.
    """
    fp_n = fp  # e.g. "fp3"
    overall_pass = True

    lines.append(_sub(f"--- heldout {fp_n} ---"))

    # ------------------------------------------------------------------ #
    # 1. Required files
    # ------------------------------------------------------------------ #
    file_map = {}
    for role in ("train", "validation", "test"):
        fname = split_dir / f"heldout_{fp_n}_{role}.csv"
        file_map[role] = fname
        if not fname.exists():
            lines.append(_fail(f"Missing file: {fname.name}"))
            overall_pass = False

    if not all(p.exists() for p in file_map.values()):
        lines.append(_info("Skipping further checks for this fp (missing files)."))
        row = {
            "split_type": split_name, "heldout_fp": fp_n,
            "check": "file_existence", "status": "FAIL",
            "detail": "one or more required CSVs missing",
        }
        rows.append(row)
        return False

    # ------------------------------------------------------------------ #
    # Load dataframes
    # ------------------------------------------------------------------ #
    dfs = {}
    for role, path in file_map.items():
        dfs[role] = pd.read_csv(path)

    # ------------------------------------------------------------------ #
    # 2. Required columns
    # ------------------------------------------------------------------ #
    col_ok = True
    for role, df in dfs.items():
        actual = set(df.columns)
        missing_req = REQUIRED_COLUMNS - actual
        if missing_req:
            lines.append(_fail(f"{role}: missing required columns {missing_req}"))
            lines.append(_info(f"  actual columns: {sorted(actual)}"))
            overall_pass = False
            col_ok = False
        else:
            lines.append(_ok(f"{role}: all required columns present"))

    # ------------------------------------------------------------------ #
    # 3. Row counts
    # ------------------------------------------------------------------ #
    for role, df in dfs.items():
        lines.append(_info(f"{role} rows: {len(df)}"))
        rows.append({
            "split_type": split_name, "heldout_fp": fp_n,
            "check": f"row_count_{role}", "status": "INFO",
            "detail": str(len(df)),
        })

    # ------------------------------------------------------------------ #
    # 4. Test set correctness: all test rows must have flight_path == fp_n
    # ------------------------------------------------------------------ #
    if "flight_path" in dfs["test"].columns:
        bad_test = dfs["test"][dfs["test"]["flight_path"] != fp_n]
        if len(bad_test) > 0:
            lines.append(_fail(
                f"test: {len(bad_test)} rows have wrong flight_path "
                f"(expected only {fp_n})"
            ))
            overall_pass = False
        else:
            lines.append(_ok(f"test: all {len(dfs['test'])} rows have flight_path == {fp_n}"))
        rows.append({
            "split_type": split_name, "heldout_fp": fp_n,
            "check": "test_flight_path_correct",
            "status": "PASS" if len(bad_test) == 0 else "FAIL",
            "detail": f"{len(bad_test)} wrong-fp rows in test",
        })

    # ------------------------------------------------------------------ #
    # 5. Train / val must NOT contain heldout fp
    # ------------------------------------------------------------------ #
    for role in ("train", "validation"):
        df = dfs[role]
        if "flight_path" not in df.columns:
            continue
        leaked = df[df["flight_path"] == fp_n]
        if len(leaked) > 0:
            lines.append(_fail(
                f"{role}: {len(leaked)} rows contain heldout {fp_n}"
            ))
            overall_pass = False
        else:
            lines.append(_ok(f"{role}: heldout {fp_n} not present"))
        rows.append({
            "split_type": split_name, "heldout_fp": fp_n,
            "check": f"{role}_no_heldout_fp",
            "status": "PASS" if len(leaked) == 0 else "FAIL",
            "detail": f"{len(leaked)} leaked rows",
        })

    # ------------------------------------------------------------------ #
    # 6. Train / validation tile_name leakage (always a FAIL)
    # ------------------------------------------------------------------ #
    if "tile_name" in dfs["train"].columns and "tile_name" in dfs["validation"].columns:
        train_tiles = set(dfs["train"]["tile_name"])
        val_tiles = set(dfs["validation"]["tile_name"])
        tv_overlap = train_tiles & val_tiles
        if tv_overlap:
            lines.append(_fail(
                f"train/val tile_name overlap: {len(tv_overlap)} shared tiles"
            ))
            overall_pass = False
        else:
            lines.append(_ok("train/val: no tile_name overlap"))
        rows.append({
            "split_type": split_name, "heldout_fp": fp_n,
            "check": "train_val_tile_name_overlap",
            "status": "PASS" if not tv_overlap else "FAIL",
            "detail": f"{len(tv_overlap)} overlapping tile_names",
        })

    # ------------------------------------------------------------------ #
    # 7 / 8. Strict vs standard tile_name overlap with test
    # ------------------------------------------------------------------ #
    if "tile_name" in dfs["test"].columns:
        test_tiles = set(dfs["test"]["tile_name"])

        for role in ("train", "validation"):
            role_tiles = set(dfs[role]["tile_name"]) if "tile_name" in dfs[role].columns else set()
            overlap = role_tiles & test_tiles

            if split_name == "strict_no_overlap":
                status = "PASS" if not overlap else "FAIL"
                msg = (
                    _ok(f"{role}/test: no tile_name overlap (strict)")
                    if not overlap
                    else _fail(f"{role}/test tile_name overlap: {len(overlap)} tiles (strict)")
                )
                if overlap:
                    overall_pass = False
            else:
                # standard: only report, do not fail
                status = "INFO"
                msg = _info(
                    f"{role}/test tile_name overlap: {len(overlap)} tiles "
                    f"({'expected by design for train/test' if role == 'train' else 'unexpected for val/test'})"
                )
                if role == "validation" and overlap:
                    # val/test overlap in standard split is still a FAIL
                    status = "FAIL"
                    msg = _fail(f"val/test tile_name overlap: {len(overlap)} tiles (standard)")
                    overall_pass = False

            lines.append(msg)
            rows.append({
                "split_type": split_name, "heldout_fp": fp_n,
                "check": f"{role}_test_tile_name_overlap",
                "status": status,
                "detail": f"{len(overlap)} overlapping tile_names",
            })

    # ------------------------------------------------------------------ #
    # 9. Duplicate checks
    # ------------------------------------------------------------------ #
    for role, df in dfs.items():
        for col in ("tile_name", "uavsar_path"):
            if col not in df.columns:
                continue
            dup_count = df[col].duplicated().sum()
            if dup_count > 0:
                lines.append(_warn(f"{role}: {dup_count} duplicate {col} values"))
            else:
                lines.append(_ok(f"{role}: no duplicate {col}"))
            rows.append({
                "split_type": split_name, "heldout_fp": fp_n,
                "check": f"{role}_duplicate_{col}",
                "status": "WARN" if dup_count > 0 else "PASS",
                "detail": f"{dup_count} duplicates",
            })

    # ------------------------------------------------------------------ #
    # 10. File path existence (sample-level check)
    # ------------------------------------------------------------------ #
    for role, df in dfs.items():
        for path_col in PATH_COLUMNS:
            if path_col not in df.columns:
                continue
            total = df[path_col].notna().sum()
            missing_paths = 0
            for rel in df[path_col].dropna():
                if not (DATA_ROOT / rel).exists():
                    missing_paths += 1
            status = "FAIL" if missing_paths > 0 else "PASS"
            msg = (
                _fail(f"{role}/{path_col}: {missing_paths}/{total} paths missing")
                if missing_paths
                else _ok(f"{role}/{path_col}: all {total} paths exist")
            )
            lines.append(msg)
            if missing_paths:
                overall_pass = False
            rows.append({
                "split_type": split_name, "heldout_fp": fp_n,
                "check": f"{role}_{path_col}_paths_exist",
                "status": status,
                "detail": f"{missing_paths}/{total} missing",
            })

    # ------------------------------------------------------------------ #
    # 11. Mask validity (rasterio pixel stats)
    # ------------------------------------------------------------------ #
    for role, df in dfs.items():
        stats = audit_masks(df, split_name, fp_n, role)
        if stats:
            lines.append(_info(
                f"{role} masks: total={stats['total_masks']} "
                f"with_flood={stats['masks_with_flood']} "
                f"no_flood={stats['masks_no_flood']} "
                f"missing={stats['masks_missing']} "
                f"pos_px={stats['total_positive_pixels']} "
                f"neg_px={stats['total_negative_pixels']}"
            ))
            rows.append({
                "split_type": split_name, "heldout_fp": fp_n,
                "check": f"{role}_mask_validity",
                "status": "INFO",
                "detail": (
                    f"total={stats['total_masks']} with_flood={stats['masks_with_flood']} "
                    f"no_flood={stats['masks_no_flood']} missing={stats['masks_missing']} "
                    f"pos_px={stats['total_positive_pixels']} neg_px={stats['total_negative_pixels']}"
                ),
            })

    return overall_pass


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_lines = []
    all_rows = []
    global_pass = True

    all_lines.append(_hdr("IEEE PNG-Filtered Split Audit"))
    all_lines.append(f"Project root : {PROJECT_ROOT}")
    all_lines.append(f"Split root   : {SPLIT_ROOT}")
    all_lines.append(f"Data root    : {DATA_ROOT}")
    all_lines.append(f"Output dir   : {OUTPUT_DIR}")

    for split_name, split_dir in SPLIT_TYPES.items():
        all_lines.append(_hdr(f"Split type: {split_name}"))

        if not split_dir.exists():
            all_lines.append(_warn(f"Directory not found: {split_dir}  — skipping"))
            continue

        split_pass = True
        for fp in FLIGHT_PATHS:
            fp_ok = audit_one_fp(split_name, split_dir, fp, all_lines, all_rows)
            if not fp_ok:
                split_pass = False
                global_pass = False

        verdict = "PASS" if split_pass else "FAIL"
        all_lines.append(_hdr(f"Split {split_name}: overall {verdict}"))

    # -------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------
    summary_path = OUTPUT_DIR / "filtered_strict_split_audit_summary.txt"
    summary_path.write_text("\n".join(all_lines) + "\n")

    csv_path = OUTPUT_DIR / "filtered_strict_split_audit_by_file.csv"
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(csv_path, index=False)

    # -------------------------------------------------------------------
    # Print to stdout
    # -------------------------------------------------------------------
    print("\n".join(all_lines))
    print()
    print(f"Summary written to: {summary_path}")
    print(f"CSV written to    : {csv_path}")
    print()
    if global_pass:
        print("OVERALL AUDIT RESULT: PASS")
    else:
        print("OVERALL AUDIT RESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
