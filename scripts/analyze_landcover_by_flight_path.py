#!/usr/bin/env python3
"""
analyze_landcover_by_flight_path.py

Computes the amount of each ESRI-style land-cover class per UAVSAR
flight path (fp1-fp7), for the IEEE PNG-filtered Florence flood dataset.

---- Index resolution (read this first) ------------------------------------
A prior attempt assumed the index lived at
`csv_splits/flood_dataset_index_ieee_png_filtered.csv` -- that path does
not exist in this repo. Inspection found:
  - `folder_split.py` (repo root) expects a file literally named
    `flood_dataset_index_ieee_png_filtered.csv` in the CURRENT WORKING
    DIRECTORY, and references an OLD `Data/land_cover_mask_tiles/` layout
    -- this was a one-time, ad-hoc reorganization script whose input CSV
    was never committed to this repo (it produced
    `New_organized_tiles/Only_PNG_Data/`, which IS committed).
  - The correct, currently-committed equivalent of that missing index is
    `csv_splits/flood_splits_ieee_png_filtered_random_train_val_test/
    master_dataset_from_heldout_tests.csv` -- built by
    `scripts/make_ieee_png_filtered_random_splits.py` by concatenating all
    seven `heldout_fp{1..7}_test.csv` files from the strict no-overlap
    splits (each held-out fp's test split is that flight path's complete
    tile set under the strict scheme, so the union covers every tile
    exactly once). EVERY row in this file already has
    `included_in_ieee_png_filtered_dataset == True` (1067/1067 rows
    checked), i.e. this file already IS the IEEE PNG-filtered dataset
    index, just under a different name/location than originally assumed.
  - Land-cover mask tiles live under `2025_Tile_Data/land_cover_mask_tiles/`
    (confirmed via `find . -type d -name land_cover_mask_tiles`), and the
    index's own `land_cover_mask_path` column values (e.g.
    `land_cover_mask_tiles/tile_10_48.tif`) are DATA-ROOT-relative and
    resolve directly under `2025_Tile_Data/`.

This script defaults to that master CSV and `2025_Tile_Data` as the data
root, but both are overridable via CLI flags, and path resolution is
robust to an index whose path columns are data-root-relative,
repo-root-relative, or already absolute (see `resolve_path`).

---- Scope -------------------------------------------------------------
Counts land-cover COMPOSITION across ALL land-cover pixels in each
flight path -- every pixel of every land-cover mask tile for that flight
path, NOT just pixels inside the flood mask. A separate, clearly-labeled
FLOODED-ONLY table is written in addition (not instead) if
`flood_mask_path` is present in the index and its files are readable.

---- Land-cover class codes (ESRI-style, as seen in these masks) -----------
    1  = Water
    2  = Trees
    4  = Flooded Vegetation
    5  = Crops
    7  = Built Area
    8  = Bare Ground
    11 = Rangeland
Any pixel value not in this set, and not equal to the raster's own
`nodata` value (per rasterio, checked per-tile), is counted separately as
"unknown" and its code recorded in `unknown_codes_seen`.

---- Outputs (under --output-dir) -----------------------------------------
    landcover_by_flight_path_summary.csv
    landcover_by_flight_path_summary.md
    landcover_by_tile.csv
    landcover_by_flight_path_flooded_pixels_only.csv   (optional, only if
        flood_mask_path is present in the index and readable)

---- Usage ------------------------------------------------------------------
    python3 scripts/analyze_landcover_by_flight_path.py
    python3 scripts/analyze_landcover_by_flight_path.py \\
        --index-csv <path> --data-root <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

import numpy as np
import rasterio

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_INDEX_CSV = (
    REPO_ROOT / "csv_splits" / "flood_splits_ieee_png_filtered_random_train_val_test"
    / "master_dataset_from_heldout_tests.csv"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "2025_Tile_Data"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "dataset_analysis" / "landcover_by_flight_path_ieee_filtered"
)

CLASS_NAMES: dict[int, str] = {
    1: "Water",
    2: "Trees",
    4: "Flooded Vegetation",
    5: "Crops",
    7: "Built Area",
    8: "Bare Ground",
    11: "Rangeland",
}
CLASS_CODES = sorted(CLASS_NAMES.keys())

REQUIRED_COLUMNS = ["flight_path", "tile_name", "land_cover_mask_path"]
DEFAULT_FILTER_COLUMN = "included_in_ieee_png_filtered_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute land-cover class composition per UAVSAR flight "
        "path for the IEEE PNG-filtered Florence flood dataset."
    )
    parser.add_argument("--index-csv", type=pathlib.Path, default=DEFAULT_INDEX_CSV,
                         help="Dataset index CSV. Default: the IEEE PNG-filtered "
                         "master index reconstructed from the strict no-overlap "
                         "held-out test splits.")
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT,
                         help="Root directory containing *_mask_tiles/ folders "
                         "(default: 2025_Tile_Data).")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--filter-column", default=DEFAULT_FILTER_COLUMN,
                         help="If this column exists in the index, only rows where "
                         "it is truthy are used (default: "
                         f"'{DEFAULT_FILTER_COLUMN}'). Pass '' to disable filtering.")
    return parser.parse_args()


def resolve_path(raw: str, data_root: pathlib.Path) -> pathlib.Path | None:
    """Robust to index path columns being data-root-relative (the common
    case, e.g. 'land_cover_mask_tiles/tile_10_48.tif'), repo-root-relative
    (e.g. '2025_Tile_Data/land_cover_mask_tiles/...'), or already absolute.
    Returns the first existing candidate, or None if none exist."""
    raw_path = pathlib.Path(raw)
    if raw_path.is_absolute():
        return raw_path if raw_path.exists() else None
    for candidate in (data_root / raw_path, REPO_ROOT / raw_path, pathlib.Path.cwd() / raw_path):
        if candidate.exists():
            return candidate
    return None


def truthy(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def count_classes(arr: np.ndarray, nodata) -> tuple[dict[int, int], int, int, set[int]]:
    """Returns (class_counts, nodata_pixels, unknown_pixels, unknown_codes_seen)."""
    class_counts: dict[int, int] = {c: 0 for c in CLASS_CODES}
    nodata_pixels = 0
    unknown_pixels = 0
    unknown_codes_seen: set[int] = set()

    values, counts = np.unique(arr, return_counts=True)
    for v, cnt in zip(values.tolist(), counts.tolist()):
        if nodata is not None and v == nodata:
            nodata_pixels += cnt
        elif v in class_counts:
            class_counts[v] += cnt
        else:
            unknown_pixels += cnt
            unknown_codes_seen.add(int(v))
    return class_counts, nodata_pixels, unknown_pixels, unknown_codes_seen


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] Loading index: {args.index_csv}")
    if not args.index_csv.exists():
        sys.exit(f"ERROR: index CSV not found at {args.index_csv}")
    with open(args.index_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    print(f"    {len(rows)} rows, columns: {fieldnames}")

    missing_required = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing_required:
        sys.exit(
            f"ERROR: index CSV is missing required column(s): {missing_required}\n"
            f"Required columns: {REQUIRED_COLUMNS}\n"
            f"Columns found: {fieldnames}"
        )

    has_flood_mask_col = "flood_mask_path" in fieldnames

    filter_col = args.filter_column
    if filter_col and filter_col in fieldnames:
        before = len(rows)
        rows = [r for r in rows if truthy(r[filter_col])]
        print(f"    Filtered by '{filter_col}' == True: {before} -> {len(rows)} rows")
    elif filter_col:
        print(f"    NOTE: filter column '{filter_col}' not present in index; "
              f"using all {len(rows)} rows unfiltered.")

    if not args.data_root.exists():
        sys.exit(f"ERROR: data root not found at {args.data_root}")

    # -- Per-tile pass --
    print(f"\n[2] Reading land-cover mask tiles from {args.data_root} ...")
    tile_rows: list[dict] = []
    fp_class_counts: dict[str, dict[int, int]] = defaultdict(lambda: {c: 0 for c in CLASS_CODES})
    fp_nodata: dict[str, int] = defaultdict(int)
    fp_unknown: dict[str, int] = defaultdict(int)
    fp_unknown_codes: dict[str, set[int]] = defaultdict(set)
    fp_tile_count: dict[str, int] = defaultdict(int)

    flooded_fp_class_counts: dict[str, dict[int, int]] = defaultdict(lambda: {c: 0 for c in CLASS_CODES})
    flooded_fp_nodata: dict[str, int] = defaultdict(int)
    flooded_fp_unknown: dict[str, int] = defaultdict(int)
    flooded_total_flood_pixels: dict[str, int] = defaultdict(int)
    any_flood_mask_readable = False

    n_missing_lc = 0
    n_missing_flood = 0

    for row in rows:
        fp = row["flight_path"]
        lc_path = resolve_path(row["land_cover_mask_path"], args.data_root)
        if lc_path is None:
            n_missing_lc += 1
            continue

        with rasterio.open(lc_path) as src:
            arr = src.read(1)
            nodata = src.nodata

        class_counts, nodata_px, unknown_px, unknown_codes = count_classes(arr, nodata)
        known_px = sum(class_counts.values())
        total_px = int(arr.size)

        fp_tile_count[fp] += 1
        for c in CLASS_CODES:
            fp_class_counts[fp][c] += class_counts[c]
        fp_nodata[fp] += nodata_px
        fp_unknown[fp] += unknown_px
        fp_unknown_codes[fp] |= unknown_codes

        tile_row = {
            "sample_id": row.get("sample_id", ""),
            "flight_path": fp,
            "tile_name": row["tile_name"],
            "land_cover_mask_path": row["land_cover_mask_path"],
            "total_pixels": total_px,
            "known_pixels": known_px,
            "nodata_pixels": nodata_px,
            "unknown_pixels": unknown_px,
            "unknown_codes": ";".join(str(c) for c in sorted(unknown_codes)),
        }
        for c in CLASS_CODES:
            tile_row[f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pixels"] = class_counts[c]
        for c in CLASS_CODES:
            pct = 100.0 * class_counts[c] / known_px if known_px else float("nan")
            tile_row[f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pct"] = pct
        tile_rows.append(tile_row)

        # -- Optional flooded-only accounting --
        if has_flood_mask_col and row.get("flood_mask_path"):
            flood_path = resolve_path(row["flood_mask_path"], args.data_root)
            if flood_path is None:
                n_missing_flood += 1
            else:
                with rasterio.open(flood_path) as fsrc:
                    flood_arr = fsrc.read(1)
                if flood_arr.shape != arr.shape:
                    print(f"    WARNING: shape mismatch for {row['tile_name']} "
                          f"(land-cover {arr.shape} vs. flood {flood_arr.shape}); "
                          f"skipping flooded-only accounting for this tile.")
                else:
                    any_flood_mask_readable = True
                    flood_bool = flood_arr > 0
                    flooded_lc = arr[flood_bool]
                    f_class_counts, f_nodata_px, f_unknown_px, f_unknown_codes = count_classes(
                        flooded_lc, nodata
                    )
                    for c in CLASS_CODES:
                        flooded_fp_class_counts[fp][c] += f_class_counts[c]
                    flooded_fp_nodata[fp] += f_nodata_px
                    flooded_fp_unknown[fp] += f_unknown_px
                    flooded_total_flood_pixels[fp] += int(flood_bool.sum())

    print(f"    Processed {len(tile_rows)} tiles ({n_missing_lc} land-cover mask file(s) "
          f"could not be found and were skipped).")
    if has_flood_mask_col and n_missing_flood:
        print(f"    ({n_missing_flood} flood mask file(s) could not be found; "
              f"skipped for the flooded-only table only.)")

    if not tile_rows:
        sys.exit("ERROR: no tiles were successfully processed -- nothing to summarize.")

    # -- Write per-tile CSV --
    tile_csv_path = output_dir / "landcover_by_tile.csv"
    tile_fieldnames = list(tile_rows[0].keys())
    with open(tile_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tile_fieldnames)
        writer.writeheader()
        writer.writerows(tile_rows)
    print(f"\n[3] Written: {tile_csv_path.relative_to(REPO_ROOT) if _is_under_repo(tile_csv_path) else tile_csv_path}")

    # -- Build per-flight-path summary rows --
    fps_sorted = sorted(fp_class_counts.keys(), key=lambda s: (len(s), s))
    summary_rows = []
    for fp in fps_sorted:
        known_total = sum(fp_class_counts[fp].values())
        row_out = {
            "flight_path": fp,
            "num_tiles": fp_tile_count[fp],
            "total_known_landcover_pixels": known_total,
            "nodata_pixels": fp_nodata[fp],
            "unknown_pixels": fp_unknown[fp],
            "unknown_codes_seen": ";".join(str(c) for c in sorted(fp_unknown_codes[fp])),
        }
        for c in CLASS_CODES:
            row_out[f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pixels"] = fp_class_counts[fp][c]
        for c in CLASS_CODES:
            pct = 100.0 * fp_class_counts[fp][c] / known_total if known_total else float("nan")
            row_out[f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pct"] = pct
        summary_rows.append(row_out)

    # -- Overall (all flight paths combined) row --
    overall_known = sum(r["total_known_landcover_pixels"] for r in summary_rows)
    overall_nodata = sum(r["nodata_pixels"] for r in summary_rows)
    overall_unknown = sum(r["unknown_pixels"] for r in summary_rows)
    overall_unknown_codes = sorted(set().union(*[fp_unknown_codes[fp] for fp in fps_sorted]) if fps_sorted else set())
    overall_row = {
        "flight_path": "ALL",
        "num_tiles": sum(r["num_tiles"] for r in summary_rows),
        "total_known_landcover_pixels": overall_known,
        "nodata_pixels": overall_nodata,
        "unknown_pixels": overall_unknown,
        "unknown_codes_seen": ";".join(str(c) for c in overall_unknown_codes),
    }
    for c in CLASS_CODES:
        col = f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pixels"
        overall_row[col] = sum(r[col] for r in summary_rows)
    for c in CLASS_CODES:
        pix_col = f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pixels"
        pct_col = f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pct"
        overall_row[pct_col] = 100.0 * overall_row[pix_col] / overall_known if overall_known else float("nan")

    summary_fieldnames = list(summary_rows[0].keys())
    summary_csv_path = output_dir / "landcover_by_flight_path_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
        writer.writerow(overall_row)
    print(f"    Written: {summary_csv_path.relative_to(REPO_ROOT) if _is_under_repo(summary_csv_path) else summary_csv_path}")

    # -- Optional flooded-only table --
    flooded_csv_path = None
    if has_flood_mask_col and any_flood_mask_readable:
        flooded_rows = []
        for fp in fps_sorted:
            known_total = sum(flooded_fp_class_counts[fp].values())
            row_out = {
                "flight_path": fp,
                "total_flood_pixels": flooded_total_flood_pixels[fp],
                "total_known_landcover_pixels_within_flood_mask": known_total,
                "nodata_pixels": flooded_fp_nodata[fp],
                "unknown_pixels": flooded_fp_unknown[fp],
            }
            for c in CLASS_CODES:
                row_out[f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pixels"] = flooded_fp_class_counts[fp][c]
            for c in CLASS_CODES:
                pct = 100.0 * flooded_fp_class_counts[fp][c] / known_total if known_total else float("nan")
                row_out[f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pct"] = pct
            flooded_rows.append(row_out)
        flooded_csv_path = output_dir / "landcover_by_flight_path_flooded_pixels_only.csv"
        with open(flooded_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flooded_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flooded_rows)
        print(f"    Written: {flooded_csv_path.relative_to(REPO_ROOT) if _is_under_repo(flooded_csv_path) else flooded_csv_path} "
              f"(OPTIONAL, flooded-pixels-only breakdown)")
    elif has_flood_mask_col:
        print("    NOTE: 'flood_mask_path' column present but no flood mask files were "
              "readable; skipped the optional flooded-only table.")
    else:
        print("    NOTE: no 'flood_mask_path' column in index; skipped the optional "
              "flooded-only table.")

    # -- Markdown summary --
    print("\n[4] Writing markdown summary ...")
    md_path = output_dir / "landcover_by_flight_path_summary.md"
    write_markdown_summary(md_path, args, summary_rows, overall_row, len(rows), n_missing_lc,
                            has_flood_mask_col, flooded_csv_path is not None)
    print(f"    Written: {md_path.relative_to(REPO_ROOT) if _is_under_repo(md_path) else md_path}")

    # -- Print summary table to terminal --
    print("\n" + "=" * 100)
    print("LAND-COVER COMPOSITION BY FLIGHT PATH (all land-cover pixels, IEEE PNG-filtered dataset)")
    print("=" * 100)
    header = f"{'FP':<5}{'Tiles':>7}{'Known px':>14}" + "".join(
        f"{CLASS_NAMES[c][:10]+'%':>13}" for c in CLASS_CODES
    ) + f"{'Unknown':>10}{'Nodata':>10}"
    print(header)
    for r in summary_rows + [overall_row]:
        line = f"{r['flight_path']:<5}{r['num_tiles']:>7}{r['total_known_landcover_pixels']:>14,}"
        for c in CLASS_CODES:
            pct = r[f"class_{c}_{CLASS_NAMES[c].replace(' ', '_').lower()}_pct"]
            line += f"{pct:>12.2f}%"
        line += f"{r['unknown_pixels']:>10,}{r['nodata_pixels']:>10,}"
        print(line)
    if any(r["unknown_pixels"] for r in summary_rows):
        print("\nUnknown codes seen by flight path:")
        for r in summary_rows:
            if r["unknown_codes_seen"]:
                print(f"  {r['flight_path']}: {r['unknown_codes_seen']}")
    else:
        print("\nNo unknown land-cover codes encountered (every pixel was either a known "
              "class code or the raster's declared nodata value).")


def _is_under_repo(p: pathlib.Path) -> bool:
    try:
        p.relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def write_markdown_summary(
    md_path: pathlib.Path,
    args: argparse.Namespace,
    summary_rows: list[dict],
    overall_row: dict,
    n_index_rows: int,
    n_missing_lc: int,
    has_flood_mask_col: bool,
    wrote_flooded_table: bool,
) -> None:
    def fmt_pct_cols(r: dict) -> str:
        return " | ".join(
            f"{r[f'class_{c}_' + CLASS_NAMES[c].replace(' ', '_').lower() + '_pct']:.2f}%"
            for c in CLASS_CODES
        )

    header_classes = " | ".join(f"{CLASS_NAMES[c]} (%)" for c in CLASS_CODES)
    table_rows = "\n".join(
        f"| {r['flight_path']} | {r['num_tiles']} | {r['total_known_landcover_pixels']:,} | "
        f"{fmt_pct_cols(r)} | {r['unknown_pixels']:,} | {r['nodata_pixels']:,} | "
        f"{r['unknown_codes_seen'] or '(none)'} |"
        for r in summary_rows
    )
    overall_line = (
        f"| **{overall_row['flight_path']}** | {overall_row['num_tiles']} | "
        f"{overall_row['total_known_landcover_pixels']:,} | {fmt_pct_cols(overall_row)} | "
        f"{overall_row['unknown_pixels']:,} | {overall_row['nodata_pixels']:,} | "
        f"{overall_row['unknown_codes_seen'] or '(none)'} |"
    )

    class_pixel_table_rows = "\n".join(
        "| " + r["flight_path"] + " | " + " | ".join(
            f"{r[f'class_{c}_' + CLASS_NAMES[c].replace(' ', '_').lower() + '_pixels']:,}"
            for c in CLASS_CODES
        ) + " |"
        for r in summary_rows
    )

    flooded_note = (
        "An additional, clearly-separate **flooded-pixels-only** table "
        "(`landcover_by_flight_path_flooded_pixels_only.csv`) was also written, "
        "breaking down land-cover composition restricted to pixels inside each "
        "tile's flood-change mask (`flood_mask_path`), for comparison. It is "
        "NOT the main result."
        if wrote_flooded_table else (
            "A `flood_mask_path` column exists in the index, but no flood mask "
            "files were readable, so no flooded-only table was written."
            if has_flood_mask_col else
            "No `flood_mask_path` column was present in the index, so no "
            "flooded-only table was written."
        )
    )

    md = f"""# Land-Cover Composition by Flight Path (IEEE PNG-Filtered Florence Dataset)

Generated by `scripts/analyze_landcover_by_flight_path.py`.

## Scope

This counts land-cover class composition across **ALL land-cover pixels**
in each flight path's tile set -- every pixel of every land-cover mask
tile, **not** restricted to flooded pixels. {flooded_note}

## Index / data root used

| Setting | Value |
|---|---|
| Index CSV | `{args.index_csv}` |
| Data root | `{args.data_root}` |
| Index rows read | {n_index_rows} |
| Land-cover mask files not found (skipped) | {n_missing_lc} |

See this script's module docstring for how the correct index path was
identified (the originally-assumed `flood_dataset_index_ieee_png_filtered.csv`
does not exist in this repo).

## Land-cover class codes used

| Code | Class |
|---:|---|
{chr(10).join(f"| {c} | {CLASS_NAMES[c]} |" for c in CLASS_CODES)}

Any pixel value not in this table, and not equal to the raster's own
`nodata` value, is counted as "unknown" and its code is recorded.

## Summary table (percent of known land-cover pixels per flight path)

| FP | Tiles | Known px | {header_classes} | Unknown px | Nodata px | Unknown codes |
|---|---:|---:|{"---:|" * len(CLASS_CODES)}---:|---:|---|
{table_rows}
{overall_line}

## Pixel counts per class (absolute)

| FP | {" | ".join(CLASS_NAMES[c] for c in CLASS_CODES)} |
|---|{"---:|" * len(CLASS_CODES)}
{class_pixel_table_rows}

## Limitations

- Percentages are computed relative to each flight path's **total known
  land-cover pixels** (excludes nodata and unknown-code pixels from the
  denominator).
- This is the IEEE PNG-filtered dataset index reconstructed from the
  strict no-overlap held-out test splits' union, not a separately
  maintained master index file (see module docstring).
- Land-cover mask rasters are treated as single-band; only band 1 is read.
"""
    md_path.write_text(md)


if __name__ == "__main__":
    main()
