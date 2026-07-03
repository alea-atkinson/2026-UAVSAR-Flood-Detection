#!/usr/bin/env python3
"""Create flood-fraction bucket CSVs from an inventory CSV.

The default input is the Florence Only_PNG_Data inventory created from the tile
tree. Despite the directory name, the current inventory points at GeoTIFF files.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated.*",
    category=DeprecationWarning,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY_CSV = (
    REPO_ROOT
    / "csv_organized_tiles"
    / "Only_PNG_Data"
    / "only_png_inventory"
    / "jz_local_all_files_inventory.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "csv_organized_tiles"
    / "Only_PNG_Data"
    / "only_png_inventory"
    / "flood_fraction_splits"
)
BUCKETS = [
    ("00-01", 0.00, 0.01),
    ("01-02", 0.01, 0.02),
    ("02-05", 0.02, 0.05),
    ("05-10", 0.05, 0.10),
    ("10-25", 0.10, 0.25),
    ("25-50", 0.25, 0.50),
    ("50", 0.50, float("inf")),
]
OUTPUT_COLUMNS = [
    "uavsar_path",
    "flood_mask_path",
    "flight_path",
    "tile_name",
    "flood_fraction",
    "valid_pixels",
    "flood_pixels",
]
SUMMARY_COLUMNS = ["bucket", "lower_inclusive", "upper_exclusive", "tile_count"]


@dataclass(frozen=True)
class TilePair:
    uavsar_path: Path
    flood_mask_path: Path
    flight_path: str
    tile_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Florence flood-fraction bucket CSV splits."
    )
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=DEFAULT_INVENTORY_CSV,
        help="Inventory CSV with file_path, relative_path, and inferred_type columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where bucket CSVs and summary CSV will be written.",
    )
    parser.add_argument("--dataset-name", default="florence")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--flood-value", type=int, default=1)
    parser.add_argument("--nodata-value", type=int, default=255)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute bucket counts without writing CSV files.",
    )
    return parser.parse_args()


def resolve_existing_path(path_text: str, inventory_csv: Path) -> Path:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([REPO_ROOT / path, inventory_csv.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def flight_path_from_relative(relative_path: str) -> str:
    parts = Path(relative_path).parts
    return parts[0] if parts else ""


def load_pairs(inventory_csv: Path) -> list[TilePair]:
    if not inventory_csv.is_file():
        raise FileNotFoundError(f"Inventory CSV not found: {inventory_csv}")

    sar_by_key: dict[tuple[str, str], Path] = {}
    mask_by_key: dict[tuple[str, str], Path] = {}

    with inventory_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"file_path", "relative_path", "inferred_type", "filename"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Inventory CSV missing required columns: {sorted(missing)}")

        for row in reader:
            relative_path = row["relative_path"]
            flight_path = flight_path_from_relative(relative_path)
            key = (flight_path, row["filename"])
            file_path = resolve_existing_path(row["file_path"], inventory_csv)
            if row["inferred_type"] == "sar":
                sar_by_key[key] = file_path
            elif row["inferred_type"] == "flood_mask":
                mask_by_key[key] = file_path

    missing_masks = sorted(set(sar_by_key) - set(mask_by_key))
    missing_sar = sorted(set(mask_by_key) - set(sar_by_key))
    if missing_masks or missing_sar:
        example_missing_mask = missing_masks[:5]
        example_missing_sar = missing_sar[:5]
        raise ValueError(
            "Could not pair all SAR and flood mask files. "
            f"Missing masks examples: {example_missing_mask}; "
            f"missing SAR examples: {example_missing_sar}"
        )

    pairs = []
    for key in sorted(sar_by_key):
        flight_path, tile_name = key
        pairs.append(
            TilePair(
                uavsar_path=sar_by_key[key],
                flood_mask_path=mask_by_key[key],
                flight_path=flight_path,
                tile_name=tile_name,
            )
        )
    return pairs


def compute_flood_fraction(
    pair: TilePair, flood_value: int, nodata_value: int
) -> dict[str, str]:
    with rasterio.open(pair.uavsar_path) as src:
        sar = src.read()
        if sar.ndim != 3:
            raise ValueError(f"Expected SAR bands for {pair.uavsar_path}")
        sar_valid = ~(sar[: min(3, sar.shape[0])] == 0).all(axis=0)

    with rasterio.open(pair.flood_mask_path) as src:
        mask = src.read(1)
        mask_nodata = src.nodata if src.nodata is not None else nodata_value

    mask_valid = (mask != mask_nodata) & (mask != nodata_value)
    valid = sar_valid & mask_valid
    valid_pixels = int(valid.sum())
    flood_pixels = int(((mask == flood_value) & valid).sum())
    flood_fraction = flood_pixels / valid_pixels if valid_pixels else 0.0

    return {
        "uavsar_path": pair.uavsar_path.as_posix(),
        "flood_mask_path": pair.flood_mask_path.as_posix(),
        "flight_path": pair.flight_path,
        "tile_name": pair.tile_name,
        "flood_fraction": f"{flood_fraction:.8f}",
        "valid_pixels": str(valid_pixels),
        "flood_pixels": str(flood_pixels),
    }


def bucket_for_fraction(flood_fraction: float) -> str:
    for label, lower, upper in BUCKETS:
        if lower <= flood_fraction < upper:
            return label
    raise ValueError(f"Flood fraction outside bucket range: {flood_fraction}")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.inventory_csv.resolve())
    bucket_rows: dict[str, list[dict[str, str]]] = {label: [] for label, _, _ in BUCKETS}
    all_rows = []

    for index, pair in enumerate(pairs, start=1):
        row = compute_flood_fraction(pair, args.flood_value, args.nodata_value)
        label = bucket_for_fraction(float(row["flood_fraction"]))
        bucket_rows[label].append(row)
        all_rows.append(row)
        if index % 100 == 0:
            print(f"Processed {index}/{len(pairs)} tiles")

    output_dir = args.output_dir.resolve()
    prefix = f"{args.dataset_name}_{args.split_name}"

    print(f"Inventory CSV: {args.inventory_csv.resolve()}")
    print(f"Output directory: {output_dir}")
    print(f"Tile pairs: {len(pairs)}")
    print(f"Dry run: {args.dry_run}")
    print("Bucket counts:")
    for label, _, _ in BUCKETS:
        print(f"  {label}: {len(bucket_rows[label])}")

    if args.dry_run:
        return

    write_csv(output_dir / f"{prefix}_all_with_flood_fraction.csv", all_rows)
    for label, _, _ in BUCKETS:
        write_csv(output_dir / f"{prefix}_{label}.csv", bucket_rows[label])

    summary_path = output_dir / f"{prefix}_summary.csv"
    summary_rows: list[dict[str, str]] = [
        {
            "bucket": label,
            "lower_inclusive": f"{lower:.2f}",
            "upper_exclusive": "inf" if upper == float("inf") else f"{upper:.2f}",
            "tile_count": str(len(bucket_rows[label])),
        }
        for label, lower, upper in BUCKETS
    ]
    write_summary_csv(summary_path, summary_rows)
    print(f"Wrote flood-fraction CSVs to {output_dir}")


if __name__ == "__main__":
    main()
