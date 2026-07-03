#!/usr/bin/env python3
"""Inventory files in the Only_PNG_data tile folder."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "only_png_inventory"
FLOOD_FRACTION_MASKS_CSV = REPO_ROOT / "csv_organized_files" / "flood_fraction_masks.csv"

ALL_FILES_COLUMNS = [
    "file_path",
    "relative_path",
    "parent_dir",
    "filename",
    "stem",
    "suffix",
    "size_bytes",
    "is_png",
    "inferred_type",
]

PNG_COLUMNS = [
    "file_path",
    "relative_path",
    "parent_dir",
    "filename",
    "stem",
    "size_bytes",
    "width",
    "height",
    "mode",
    "num_channels",
    "inferred_type",
    "unique_values_sample",
    "read_error",
]

SUMMARY_COLUMNS = [
    "inferred_type",
    "file_count",
    "png_count",
    "mean_width",
    "mean_height",
    "example_file",
]

FLOOD_MASK_COLUMNS = [
    "file_path",
    "relative_path",
    "filename",
    "width",
    "height",
    "mode",
    "unique_values_sample",
    "read_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory files under Only_PNG_data / Only_PNG_Data."
    )
    parser.add_argument(
        "--png-data-dir",
        type=Path,
        default=None,
        help="Directory to inventory. Defaults to the discovered Only_PNG_data folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for inventory CSVs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and print summary without writing CSV files.",
    )
    return parser.parse_args()


def discover_png_data_dir() -> Path:
    likely_paths = [
        REPO_ROOT / "2025_Tile_Data" / "Only_PNG_data",
        REPO_ROOT / "2025_Tile_Data" / "Only_PNG_Data",
        REPO_ROOT / "csv_organized_tiles" / "Only_PNG_data",
        REPO_ROOT / "csv_organized_tiles" / "Only_PNG_Data",
    ]
    for path in likely_paths:
        if path.is_dir():
            return path

    exact_matches = []
    case_insensitive_matches = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_dir():
            continue
        if path.name == "Only_PNG_data":
            exact_matches.append(path)
        elif path.name.lower() == "only_png_data":
            case_insensitive_matches.append(path)

    matches = exact_matches or case_insensitive_matches
    if not matches:
        raise FileNotFoundError(
            "Could not find a directory named Only_PNG_data or Only_PNG_Data."
        )

    preferred = [
        path
        for path in matches
        if "2025" in path.as_posix() or "csv_organized_tiles" in path.as_posix()
    ]
    return sorted(preferred or matches, key=lambda item: item.as_posix())[0]


def immediate_subfolder_structure(root: Path) -> list[str]:
    lines = [root.name]
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        lines.append(f"  {child.name}")
        for grandchild in sorted(child.iterdir(), key=lambda item: item.name.lower()):
            if grandchild.is_dir():
                lines.append(f"    {grandchild.name}")
    return lines


def token_match(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    padded = f"_{normalized}_"
    for term in terms:
        if re.search(rf"(^|[^\w]){re.escape(term)}($|[^\w])", normalized):
            return True
        if f"_{term}_" in padded:
            return True
    return False


def infer_type(path_text: str | Path) -> str:
    text = Path(path_text).as_posix().lower()
    compact = text.replace("-", "_").replace(" ", "_")

    if any(term in compact for term in ("landcover", "land_cover", "nlcd")):
        return "land_cover"
    if token_match(compact, ("lcc",)):
        return "land_cover"
    if any(term in compact for term in ("flood", "change", "ground_truth")):
        return "flood_mask"
    if token_match(compact, ("mask", "cd", "label")):
        return "flood_mask"
    if any(term in compact for term in ("sar", "uavsar", "image", "input", "rgb")):
        return "sar"
    if token_match(compact, ("hh", "hv", "vv", "vh")):
        return "sar"
    return "unknown"


def is_png(path: Path) -> bool:
    return path.suffix.lower() == ".png"


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def all_file_row(path: Path, root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(root)
    return {
        "file_path": path.resolve().as_posix(),
        "relative_path": relative_path.as_posix(),
        "parent_dir": path.parent.as_posix(),
        "filename": path.name,
        "stem": path.stem,
        "suffix": path.suffix,
        "size_bytes": path.stat().st_size,
        "is_png": is_png(path),
        "inferred_type": infer_type(relative_path),
    }


def num_channels_for_image(image: Image.Image) -> int:
    return len(image.getbands())


def unique_values_sample(image: Image.Image, max_unique: int = 256) -> str:
    if len(image.getbands()) != 1:
        pixel_count = image.width * image.height
        if pixel_count > 10_000:
            return ""
        values = set(image.getdata())
        if len(values) > 64:
            return ""
        return repr(sorted(values))

    values = set()
    for value in image.getdata():
        values.add(value)
        if len(values) > max_unique:
            return f">{max_unique} unique values"
    return repr(sorted(values))


def png_row(path: Path, root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(root)
    row: dict[str, Any] = {
        "file_path": path.resolve().as_posix(),
        "relative_path": relative_path.as_posix(),
        "parent_dir": path.parent.as_posix(),
        "filename": path.name,
        "stem": path.stem,
        "size_bytes": path.stat().st_size,
        "width": "",
        "height": "",
        "mode": "",
        "num_channels": "",
        "inferred_type": infer_type(relative_path),
        "unique_values_sample": "",
        "read_error": "",
    }
    try:
        with Image.open(path) as image:
            row["width"] = image.width
            row["height"] = image.height
            row["mode"] = image.mode
            row["num_channels"] = num_channels_for_image(image)
            row["unique_values_sample"] = unique_values_sample(image)
    except Exception as exc:  # noqa: BLE001 - inventory should continue after bad files.
        row["read_error"] = f"{type(exc).__name__}: {exc}"
    return row


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    all_rows: list[dict[str, Any]], png_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    file_counts = Counter(row["inferred_type"] for row in all_rows)
    png_counts = Counter(row["inferred_type"] for row in png_rows)
    examples: dict[str, str] = {}
    dimensions: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for row in all_rows:
        examples.setdefault(row["inferred_type"], row["file_path"])
    for row in png_rows:
        if row["read_error"] or row["width"] == "" or row["height"] == "":
            continue
        dimensions[row["inferred_type"]].append((int(row["width"]), int(row["height"])))

    rows = []
    for inferred_type in sorted(file_counts):
        dims = dimensions[inferred_type]
        rows.append(
            {
                "inferred_type": inferred_type,
                "file_count": file_counts[inferred_type],
                "png_count": png_counts[inferred_type],
                "mean_width": f"{mean(width for width, _ in dims):.2f}" if dims else "",
                "mean_height": f"{mean(height for _, height in dims):.2f}" if dims else "",
                "example_file": examples.get(inferred_type, ""),
            }
        )
    return rows


def print_summary(
    root: Path,
    output_dir: Path,
    all_rows: list[dict[str, Any]],
    png_rows: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    print(f"Only_PNG_data path found: {root.resolve()}")
    print("Immediate subfolder structure:")
    for line in immediate_subfolder_structure(root):
        print(line)
    print()
    print(f"Dry run: {dry_run}")
    print(f"Inventory output directory: {output_dir.resolve()}")
    print(f"Flood mask CSV path: {FLOOD_FRACTION_MASKS_CSV.resolve()}")
    print(f"Total file count: {len(all_rows)}")
    print(f"Total PNG count: {len(png_rows)}")

    counts = Counter(row["inferred_type"] for row in all_rows)
    print("Count by inferred_type:")
    for inferred_type, count in sorted(counts.items()):
        print(f"  {inferred_type}: {count}")

    print("Example files by inferred_type:")
    examples: dict[str, list[str]] = defaultdict(list)
    for row in all_rows:
        if len(examples[row["inferred_type"]]) < 3:
            examples[row["inferred_type"]].append(row["relative_path"])
    for inferred_type in sorted(examples):
        print(f"  {inferred_type}:")
        for example in examples[inferred_type]:
            print(f"    {example}")

    print("Unique value samples for first 10 potential flood masks:")
    flood_pngs = [row for row in png_rows if row["inferred_type"] == "flood_mask"]
    if not flood_pngs:
        print("  No PNG flood masks found.")
    for row in flood_pngs[:10]:
        sample = row["unique_values_sample"] or "(not sampled)"
        error = f" read_error={row['read_error']}" if row["read_error"] else ""
        print(f"  {row['relative_path']}: {sample}{error}")


def main() -> None:
    args = parse_args()
    png_data_dir = args.png_data_dir.resolve() if args.png_data_dir else discover_png_data_dir()
    if not png_data_dir.exists() or not png_data_dir.is_dir():
        raise NotADirectoryError(f"PNG data directory does not exist: {png_data_dir}")

    files = sorted(path for path in png_data_dir.rglob("*") if path.is_file())
    all_rows = [all_file_row(path, png_data_dir) for path in files]
    png_rows = [png_row(path, png_data_dir) for path in files if is_png(path)]
    summary_rows = build_summary(all_rows, png_rows)
    flood_mask_rows = [
        {
            "file_path": row["file_path"],
            "relative_path": row["relative_path"],
            "filename": row["filename"],
            "width": row["width"],
            "height": row["height"],
            "mode": row["mode"],
            "unique_values_sample": row["unique_values_sample"],
            "read_error": row["read_error"],
        }
        for row in png_rows
        if row["inferred_type"] == "flood_mask"
    ]

    output_dir = args.output_dir.resolve()
    print_summary(png_data_dir, output_dir, all_rows, png_rows, args.dry_run)

    if args.dry_run:
        return

    write_csv(output_dir / "all_files_inventory.csv", ALL_FILES_COLUMNS, all_rows)
    write_csv(output_dir / "png_files_inventory.csv", PNG_COLUMNS, png_rows)
    write_csv(output_dir / "inventory_summary.csv", SUMMARY_COLUMNS, summary_rows)
    write_csv(FLOOD_FRACTION_MASKS_CSV, FLOOD_MASK_COLUMNS, flood_mask_rows)


if __name__ == "__main__":
    main()
