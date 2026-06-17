#!/usr/bin/env python3
"""Validate and summarize the available paper-style SPIE rerun dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def paired_flood_path(sar_path: Path, flood_dir: Path) -> Path | None:
    flood_name = sar_path.name.replace("_prep.tif", "_flood_prep.tif")
    candidate = flood_dir / flood_name
    if candidate.exists():
        return candidate
    return None


def water_ratio(mask_path: Path) -> float:
    with rasterio.open(mask_path) as src:
        mask = src.read(1, out_dtype="float32")
    valid = np.isfinite(mask)
    if not valid.any():
        return 0.0
    return float(np.mean(mask[valid] > 0.5))


def summarize_split(dataset_root: Path, split: str) -> dict[str, object]:
    sar_dir = dataset_root / split / "sar"
    flood_dir = dataset_root / split / "flood"
    sar_files = sorted(sar_dir.glob("*.tif"))
    flood_files = sorted(flood_dir.glob("*.tif"))

    matched: list[tuple[Path, Path]] = []
    unmatched_sar: list[str] = []
    for sar_path in sar_files:
        flood_path = paired_flood_path(sar_path, flood_dir)
        if flood_path is None:
            unmatched_sar.append(sar_path.name)
        else:
            matched.append((sar_path, flood_path))

    matched_flood_names = {flood_path.name for _, flood_path in matched}
    unmatched_flood = [path.name for path in flood_files if path.name not in matched_flood_names]
    ratios = np.array([water_ratio(flood_path) for _, flood_path in matched], dtype=np.float64)

    return {
        "sar_files": len(sar_files),
        "flood_files": len(flood_files),
        "matched_pairs": len(matched),
        "matching_status": "ok" if not unmatched_sar and not unmatched_flood else "mismatch",
        "unmatched_sar_files": unmatched_sar,
        "unmatched_flood_files": unmatched_flood,
        "water_ratio": {
            "min": float(ratios.min()) if ratios.size else None,
            "mean": float(ratios.mean()) if ratios.size else None,
            "median": float(np.median(ratios)) if ratios.size else None,
            "max": float(ratios.max()) if ratios.size else None,
        },
    }


def load_source_count(source_root: Path) -> int:
    total = 0
    for split in ("train", "val", "test"):
        total += len(list((source_root / split / "sar").glob("*.tif")))
    return total


def build_report(source_root: Path, dataset_root: Path) -> dict[str, object]:
    splits = {split: summarize_split(dataset_root, split) for split in ("train", "val", "test")}
    all_ratios: list[float] = []
    for split in splits.values():
        ratio_summary = split["water_ratio"]
        # Re-read per split in the inexpensive explicit loop below for exact global median.
        del ratio_summary

    for split_name in ("train", "val", "test"):
        flood_dir = dataset_root / split_name / "flood"
        for sar_path in sorted((dataset_root / split_name / "sar").glob("*.tif")):
            flood_path = paired_flood_path(sar_path, flood_dir)
            if flood_path is not None:
                all_ratios.append(water_ratio(flood_path))

    ratios = np.array(all_ratios, dtype=np.float64)
    total_filtered = int(sum(split["matched_pairs"] for split in splits.values()))
    all_ok = all(split["matching_status"] == "ok" for split in splits.values())

    return {
        "rerun_label": "paper-style rerun on available Preprocessed-128 dataset after water filtering",
        "dataset_note": (
            "The exact curated 3,873-tile SPIE dataset is unavailable in this repo. "
            "This dataset starts from the available Preprocessed-128 pairs and applies "
            "the paper water-ratio rule only: 0.05 <= water_ratio <= 0.85."
        ),
        "source_root": str(source_root.resolve()),
        "filtered_dataset_root": str(dataset_root.resolve()),
        "filter": {
            "water_ratio_min_inclusive": 0.05,
            "water_ratio_max_inclusive": 0.85,
        },
        "original_tile_pair_count": load_source_count(source_root),
        "filtered_tile_pair_count": total_filtered,
        "split_counts": {name: int(split["matched_pairs"]) for name, split in splits.items()},
        "sar_flood_matching_status": "ok" if all_ok else "mismatch",
        "splits": splits,
        "water_ratio_summary": {
            "min": float(ratios.min()) if ratios.size else None,
            "mean": float(ratios.mean()) if ratios.size else None,
            "median": float(np.median(ratios)) if ratios.size else None,
            "max": float(ratios.max()) if ratios.size else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("spie/Preprocessed-128"))
    parser.add_argument("--dataset-root", type=Path, default=Path("spie/Preprocessed-128-paper-filtered"))
    parser.add_argument("--output", type=Path, default=Path("spie/paper_style_rerun/dataset_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.source_root, args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "original_tile_pair_count": report["original_tile_pair_count"],
        "filtered_tile_pair_count": report["filtered_tile_pair_count"],
        "split_counts": report["split_counts"],
        "sar_flood_matching_status": report["sar_flood_matching_status"],
        "water_ratio_summary": report["water_ratio_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
