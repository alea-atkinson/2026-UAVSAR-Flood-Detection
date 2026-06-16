#!/usr/bin/env python3
"""Filter preprocessed SAR/flood TIFF pairs by flood-mask water coverage.

Creates a symlinked dataset view with the same train/val/test folder structure
used by the TensorFlow notebooks. By default it targets the SPIE paper settings:
5%-85% water coverage and 3,873 total pairs split 70/15/15.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

DEFAULT_TARGET_COUNTS = {"train": 2711, "val": 581, "test": 581}


@dataclass(frozen=True)
class PairRecord:
    split: str
    sar_path: Path
    flood_path: Path
    water_ratio: float


def pair_files(source_root: Path, split: str) -> list[tuple[Path, Path]]:
    sar_dir = source_root / split / "sar"
    flood_dir = source_root / split / "flood"
    flood_map = {
        path.stem.replace("_flood_prep", "_prep"): path
        for path in flood_dir.glob("*.tif")
    }
    pairs = []
    for sar_path in sorted(sar_dir.glob("*.tif")):
        flood_path = flood_map.get(sar_path.stem)
        if flood_path is not None:
            pairs.append((sar_path, flood_path))
    return pairs


def water_ratio(mask_path: Path) -> float:
    with rasterio.open(mask_path) as src:
        mask = src.read(1, out_dtype="float32")
    finite = np.isfinite(mask)
    if not finite.any():
        return 0.0
    return float(np.mean(mask[finite] > 0.5))


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel_src)


def parse_target_counts(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target counts must be train,val,test")
    train, val, test = (int(part) for part in parts)
    return {"train": train, "val": val, "test": test}


def classify_ratio(ratio: float, min_ratio: float, max_ratio: float) -> str:
    if ratio < min_ratio:
        return "below_min"
    if ratio > max_ratio:
        return "above_max"
    return "kept"


def build_filtered_dataset(
    source_root: Path,
    output_root: Path,
    min_water_ratio: float,
    max_water_ratio: float,
    target_counts: dict[str, int] | None,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "description": "Water-ratio filtered symlink view of Preprocessed-128.",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "seed": seed,
        "quality_thresholds": {
            "min_water_ratio": min_water_ratio,
            "max_water_ratio": max_water_ratio,
        },
        "target_counts": target_counts,
        "splits": {},
    }

    final_counts: dict[str, int] = {}
    total_source = 0
    total_kept_available = 0
    total_selected = 0
    total_rejected = 0

    for split in ["train", "val", "test"]:
        source_pairs = pair_files(source_root, split)
        total_source += len(source_pairs)
        records: list[PairRecord] = []
        rejection_counts: Counter[str] = Counter()
        ratio_values: list[float] = []

        for sar_path, flood_path in source_pairs:
            ratio = water_ratio(flood_path)
            ratio_values.append(ratio)
            bucket = classify_ratio(ratio, min_water_ratio, max_water_ratio)
            rejection_counts[bucket] += 1
            if bucket == "kept":
                records.append(PairRecord(split, sar_path, flood_path, ratio))

        total_kept_available += len(records)
        target_count = target_counts[split] if target_counts else len(records)
        if len(records) < target_count:
            raise ValueError(
                f"Not enough eligible {split} pairs after filtering: "
                f"{len(records)} available, {target_count} requested"
            )

        selected = rng.sample(records, target_count) if target_counts else records
        selected = sorted(selected, key=lambda rec: rec.sar_path.name)
        final_counts[split] = len(selected)
        total_selected += len(selected)
        total_rejected += len(source_pairs) - len(records)

        for record in selected:
            safe_symlink(record.sar_path, output_root / split / "sar" / record.sar_path.name)
            safe_symlink(record.flood_path, output_root / split / "flood" / record.flood_path.name)

        ratio_array = np.array(ratio_values, dtype=np.float64)
        selected_ratios = np.array([record.water_ratio for record in selected], dtype=np.float64)
        manifest["splits"][split] = {
            "source_matched_pairs": len(source_pairs),
            "eligible_after_filter": len(records),
            "selected_pairs": len(selected),
            "rejected_below_min": int(rejection_counts["below_min"]),
            "rejected_above_max": int(rejection_counts["above_max"]),
            "source_water_ratio_summary": {
                "min": float(ratio_array.min()) if ratio_array.size else None,
                "mean": float(ratio_array.mean()) if ratio_array.size else None,
                "max": float(ratio_array.max()) if ratio_array.size else None,
            },
            "selected_water_ratio_summary": {
                "min": float(selected_ratios.min()) if selected_ratios.size else None,
                "mean": float(selected_ratios.mean()) if selected_ratios.size else None,
                "max": float(selected_ratios.max()) if selected_ratios.size else None,
            },
            "sample_sar_files": [record.sar_path.name for record in selected[:5]],
        }

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    summary = {
        "dataset_statistics": {
            "total_input_pairs": total_source,
            "eligible_pairs_after_quality_filter": total_kept_available,
            "total_preprocessed_pairs": total_selected,
            "high_quality_pairs": total_selected,
            "rejected_pairs": total_rejected,
        },
        "final_splits": final_counts,
        "preprocessing_settings": {
            "tile_size": "128x128",
            "normalization": "log_transform_standardization",
            "speckle_filter": "gaussian_sigma_0.5",
            "quality_thresholds": {
                "min_water_ratio": min_water_ratio,
                "max_water_ratio": max_water_ratio,
            },
        },
        "paper_reference": {
            "total_pairs": 3873 if target_counts == DEFAULT_TARGET_COUNTS else None,
            "split": "70/15/15" if target_counts == DEFAULT_TARGET_COUNTS else None,
            "batch_size": 8,
            "learning_rate": 0.001,
            "dropout": 0.1,
            "epochs": 80,
        },
    }
    manifest["final_splits"] = final_counts
    manifest["dataset_statistics"] = summary["dataset_statistics"]

    (output_root / "split_organization.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (metadata_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("spie/spie/Preprocessed-128"))
    parser.add_argument("--output-root", type=Path, default=Path("spie/spie/Preprocessed-128-paper-filtered"))
    parser.add_argument("--min-water-ratio", type=float, default=0.05)
    parser.add_argument("--max-water-ratio", type=float, default=0.85)
    parser.add_argument("--target-counts", default="2711,581,581", help="train,val,test counts; use empty string to keep all eligible")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_counts = parse_target_counts(args.target_counts)
    manifest = build_filtered_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        min_water_ratio=args.min_water_ratio,
        max_water_ratio=args.max_water_ratio,
        target_counts=target_counts,
        seed=args.seed,
    )
    print(json.dumps({
        "output_root": str(args.output_root),
        "quality_thresholds": manifest["quality_thresholds"],
        "final_splits": manifest["final_splits"],
        "dataset_statistics": manifest["dataset_statistics"],
    }, indent=2))


if __name__ == "__main__":
    main()
