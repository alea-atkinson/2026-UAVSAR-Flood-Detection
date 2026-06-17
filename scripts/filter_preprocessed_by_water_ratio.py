#!/usr/bin/env python3
"""Filter preprocessed SAR/flood TIFF pairs by flood-mask water coverage.

Creates a symlinked dataset view with the same train/val/test folder structure
used by the TensorFlow notebooks. The filtered pairs are pooled across the
source splits and repartitioned into a fresh 70/15/15 train/val/test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class PairRecord:
    source_split: str
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


def classify_ratio(ratio: float, min_ratio: float, max_ratio: float) -> str:
    if ratio < min_ratio:
        return "below_min"
    if ratio > max_ratio:
        return "above_max"
    return "kept"


def split_counts(total: int) -> dict[str, int]:
    train = int(round(total * SPLIT_FRACTIONS["train"]))
    val = int(round(total * SPLIT_FRACTIONS["val"]))
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def output_name(record: PairRecord, kind: str) -> str:
    """Avoid rare filename collisions after pooling previous source splits."""
    digest = hashlib.sha1(str(record.sar_path.resolve()).encode("utf-8")).hexdigest()[:10]
    sar_stem = record.sar_path.stem
    if kind == "sar":
        return f"{record.source_split}_{digest}_{sar_stem}{record.sar_path.suffix}"
    flood_stem = sar_stem.replace("_prep", "_flood_prep")
    return f"{record.source_split}_{digest}_{flood_stem}{record.flood_path.suffix}"


def prepare_output_root(output_root: Path) -> None:
    if output_root.exists():
        for child in output_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def copy_metadata(source_root: Path, output_root: Path) -> None:
    source_metadata = source_root / "metadata"
    output_metadata = output_root / "metadata"
    output_metadata.mkdir(parents=True, exist_ok=True)
    for name in ["normalization_stats.json", "channel_analysis.json"]:
        source_file = source_metadata / name
        if source_file.exists():
            shutil.copy2(source_file, output_metadata / name)


def build_filtered_dataset(
    source_root: Path,
    output_root: Path,
    min_water_ratio: float,
    max_water_ratio: float,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    prepare_output_root(output_root)

    manifest: dict[str, object] = {
        "description": "SPIE paper water-ratio filtered symlink view of Preprocessed-128.",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "seed": seed,
        "split_strategy": "pool eligible pairs across source splits, shuffle, then split 70/15/15",
        "split_fractions": SPLIT_FRACTIONS,
        "quality_thresholds": {
            "min_water_ratio": min_water_ratio,
            "max_water_ratio": max_water_ratio,
        },
        "splits": {},
    }

    source_split_counts: dict[str, dict[str, int]] = {}
    eligible_records: list[PairRecord] = []
    total_source = 0
    total_rejected = 0
    total_below_min = 0
    total_above_max = 0
    all_ratio_values: list[float] = []
    eligible_ratio_values: list[float] = []

    for source_split in ["train", "val", "test"]:
        source_pairs = pair_files(source_root, source_split)
        total_source += len(source_pairs)
        rejection_counts: Counter[str] = Counter()
        ratio_values: list[float] = []

        for sar_path, flood_path in source_pairs:
            ratio = water_ratio(flood_path)
            ratio_values.append(ratio)
            all_ratio_values.append(ratio)
            bucket = classify_ratio(ratio, min_water_ratio, max_water_ratio)
            rejection_counts[bucket] += 1
            if bucket == "kept":
                eligible_records.append(PairRecord(source_split, sar_path, flood_path, ratio))
                eligible_ratio_values.append(ratio)

        total_below_min += int(rejection_counts["below_min"])
        total_above_max += int(rejection_counts["above_max"])
        total_rejected += int(rejection_counts["below_min"] + rejection_counts["above_max"])

        ratio_array = np.array(ratio_values, dtype=np.float64)
        source_split_counts[source_split] = {
            "source_matched_pairs": len(source_pairs),
            "eligible_after_filter": int(rejection_counts["kept"]),
            "rejected_below_min": int(rejection_counts["below_min"]),
            "rejected_above_max": int(rejection_counts["above_max"]),
            "water_ratio_min": float(ratio_array.min()) if ratio_array.size else None,
            "water_ratio_mean": float(ratio_array.mean()) if ratio_array.size else None,
            "water_ratio_max": float(ratio_array.max()) if ratio_array.size else None,
        }

    rng.shuffle(eligible_records)
    final_counts = split_counts(len(eligible_records))
    train_end = final_counts["train"]
    val_end = train_end + final_counts["val"]
    selected_by_split = {
        "train": eligible_records[:train_end],
        "val": eligible_records[train_end:val_end],
        "test": eligible_records[val_end:],
    }

    for split, selected in selected_by_split.items():
        selected = sorted(selected, key=lambda rec: (rec.source_split, rec.sar_path.name))
        for record in selected:
            safe_symlink(record.sar_path, output_root / split / "sar" / output_name(record, "sar"))
            safe_symlink(record.flood_path, output_root / split / "flood" / output_name(record, "flood"))

        selected_ratios = np.array([record.water_ratio for record in selected], dtype=np.float64)
        source_counts = Counter(record.source_split for record in selected)
        manifest["splits"][split] = {
            "selected_pairs": len(selected),
            "source_split_counts": dict(source_counts),
            "selected_water_ratio_summary": {
                "min": float(selected_ratios.min()) if selected_ratios.size else None,
                "mean": float(selected_ratios.mean()) if selected_ratios.size else None,
                "max": float(selected_ratios.max()) if selected_ratios.size else None,
            },
            "sample_source_sar_files": [record.sar_path.name for record in selected[:5]],
        }

    all_ratio_array = np.array(all_ratio_values, dtype=np.float64)
    eligible_ratio_array = np.array(eligible_ratio_values, dtype=np.float64)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    summary = {
        "dataset_statistics": {
            "total_input_pairs": total_source,
            "eligible_pairs_after_quality_filter": len(eligible_records),
            "total_preprocessed_pairs": len(eligible_records),
            "high_quality_pairs": len(eligible_records),
            "rejected_pairs": total_rejected,
            "rejected_below_min": total_below_min,
            "rejected_above_max": total_above_max,
        },
        "final_splits": final_counts,
        "preprocessing_settings": {
            "tile_size": "128x128",
            "normalization": "log_transform_standardization",
            "speckle_filter": "source Preprocessed-128 metadata: gaussian_sigma_0.5; source filenames indicate refined_lee_standard",
            "quality_thresholds": {
                "min_water_ratio": min_water_ratio,
                "max_water_ratio": max_water_ratio,
            },
        },
        "water_ratio_summary": {
            "source": {
                "min": float(all_ratio_array.min()) if all_ratio_array.size else None,
                "mean": float(all_ratio_array.mean()) if all_ratio_array.size else None,
                "max": float(all_ratio_array.max()) if all_ratio_array.size else None,
            },
            "eligible": {
                "min": float(eligible_ratio_array.min()) if eligible_ratio_array.size else None,
                "mean": float(eligible_ratio_array.mean()) if eligible_ratio_array.size else None,
                "max": float(eligible_ratio_array.max()) if eligible_ratio_array.size else None,
            },
        },
        "source_splits": source_split_counts,
        "paper_reference": {
            "split": "70/15/15",
            "min_water_ratio": 0.05,
            "max_water_ratio": 0.85,
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
    copy_metadata(source_root, output_root)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("spie/Preprocessed-128"))
    parser.add_argument("--output-root", type=Path, default=Path("spie/Preprocessed-128-paper-filtered"))
    parser.add_argument("--min-water-ratio", type=float, default=0.05)
    parser.add_argument("--max-water-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_filtered_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        min_water_ratio=args.min_water_ratio,
        max_water_ratio=args.max_water_ratio,
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
