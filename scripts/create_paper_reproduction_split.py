#!/usr/bin/env python3
"""Create the 3,873-pair SPIE paper reproduction split for Preprocessed-128.

The generated dataset is a symlink view over spie/spie/Preprocessed-128 so the
existing notebooks can use the same train/val/test folder loader without copying
large TIFF files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

PAPER_COUNTS = {"train": 2711, "val": 581, "test": 581}


def pair_files(split_dir: Path) -> list[tuple[Path, Path]]:
    sar_dir = split_dir / "sar"
    flood_dir = split_dir / "flood"
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


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel_src)


def build_split(source_root: Path, output_root: Path, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "description": "SPIE paper reproduction split generated as symlinks from Preprocessed-128.",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "seed": seed,
        "target_total_pairs": sum(PAPER_COUNTS.values()),
        "split_counts": PAPER_COUNTS,
        "splits": {},
    }

    for split, count in PAPER_COUNTS.items():
        pairs = pair_files(source_root / split)
        if len(pairs) < count:
            raise ValueError(f"Not enough matched pairs in {source_root / split}: {len(pairs)} < {count}")

        selected = sorted(rng.sample(pairs, count), key=lambda pair: pair[0].name)
        sar_out = output_root / split / "sar"
        flood_out = output_root / split / "flood"
        sar_out.mkdir(parents=True, exist_ok=True)
        flood_out.mkdir(parents=True, exist_ok=True)

        for sar_path, flood_path in selected:
            safe_symlink(sar_path, sar_out / sar_path.name)
            safe_symlink(flood_path, flood_out / flood_path.name)

        manifest["splits"][split] = {
            "source_matched_pairs": len(pairs),
            "selected_pairs": len(selected),
            "sample_sar_files": [pair[0].name for pair in selected[:5]],
            "sample_flood_files": [pair[1].name for pair in selected[:5]],
        }

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    summary = {
        "dataset_statistics": {
            "total_input_pairs": sum(PAPER_COUNTS.values()),
            "total_preprocessed_pairs": sum(PAPER_COUNTS.values()),
            "high_quality_pairs": sum(PAPER_COUNTS.values()),
            "rejected_pairs": None,
        },
        "final_splits": PAPER_COUNTS,
        "preprocessing_settings": {
            "tile_size": "128x128",
            "normalization": "log_transform_standardization",
            "speckle_filter": "gaussian_sigma_0.5",
            "quality_thresholds": {
                "min_water_ratio": 0.005,
                "max_water_ratio": 0.85,
            },
        },
        "paper_reference": {
            "total_pairs": 3873,
            "split": "70/15/15",
            "batch_size": 8,
            "learning_rate": 0.001,
            "dropout": 0.1,
            "epochs": 80,
        },
    }
    (metadata_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_root / "split_organization.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("spie/spie/Preprocessed-128"))
    parser.add_argument("--output-root", type=Path, default=Path("spie/spie/Preprocessed-128-paper"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_split(args.source_root, args.output_root, args.seed)
    print(json.dumps({"output_root": str(args.output_root), "split_counts": manifest["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
