#!/usr/bin/env python3
"""Create no-overlap bucket-targeted train/val/test splits."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "florence_tif_flood_buckets"
    / "all_tif_flood_stats.csv"
)
FALLBACK_INPUTS = [
    REPO_ROOT
    / "csv_organized_tiles"
    / "Only_PNG_Data"
    / "only_png_inventory"
    / "flood_fraction_splits"
    / "florence_test_all_with_flood_fraction.csv",
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "florence_tif_flood_buckets"
    / "all_tif_flood_stats.csv",
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "florence_png_flood_buckets"
    / "all_png_flood_stats.csv",
]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "bucket_targeted_splits"

TARGET_BUCKETS = [
    "00_01_percent",
    "01_02_percent",
    "02_05_percent",
    "05_10_percent",
    "10_25_percent",
    "25_50_percent",
    "gt_50_percent",
]
COUNT_COLUMNS = [f"count_{bucket}" for bucket in TARGET_BUCKETS]

FLOOD_BIN_MAP = {
    "0-1%": "00_01_percent",
    "0_1%": "00_01_percent",
    "00-01%": "00_01_percent",
    "00_01_percent": "00_01_percent",
    "1-2%": "01_02_percent",
    "1_2%": "01_02_percent",
    "01-02%": "01_02_percent",
    "01_02_percent": "01_02_percent",
    "2-5%": "02_05_percent",
    "2_5%": "02_05_percent",
    "02-05%": "02_05_percent",
    "02_05_percent": "02_05_percent",
    "5-10%": "05_10_percent",
    "5_10%": "05_10_percent",
    "05-10%": "05_10_percent",
    "05_10_percent": "05_10_percent",
    "10-25%": "10_25_percent",
    "10_25%": "10_25_percent",
    "10_25_percent": "10_25_percent",
    "25-50%": "25_50_percent",
    "25_50%": "25_50_percent",
    "25_50_percent": "25_50_percent",
    ">50%": "gt_50_percent",
    "gt_50_percent": "gt_50_percent",
    "50": "gt_50_percent",
    "50%+": "gt_50_percent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create no-overlap bucket-targeted train/val/test CSV splits."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-test-frac", type=float, default=0.50)
    parser.add_argument("--val-frac-of-trainval", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=1)
    return parser.parse_args()


def resolve_input_csv(path: Path) -> Path:
    candidates = [path if path.is_absolute() else REPO_ROOT / path]
    candidates.extend(FALLBACK_INPUTS)
    for candidate in candidates:
        if candidate.is_file():
            if candidate != candidates[0]:
                print(f"Input CSV not found at requested path: {candidates[0]}")
                print(f"Using fallback full flood-fraction CSV: {candidate}")
            return candidate
    raise FileNotFoundError(
        "No full flood-fraction CSV found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def normalize_flood_bin(value: str) -> str | None:
    clean = value.strip()
    if not clean:
        return None
    lowered = clean.lower()
    if lowered in {"zero", "invalid"}:
        return None
    return FLOOD_BIN_MAP.get(clean) or FLOOD_BIN_MAP.get(lowered)


def infer_bin_from_fraction(value: str) -> str | None:
    if value.strip() == "":
        return None
    fraction = float(value)
    if 0.00 <= fraction < 0.01:
        return "00_01_percent"
    if 0.01 <= fraction < 0.02:
        return "01_02_percent"
    if 0.02 <= fraction < 0.05:
        return "02_05_percent"
    if 0.05 <= fraction < 0.10:
        return "05_10_percent"
    if 0.10 <= fraction < 0.25:
        return "10_25_percent"
    if 0.25 <= fraction < 0.50:
        return "25_50_percent"
    if fraction >= 0.50:
        return "gt_50_percent"
    return None


def path_stem(value: str) -> str:
    return Path(value).stem


def infer_tile_stem(row: dict[str, str]) -> str:
    for column in ("tile_name", "mask_path", "flood_mask_path", "file_path"):
        value = row.get(column, "").strip()
        if value:
            return path_stem(value)
    value = row.get("sample_id", "").strip()
    if value:
        return value
    raise ValueError(f"Could not infer tile_stem for row: {row}")


def load_rows(input_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {input_csv}")
        original_columns = list(reader.fieldnames)
        rows = list(reader)

    if len(rows) != 1067:
        print(f"WARNING: loaded input CSV has {len(rows)} rows, expected 1067.")

    filtered = []
    excluded = 0
    for row in rows:
        if "flood_bin" in row:
            canonical = normalize_flood_bin(row.get("flood_bin", ""))
        elif "flood_fraction" in row:
            canonical = infer_bin_from_fraction(row["flood_fraction"])
        else:
            raise ValueError("Input CSV needs flood_bin or flood_fraction column.")

        if canonical is None:
            excluded += 1
            continue

        row = dict(row)
        row["tile_stem"] = infer_tile_stem(row)
        row["canonical_flood_bin"] = canonical
        filtered.append(row)

    if excluded:
        print(f"Excluded {excluded} rows with missing/zero/invalid flood_bin.")

    counts = Counter(row["canonical_flood_bin"] for row in filtered)
    for bucket in TARGET_BUCKETS:
        if counts[bucket] < 1:
            raise ValueError(f"Target bucket has no rows: {bucket}")

    return filtered, original_columns


def split_group_ids(
    group_ids: Iterable[str], fraction: float, rng: random.Random
) -> set[str]:
    group_list = sorted(set(group_ids))
    rng.shuffle(group_list)
    count = round(len(group_list) * fraction)
    if group_list and fraction > 0:
        count = max(1, count)
    count = min(count, len(group_list))
    return set(group_list[:count])


def add_split_fields(
    rows: list[dict[str, str]], target_bucket: str, split: str, seed: int
) -> list[dict[str, str]]:
    output = []
    for row in rows:
        out = dict(row)
        out["target_bucket"] = target_bucket
        out["split"] = split
        out["seed"] = str(seed)
        output.append(out)
    return output


def flood_fraction_values(rows: list[dict[str, str]]) -> list[float]:
    values = []
    for row in rows:
        value = row.get("flood_fraction", "").strip()
        if value:
            values.append(float(value))
    return values


def split_summary_row(
    target_bucket: str, split: str, rows: list[dict[str, str]]
) -> dict[str, str]:
    counts = Counter(row["canonical_flood_bin"] for row in rows)
    fractions = flood_fraction_values(rows)
    result = {
        "target_bucket": target_bucket,
        "split": split,
        "total_rows": str(len(rows)),
        "unique_tile_stems": str(len({row["tile_stem"] for row in rows})),
    }
    for bucket, column in zip(TARGET_BUCKETS, COUNT_COLUMNS, strict=True):
        result[column] = str(counts[bucket])
    result.update(
        {
            "mean_flood_fraction": f"{mean(fractions):.8f}" if fractions else "",
            "median_flood_fraction": f"{median(fractions):.8f}" if fractions else "",
            "min_flood_fraction": f"{min(fractions):.8f}" if fractions else "",
            "max_flood_fraction": f"{max(fractions):.8f}" if fractions else "",
        }
    )
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def overlap_report(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> tuple[str, bool]:
    train = {row["tile_stem"] for row in train_rows}
    val = {row["tile_stem"] for row in val_rows}
    test = {row["tile_stem"] for row in test_rows}
    tv = train & val
    tt = train & test
    vt = val & test
    passed = not tv and not tt and not vt
    lines = [
        f"train tile_stems: {len(train)}",
        f"val tile_stems: {len(val)}",
        f"test tile_stems: {len(test)}",
        f"train/val overlap count: {len(tv)}",
        f"train/test overlap count: {len(tt)}",
        f"val/test overlap count: {len(vt)}",
        "PASS" if passed else "FAIL",
    ]
    return "\n".join(lines) + "\n", passed


def create_split_for_target(
    rows: list[dict[str, str]],
    target_bucket: str,
    seed: int,
    target_test_frac: float,
    val_frac_of_trainval: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], bool]:
    rng = random.Random(f"{seed}:{target_bucket}")
    target_rows = [row for row in rows if row["canonical_flood_bin"] == target_bucket]
    target_groups = {row["tile_stem"] for row in target_rows}
    test_groups = split_group_ids(target_groups, target_test_frac, rng)

    # Assign by tile_stem across the full filtered dataset to prevent leakage.
    test_rows = [row for row in rows if row["tile_stem"] in test_groups]
    trainval_pool = [row for row in rows if row["tile_stem"] not in test_groups]

    if any(row["canonical_flood_bin"] != target_bucket for row in test_rows):
        raise ValueError(
            f"Test split for {target_bucket} contains non-target bucket rows. "
            "This indicates duplicate tile_stem values across buckets."
        )

    trainval_groups = {row["tile_stem"] for row in trainval_pool}
    val_groups = split_group_ids(trainval_groups, val_frac_of_trainval, rng)
    val_rows = [row for row in trainval_pool if row["tile_stem"] in val_groups]
    train_rows = [row for row in trainval_pool if row["tile_stem"] not in val_groups]

    if len(train_rows) + len(val_rows) + len(test_rows) != len(rows):
        raise ValueError(f"Split rows do not sum to filtered dataset size for {target_bucket}")

    target_test_rows = [row for row in test_rows if row["canonical_flood_bin"] == target_bucket]
    if len(target_test_rows) != len(test_rows):
        raise ValueError(f"Test rows for {target_bucket} are not target-only.")

    report, passed = overlap_report(train_rows, val_rows, test_rows)
    if not passed:
        raise ValueError(f"Overlap check failed for {target_bucket}\n{report}")

    return train_rows, val_rows, test_rows, passed


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_test_frac < 1.0:
        raise ValueError("--target-test-frac must be between 0 and 1")
    if not 0.0 < args.val_frac_of_trainval < 1.0:
        raise ValueError("--val-frac-of-trainval must be between 0 and 1")
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1")

    input_csv = resolve_input_csv(args.input_csv)
    rows, original_columns = load_rows(input_csv)
    output_dir = args.output_dir.resolve()
    output_columns = original_columns + [
        column
        for column in ("tile_stem", "canonical_flood_bin", "target_bucket", "split", "seed")
        if column not in original_columns
    ]
    summary_columns = [
        "target_bucket",
        "split",
        "total_rows",
        "unique_tile_stems",
        *COUNT_COLUMNS,
        "mean_flood_fraction",
        "median_flood_fraction",
        "min_flood_fraction",
        "max_flood_fraction",
    ]
    overall_columns = [
        "seed",
        "target_bucket",
        "train_rows",
        "val_rows",
        "test_rows",
        "train_unique_tile_stems",
        "val_unique_tile_stems",
        "test_unique_tile_stems",
        "target_bucket_total_rows",
        "target_bucket_test_rows",
        "target_bucket_test_fraction",
        "overlap_passed",
    ]

    filtered_size = len(rows)
    bucket_counts = Counter(row["canonical_flood_bin"] for row in rows)
    print(f"Input CSV: {input_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Filtered dataset rows: {filtered_size}")
    print("Filtered bucket counts:")
    for bucket in TARGET_BUCKETS:
        print(f"  {bucket}: {bucket_counts[bucket]}")

    for seed_offset in range(args.num_seeds):
        seed = args.seed + seed_offset
        seed_dir_name = f"seed_{seed}" if args.num_seeds == 1 else f"seed_{seed_offset}"
        seed_dir = output_dir / seed_dir_name
        overall_rows = []
        print(f"\nSeed {seed}")

        for target_bucket in TARGET_BUCKETS:
            target_dir = seed_dir / f"target_{target_bucket}"
            train_rows, val_rows, test_rows, overlap_passed = create_split_for_target(
                rows,
                target_bucket,
                seed,
                args.target_test_frac,
                args.val_frac_of_trainval,
            )

            split_rows = {
                "train": add_split_fields(train_rows, target_bucket, "train", seed),
                "val": add_split_fields(val_rows, target_bucket, "val", seed),
                "test": add_split_fields(test_rows, target_bucket, "test", seed),
            }
            write_csv(target_dir / "train.csv", output_columns, split_rows["train"])
            write_csv(target_dir / "val.csv", output_columns, split_rows["val"])
            write_csv(target_dir / "test.csv", output_columns, split_rows["test"])

            summary_rows = [
                split_summary_row(target_bucket, split, split_rows[split])
                for split in ("train", "val", "test")
            ]
            write_csv(target_dir / "split_summary.csv", summary_columns, summary_rows)

            report, _ = overlap_report(
                split_rows["train"], split_rows["val"], split_rows["test"]
            )
            (target_dir / "overlap_check.txt").write_text(report, encoding="utf-8")

            target_total = bucket_counts[target_bucket]
            target_test = sum(
                1 for row in split_rows["test"] if row["canonical_flood_bin"] == target_bucket
            )
            overall_rows.append(
                {
                    "seed": str(seed),
                    "target_bucket": target_bucket,
                    "train_rows": str(len(split_rows["train"])),
                    "val_rows": str(len(split_rows["val"])),
                    "test_rows": str(len(split_rows["test"])),
                    "train_unique_tile_stems": str(
                        len({row["tile_stem"] for row in split_rows["train"]})
                    ),
                    "val_unique_tile_stems": str(
                        len({row["tile_stem"] for row in split_rows["val"]})
                    ),
                    "test_unique_tile_stems": str(
                        len({row["tile_stem"] for row in split_rows["test"]})
                    ),
                    "target_bucket_total_rows": str(target_total),
                    "target_bucket_test_rows": str(target_test),
                    "target_bucket_test_fraction": f"{target_test / target_total:.6f}",
                    "overlap_passed": str(overlap_passed),
                }
            )

            print(
                f"  {target_bucket}: train={len(train_rows)} "
                f"val={len(val_rows)} test={len(test_rows)} "
                f"target_test={target_test}/{target_total}"
            )

        write_csv(seed_dir / "overall_summary.csv", overall_columns, overall_rows)

    print("\nEvery overlap check passed.")


if __name__ == "__main__":
    main()
