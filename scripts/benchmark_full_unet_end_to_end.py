#!/usr/bin/env python3
"""
benchmark_full_unet_end_to_end.py

End-to-end PyTorch inference THROUGHPUT benchmark for the trained UAVSAR
flood-segmentation U-Net: reading real UAVSAR tiles from disk with
rasterio, selecting the first 3 SAR bands, per-tile normalization,
batching via a DataLoader, host-to-device transfer, and the full U-Net
forward pass -- ALL inside the timed region.

This is a DELIBERATE COMPANION to scripts/benchmark_full_unet_inference.py,
NOT a replacement. That script measures MODEL FORWARD PASS ONLY --
synthetic or real tensors are constructed/loaded and moved to the target
device BEFORE any timing starts, so its numbers exclude disk I/O,
normalization, and host-to-device transfer entirely. This script measures
the OPPOSITE thing on purpose: practical, end-to-end dataset-pass
throughput, including every step a real inference pipeline would actually
pay for. The two scripts answer different questions and neither number
should be quoted as if it were the other.

---- Sources inspected before writing this script -----------------------
- scripts/benchmark_full_unet_inference.py: the `DoubleConv`/`UNet` class
  definitions (copied verbatim below, unchanged, for the same reason
  documented there -- train_unet_baseline_tuned.py has a module-level
  `--help`-argv check that would fire incorrectly if imported directly),
  the checkpoint-loading convention, and the CSV-merge pattern used to
  avoid one run's results silently overwriting another's.
- scripts/train_unet_baseline_tuned.py: `FloodTileDataset._normalize_per_tile`
  (reused verbatim below) and the checkpoint's saved `base_channels`
  argument.
- models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt:
  confirmed to store `base_channels=32` under its saved `args` dict.

---- Timing scope (IMPORTANT) --------------------------------------------
Every reported timing INCLUDES:
  - rasterio disk reads of the SAR tile file
  - selecting the first 3 bands
  - per-tile normalization (percentile-clip + zero-mean/unit-variance)
  - batching (via DataLoader, optionally multi-worker)
  - host-to-device transfer
  - the full U-Net forward pass under torch.no_grad()
  - (optional, only if --apply-sigmoid-threshold is passed) a cheap
    sigmoid + 0.5 threshold, clearly flagged in the notes column when used
It does NOT compute Dice/IoU/recall or any other accuracy metric, and it
does NOT estimate or claim any FPGA timing.

---- Usage ----------------------------------------------------------------
    python scripts/benchmark_full_unet_end_to_end.py \\
        --device cuda --batch-sizes 1 8 --num-workers-list 0 2 \\
        --passes 3 --max-tiles 64

    python scripts/benchmark_full_unet_end_to_end.py \\
        --device cpu --batch-sizes 1 8 --num-workers-list 0 2 \\
        --passes 2 --max-tiles 32

---- Outputs --------------------------------------------------------------
outputs/hardware_benchmarks/full_unet_end_to_end_timing/full_unet_end_to_end_timing.csv
outputs/hardware_benchmarks/full_unet_end_to_end_timing/full_unet_end_to_end_timing_summary.md
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
DEFAULT_SPLIT_CSV = (
    REPO_ROOT / "csv_splits" / "flood_splits_ieee_png_filtered_standard_strict_train_val"
    / "strict_no_overlap" / "heldout_fp2_test.csv"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "2025_Tile_Data"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "hardware_benchmarks" / "full_unet_end_to_end_timing"


# ---------------------------------------------------------------------------
# Model -- DoubleConv/UNet copied VERBATIM from scripts/benchmark_full_unet_inference.py
# (itself copied verbatim from train_unet_baseline_tuned.py; see that
# script's docstring for why this is a copy, not an import).
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small, plain U-Net for binary segmentation. Identical to the class of
    the same name in train_unet_baseline_tuned.py."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        x = self.bottleneck(self.pool(enc4))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, enc4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, enc3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, enc2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, enc1], dim=1))
        return self.out(x)


# ---------------------------------------------------------------------------
# Normalization -- reused verbatim from FloodTileDataset._normalize_per_tile
# in train_unet_baseline_tuned.py.
# ---------------------------------------------------------------------------
def normalize_per_tile(sar: np.ndarray) -> np.ndarray:
    sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid = sar[np.isfinite(sar)]
    if valid.size == 0:
        return np.zeros_like(sar, dtype=np.float32)
    low, high = np.percentile(valid, [1.0, 99.0])
    if high > low:
        sar = np.clip(sar, low, high)
    mean = float(sar.mean())
    std = float(sar.std())
    if std < 1e-6:
        return np.zeros_like(sar, dtype=np.float32)
    return ((sar - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset -- SAR tile ONLY (no mask needed; this benchmarks inference
# throughput, not accuracy). Disk read + band selection + normalization
# happen inside __getitem__, i.e. INSIDE the timed region whenever this
# Dataset is iterated through a DataLoader during a timed pass.
# ---------------------------------------------------------------------------
class SARTileOnlyDataset(Dataset):
    def __init__(self, rows: list[dict], data_root: pathlib.Path) -> None:
        self.rows = rows
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.rows[index]
        tile_path = self.data_root / row["uavsar_path"]
        with rasterio.open(tile_path) as src:
            sar = src.read(out_dtype="float32")
        sar = sar[:3]
        sar = normalize_per_tile(sar)
        return torch.from_numpy(sar)


def build_usable_rows(split_csv: pathlib.Path, data_root: pathlib.Path, max_tiles: int) -> list[dict]:
    """Preflight ONLY: filters split rows to those whose tile file exists on
    disk, up to max_tiles (or all rows if max_tiles <= 0). This is a cheap
    Path.exists() check, NOT a file read -- the actual rasterio read/
    normalize/transfer/forward cost for each of these rows is paid fresh,
    every pass, inside the timed DataLoader loop below. Done once, shared
    across every device/batch-size/num-workers config so all configs
    benchmark the exact same tile set."""
    if not split_csv.exists():
        raise SystemExit(f"ERROR: split CSV not found at {split_csv}")
    if not data_root.exists():
        raise SystemExit(f"ERROR: data root not found at {data_root}")

    with open(split_csv, newline="") as f:
        all_rows = list(csv.DictReader(f))

    limit = len(all_rows) if max_tiles <= 0 else min(max_tiles, len(all_rows))
    usable = []
    for row in all_rows:
        if len(usable) >= limit:
            break
        if (data_root / row["uavsar_path"]).exists():
            usable.append(row)
    return usable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end PyTorch inference throughput benchmark "
        "(disk read + normalization + batching + host-to-device transfer "
        "+ full U-Net forward pass), unlike benchmark_full_unet_inference.py "
        "which measures model forward pass only."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-csv", type=pathlib.Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8],
                         help="Batch sizes to benchmark in this single invocation (default: 1 8).")
    parser.add_argument("--num-workers-list", type=int, nargs="+", default=[0, 2],
                         help="DataLoader num_workers values to benchmark (default: 0 2).")
    parser.add_argument("--passes", type=int, default=3,
                         help="Number of full passes over the (possibly capped) split "
                         "per device/batch-size/num-workers config (default: 3).")
    parser.add_argument("--max-tiles", type=int, default=64,
                         help="Cap on tiles used per pass, taken from the start of the "
                         "split CSV. <= 0 means use all available tiles in the split "
                         "(default: 64).")
    parser.add_argument("--cpu-time-budget-sec", type=float, default=120.0,
                         help="On CPU only: if the requested --passes would project "
                         "(from the first completed pass) to longer than this many "
                         "seconds for a given config, remaining passes for that config "
                         "are skipped and the cap is reported in notes (default: 120.0).")
    parser.add_argument("--apply-sigmoid-threshold", action="store_true", default=False,
                         help="If set, apply a cheap sigmoid + 0.5 threshold to the "
                         "model output inside the timed region (default: off). This "
                         "is flagged explicitly in the notes column when used.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading -- same convention as benchmark_full_unet_inference.py /
# train_unet_baseline_tuned.py.
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: pathlib.Path, device: torch.device) -> UNet:
    print(f"[1] Loading checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"ERROR: checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    base_channels = 32
    saved_args = checkpoint.get("args")
    if isinstance(saved_args, dict) and "base_channels" in saved_args:
        base_channels = int(saved_args["base_channels"])
    print(f"    base_channels (from checkpoint's saved args, default 32 if absent): {base_channels}")

    model = UNet(in_channels=3, out_channels=1, base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Timed end-to-end pass: disk read -> band select -> normalize -> batch ->
# host-to-device -> forward (-> optional sigmoid+threshold), ALL inside the
# timed region, once per DataLoader batch, for one full pass over the
# dataset.
# ---------------------------------------------------------------------------
def time_one_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    apply_sigmoid_threshold: bool,
) -> tuple[float, int]:
    """Returns (elapsed_sec, num_tiles_processed_this_pass)."""
    pin = device.type == "cuda"
    n_processed = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=pin)
            out = model(batch)
            if apply_sigmoid_threshold:
                _ = torch.sigmoid(out) > 0.5
            n_processed += batch.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0), n_processed


def run_one_config(
    model: nn.Module,
    rows: list[dict],
    data_root: pathlib.Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    passes: int,
    cpu_time_budget_sec: float,
    apply_sigmoid_threshold: bool,
) -> dict[str, object]:
    dataset = SARTileOnlyDataset(rows, data_root)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    pass_elapsed_sec: list[float] = []
    num_tiles_processed = 0
    cap_note = ""
    requested_passes = passes

    for pass_idx in range(passes):
        elapsed, n_processed = time_one_pass(model, loader, device, apply_sigmoid_threshold)
        pass_elapsed_sec.append(elapsed)
        num_tiles_processed = n_processed  # same every pass (no shuffling, fixed dataset)

        if device.type == "cpu":
            projected_total = elapsed * requested_passes
            if projected_total > cpu_time_budget_sec and (pass_idx + 1) < requested_passes:
                cap_note = (
                    f"CPU timing capped: requested --passes={requested_passes} projected "
                    f"to ~{projected_total:.1f}s (> {cpu_time_budget_sec:.0f}s budget) "
                    f"based on the first pass; stopped after {pass_idx + 1} pass(es)."
                )
                print(f"    NOTE: {cap_note}")
                break

    total_elapsed_sec = sum(pass_elapsed_sec)
    actual_passes = len(pass_elapsed_sec)
    mean_pass_sec = total_elapsed_sec / actual_passes if actual_passes else float("nan")
    total_tiles_all_passes = num_tiles_processed * actual_passes
    tiles_per_sec = total_tiles_all_passes / total_elapsed_sec if total_elapsed_sec > 0 else float("nan")
    ms_per_tile = (total_elapsed_sec * 1000.0 / total_tiles_all_passes) if total_tiles_all_passes else float("nan")

    notes_parts = []
    if cap_note:
        notes_parts.append(cap_note)
    if apply_sigmoid_threshold:
        notes_parts.append("sigmoid+0.5-threshold applied inside timed region")
    if actual_passes < requested_passes and not cap_note:
        notes_parts.append(f"only {actual_passes}/{requested_passes} passes completed")
    notes_parts.append(
        f"pass elapsed (sec): {['%.3f' % p for p in pass_elapsed_sec]} "
        f"(first pass may include DataLoader worker startup overhead when num_workers>0)"
    )

    return {
        "num_tiles_processed": num_tiles_processed,
        "passes": actual_passes,
        "requested_passes": requested_passes,
        "total_elapsed_sec": total_elapsed_sec,
        "mean_pass_sec": mean_pass_sec,
        "tiles_per_sec": tiles_per_sec,
        "ms_per_tile": ms_per_tile,
        "notes": "; ".join(notes_parts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "full_unet_end_to_end_timing.csv"
    md_path = output_dir / "full_unet_end_to_end_timing_summary.md"

    print("=" * 78)
    print("Environment")
    print("=" * 78)
    print(f"torch version    : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available   : {cuda_available}")
    device = torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if (device.type == "cuda" and cuda_available) else "N/A"
    print(f"Device requested : {device}")
    print(f"GPU name         : {gpu_name}")
    if device.type == "cuda" and not cuda_available:
        raise SystemExit("ERROR: --device cuda requested but CUDA is not available.")

    model = load_model(args.checkpoint, device)
    print(f"    Model moved to device: {device}\n")

    print(f"[2] Building usable tile row list from split CSV (preflight existence "
          f"check only -- no file reads yet) ...")
    print(f"    Split CSV   : {args.split_csv}")
    print(f"    Data root   : {args.data_root}")
    print(f"    Max tiles   : {'all' if args.max_tiles <= 0 else args.max_tiles}")
    rows = build_usable_rows(args.split_csv, args.data_root, args.max_tiles)
    num_tiles_requested = len(rows)
    print(f"    Usable tiles found on disk: {len(rows)}")
    if not rows:
        raise SystemExit("ERROR: no usable tiles found -- cannot benchmark.")

    print("\n" + "=" * 78)
    print(f"Benchmarking END-TO-END dataset pass (disk read + normalize + batch + "
          f"host-to-device + forward), passes={args.passes}")
    print("=" * 78)

    result_rows: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        for num_workers in args.num_workers_list:
            print(f"\n  Config: device={device} batch_size={batch_size} num_workers={num_workers}")
            cfg_result = run_one_config(
                model, rows, args.data_root, device, batch_size, num_workers,
                args.passes, args.cpu_time_budget_sec, args.apply_sigmoid_threshold,
            )
            row = {
                "device": str(device),
                "gpu_name": gpu_name,
                "checkpoint": str(args.checkpoint.relative_to(REPO_ROOT)),
                "split_csv": str(args.split_csv.relative_to(REPO_ROOT)),
                "data_root": str(args.data_root.relative_to(REPO_ROOT)),
                "num_tiles_requested": num_tiles_requested,
                "num_tiles_processed": cfg_result["num_tiles_processed"],
                "batch_size": batch_size,
                "num_workers": num_workers,
                "passes": cfg_result["passes"],
                "total_elapsed_sec": cfg_result["total_elapsed_sec"],
                "mean_pass_sec": cfg_result["mean_pass_sec"],
                "tiles_per_sec": cfg_result["tiles_per_sec"],
                "ms_per_tile": cfg_result["ms_per_tile"],
                "includes_data_loading": True,
                "includes_normalization": True,
                "includes_host_to_device_transfer": True,
                "includes_model_forward": True,
                "notes": cfg_result["notes"],
            }
            result_rows.append(row)
            print(f"    -> passes={cfg_result['passes']}/{cfg_result['requested_passes']}  "
                  f"total={cfg_result['total_elapsed_sec']:.3f}s  "
                  f"mean_pass={cfg_result['mean_pass_sec']:.3f}s  "
                  f"tiles/s={cfg_result['tiles_per_sec']:.3f}  "
                  f"ms/tile={cfg_result['ms_per_tile']:.2f}")

    # -- Merge with any prior run's rows (e.g. a previous --device cuda run
    # followed by a --device cpu run, or a different batch/num_workers set)
    # so re-running this script never silently clobbers other configs'
    # results -- same pattern as benchmark_full_unet_inference.py. --
    fieldnames = [
        "device", "gpu_name", "checkpoint", "split_csv", "data_root",
        "num_tiles_requested", "num_tiles_processed", "batch_size", "num_workers",
        "passes", "total_elapsed_sec", "mean_pass_sec", "tiles_per_sec", "ms_per_tile",
        "includes_data_loading", "includes_normalization",
        "includes_host_to_device_transfer", "includes_model_forward", "notes",
    ]
    merged_by_key: dict[tuple[str, int, int], dict[str, object]] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for prior_row in csv.DictReader(handle):
                key = (prior_row["device"], int(prior_row["batch_size"]), int(prior_row["num_workers"]))
                for int_field in ("num_tiles_requested", "num_tiles_processed", "batch_size",
                                  "num_workers", "passes"):
                    prior_row[int_field] = int(prior_row[int_field])
                for float_field in ("total_elapsed_sec", "mean_pass_sec", "tiles_per_sec", "ms_per_tile"):
                    prior_row[float_field] = float(prior_row[float_field])
                for bool_field in ("includes_data_loading", "includes_normalization",
                                   "includes_host_to_device_transfer", "includes_model_forward"):
                    prior_row[bool_field] = prior_row[bool_field] in ("True", "true", "1")
                merged_by_key[key] = prior_row
    for new_row in result_rows:
        key = (new_row["device"], new_row["batch_size"], new_row["num_workers"])
        merged_by_key[key] = new_row
    merged_rows = list(merged_by_key.values())

    print(f"\n[3] Writing CSV: {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"[4] Writing markdown summary: {md_path}")
    write_markdown_summary(md_path, merged_rows, cuda_available, gpu_name, device, args)
    print(f"    Written: {md_path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 78)
    print("Done. Outputs:")
    print(f"  {csv_path.relative_to(REPO_ROOT)}")
    print(f"  {md_path.relative_to(REPO_ROOT)}")
    print("=" * 78)


def write_markdown_summary(
    md_path: pathlib.Path,
    rows: list[dict[str, object]],
    cuda_available: bool,
    gpu_name: str,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    table_rows = "\n".join(
        f"| {r['device']} | {r['batch_size']} | {r['num_workers']} | {r['passes']} | "
        f"{r['num_tiles_processed']} | {r['total_elapsed_sec']:.3f} | "
        f"{r['mean_pass_sec']:.3f} | {r['tiles_per_sec']:.3f} | {r['ms_per_tile']:.2f} |"
        for r in sorted(rows, key=lambda r: (str(r["device"]), int(r["batch_size"]), int(r["num_workers"])))
    )
    notes_rows = "\n".join(
        f"- **device={r['device']} batch={r['batch_size']} workers={r['num_workers']}**: {r['notes']}"
        for r in rows if r["notes"]
    )
    if not notes_rows:
        notes_rows = "(no capping or extra notes triggered for this run)"

    distinct_devices = sorted({str(r["device"]) for r in rows})
    multi_device_note = (
        f"\nThis file accumulates results across separate invocations of this "
        f"script; rows are present for device(s): {', '.join(distinct_devices)}. "
        f"The Environment table below reflects only the MOST RECENT invocation "
        f"({device}) -- each row's own `device` column is authoritative for "
        f"that row's measurement.\n"
        if len(distinct_devices) > 1 else ""
    )

    md = f"""# Full U-Net End-to-End Inference Throughput Benchmark

Generated by `scripts/benchmark_full_unet_end_to_end.py`.

## Claim boundary (read this first)

- **This is END-TO-END dataset-pass timing, NOT model-forward-pass-only
  timing.** It is a deliberate companion to
  `scripts/benchmark_full_unet_inference.py`, which explicitly EXCLUDES
  disk I/O, normalization, and host-to-device transfer (those tensors are
  prepared before its timed loop starts). This script measures the
  opposite: a realistic full pass over real UAVSAR tiles, with every step
  a real pipeline pays for included in the timed region.
- **Timing INCLUDES**: rasterio disk reads of each SAR tile file,
  selecting the first 3 bands, per-tile normalization (percentile-clip +
  zero-mean/unit-variance, identical to
  `FloodTileDataset._normalize_per_tile` in
  `scripts/train_unet_baseline_tuned.py`), DataLoader batching, host-to-
  device transfer, and the full U-Net forward pass under `torch.no_grad()`.
  {"A cheap sigmoid + 0.5 threshold is ALSO included (see notes column for which rows)." if args.apply_sigmoid_threshold else "No sigmoid/threshold step was applied in this run (--apply-sigmoid-threshold was not set)."}
- **This does NOT compute Dice/IoU/recall or any other accuracy metric**,
  and it does NOT estimate or claim any FPGA timing.
- **Results can vary run to run** depending on OS filesystem cache state
  (a tile read twice may be served from page cache the second time),
  disk state, DataLoader worker scheduling, and other CPU load on the
  machine at the time of the run -- this is expected for any real
  end-to-end I/O-inclusive benchmark and is not a bug.

## Environment
{multi_device_note}
| Field | Value |
|---|---|
| torch version | {torch.__version__} |
| CUDA available | {cuda_available} |
| GPU name | {gpu_name} |
| Device used (this invocation) | {device} |
| Checkpoint | `{args.checkpoint.relative_to(REPO_ROOT) if args.checkpoint.is_relative_to(REPO_ROOT) else args.checkpoint}` |
| Split CSV | `{args.split_csv.relative_to(REPO_ROOT) if args.split_csv.is_relative_to(REPO_ROOT) else args.split_csv}` |
| Data root | `{args.data_root.relative_to(REPO_ROOT) if args.data_root.is_relative_to(REPO_ROOT) else args.data_root}` |
| Requested max tiles | {"all" if args.max_tiles <= 0 else args.max_tiles} |
| Requested passes (this invocation) | {args.passes} |
| Requested batch sizes (this invocation) | {args.batch_sizes} |
| Requested num_workers values (this invocation) | {args.num_workers_list} |

## Timing table (measured, end-to-end: disk read + normalize + batch + host-to-device + forward)

| Device | Batch | Workers | Passes | Tiles/pass | Total (sec) | Mean pass (sec) | Tiles/sec | ms/tile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

`Tiles/sec` and `ms/tile` are computed across ALL passes combined
(`total tiles processed = tiles/pass * passes`, divided into
`total_elapsed_sec`), not just the last pass.

## Notes / caps triggered

{notes_rows}

## Comparison to the forward-pass-only benchmark

`scripts/benchmark_full_unet_inference.py` measures MODEL FORWARD PASS
ONLY (tensors already on-device before timing starts) -- its per-image
timings are expected to be substantially FASTER than this script's
end-to-end `ms/tile` figures, because this script additionally pays for
disk I/O, per-tile normalization, DataLoader batching overhead, and
host-to-device transfer on every single tile, every pass. The gap between
the two scripts' numbers is itself informative: it approximates the
"non-model" overhead of a real inference pipeline reading from disk,
which the forward-pass-only benchmark deliberately excludes. See
`outputs/hardware_benchmarks/full_unet_inference_timing/full_unet_timing_summary.md`
for the forward-pass-only figures to compare against.

## Limitations

- **First pass may be slower than later passes**, especially with
  `num_workers > 0` (DataLoader worker process startup) or a cold OS file
  cache (first read from disk vs. later cached reads). All passes are
  included in `total_elapsed_sec`/`mean_pass_sec` -- this script does NOT
  discard a "warmup" pass, unlike the forward-pass-only benchmark's
  explicit warmup iterations. Per-pass elapsed times are listed in the
  `notes` column for inspection.
- **`num_tiles_processed` may be less than `num_tiles_requested`** if some
  tile files referenced by the split CSV are missing on disk; both figures
  are reported explicitly.
- **Same fixed tile subset every pass** (no shuffling, `shuffle=False`),
  for reproducibility -- this is not a claim about performance on a
  randomly-ordered or larger sample of the split.
- **CPU timing may be capped.** If `--device cpu` and the first pass's
  elapsed time projects (times the requested `--passes`) to longer than
  `--cpu-time-budget-sec` (default 120s), remaining passes for that
  config are skipped and this is reported in the `notes` column.
- **No accuracy metric.** Dice/IoU/recall are not computed here.
- **No FPGA numbers.** This script does not estimate or claim any FPGA
  timing.
- **Single machine, single GPU (if CUDA).** No multi-GPU, no mixed
  precision, no `torch.compile`, no TensorRT/ONNX export.
- **This does not retrain, fine-tune, or otherwise modify the
  checkpoint**; it only loads `model_state_dict` for inference-mode
  benchmarking.
- **Results depend on machine state** (filesystem cache, disk load, other
  processes) at the time of the run and are not guaranteed to reproduce
  exactly on a re-run, unlike the more controlled forward-pass-only
  benchmark.
"""
    md_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
