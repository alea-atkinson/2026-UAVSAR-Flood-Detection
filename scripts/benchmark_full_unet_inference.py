#!/usr/bin/env python3
"""
benchmark_full_unet_inference.py

Full-model PyTorch inference timing baseline for the trained UAVSAR
flood-segmentation U-Net (NOT just the first Conv-BN-ReLU stage -- the
complete encoder/bottleneck/decoder forward pass). This is the
full-software reference point the hardware feasibility discussion has
been missing: prior benchmarks
(scripts/benchmark_first_layer_gpu_vs_fpga_estimate.py) only measured the
first Conv-BN-ReLU stage against FPGA cycle estimates for that one stage.

---- Scope -------------------------------------------------------------
This benchmarks the COMPLETE U-Net forward pass (all encoder stages,
bottleneck, all decoder stages, final 1x1 conv), in inference mode
(`model.eval()`, `torch.no_grad()`). It does NOT implement or estimate any
FPGA timing, does NOT modify the checkpoint, and does NOT evaluate
segmentation accuracy (Dice/IoU/recall) -- see the tuned training script's
own evaluation loop for that.

---- Sources inspected before writing this script -----------------------
- scripts/train_unet_baseline_tuned.py: the `DoubleConv`/`UNet` class
  definitions (copied verbatim below, unchanged) and the checkpoint
  loading convention (`torch.load(..., map_location=device,
  weights_only=False)` then `model.load_state_dict(checkpoint["model_state_dict"])`),
  and `FloodTileDataset._normalize_per_tile` (reused verbatim for the
  real-tile benchmark mode).
- models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt:
  confirmed to store `base_channels=32` under its saved `args` dict,
  matching the `UNet(in_channels=3, out_channels=1, base_channels=32)`
  default.
- scripts/benchmark_first_layer_gpu_vs_fpga_estimate.py: timing convention
  (torch.cuda.Event pairs recorded without inter-iteration sync, then one
  `torch.cuda.synchronize()` before reading elapsed times; CPU wall-clock
  fallback) and output-file structure, reused here for consistency.

Note: the UNet/DoubleConv classes are copied verbatim into this file
rather than imported from train_unet_baseline_tuned.py, because that
script has a module-level `if "-h"/"--help" in sys.argv: print_basic_help();
raise SystemExit(0)` check that would fire on THIS script's own
--help/--device/etc. argv and exit early if imported directly.

---- Timing scope (IMPORTANT) --------------------------------------------
All reported timings are MODEL FORWARD PASS ONLY. For synthetic inputs,
random tensors are constructed once, on-device, before the timed loop.
For real-tile inputs, tiles are loaded from disk, normalized, converted to
tensors, and moved to the target device BEFORE the timed loop begins --
none of that data loading/preprocessing time is included in the reported
mean/median/min/max/std timings. This is stated explicitly in every output
row's `timing_scope` field and in the markdown summary.

---- Usage ----------------------------------------------------------------
    python scripts/benchmark_full_unet_inference.py --device cuda --warmup 20 --iters 100
    python scripts/benchmark_full_unet_inference.py --device cpu --warmup 5 --iters 20

---- Outputs --------------------------------------------------------------
outputs/hardware_benchmarks/full_unet_inference_timing/full_unet_timing.csv
outputs/hardware_benchmarks/full_unet_inference_timing/full_unet_timing_summary.md
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import time

import numpy as np
import rasterio
import torch
import torch.nn as nn

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
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "hardware_benchmarks" / "full_unet_inference_timing"

TIMING_SCOPE_NOTE = "model_forward_pass_only (data loading/preprocessing excluded from timed loop)"


# ---------------------------------------------------------------------------
# Model -- DoubleConv/UNet copied VERBATIM from scripts/train_unet_baseline_tuned.py
# (see module docstring for why this is a copy, not an import).
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
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full U-Net PyTorch inference timing benchmark "
        "(synthetic inputs + real UAVSAR tiles)."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-csv", type=pathlib.Path, default=DEFAULT_SPLIT_CSV,
                         help="Split CSV to draw real tiles from (default: fp2 test split).")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (default: 20).")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations (default: 100).")
    parser.add_argument("--cpu-time-budget-sec", type=float, default=60.0,
                         help="On CPU only: if the requested --iters would project to "
                         "longer than this many seconds (estimated from one warmup "
                         "iteration), --iters is automatically reduced and the cap is "
                         "reported in the output notes (default: 60.0).")
    parser.add_argument("--include-128", action="store_true", default=True,
                         help="Include the optional synthetic 128x128 batch-1 "
                         "configuration (default: on).")
    parser.add_argument("--no-include-128", dest="include_128", action="store_false")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading -- same convention as train_unet_baseline_tuned.py:
# torch.load(..., map_location=device, weights_only=False), then
# model.load_state_dict(checkpoint["model_state_dict"]).
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
# Real-tile loading (data loading/preprocessing -- NOT part of timed loop)
# ---------------------------------------------------------------------------
def load_real_tiles(split_csv: pathlib.Path, data_root: pathlib.Path, max_tiles: int) -> list[np.ndarray]:
    """Returns a list of (3, H, W) float32 normalized SAR arrays, loaded and
    preprocessed OUTSIDE of any timed region."""
    if not split_csv.exists():
        print(f"    WARNING: split CSV not found at {split_csv}; no real tiles loaded.")
        return []
    if not data_root.exists():
        print(f"    WARNING: data root not found at {data_root}; no real tiles loaded.")
        return []

    with open(split_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    tiles = []
    for row in rows:
        if len(tiles) >= max_tiles:
            break
        tile_path = data_root / row["uavsar_path"]
        if not tile_path.exists():
            continue
        with rasterio.open(tile_path) as src:
            sar = src.read(out_dtype="float32")
        if sar.shape[0] < 3:
            continue
        sar = sar[:3]
        tiles.append(normalize_per_tile(sar))
    return tiles


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_forward_pass(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
    cpu_time_budget_sec: float,
) -> tuple[list[float], int, str]:
    """Returns (times_ms, actual_iters_used, cap_note). `x` must already be
    on `device` -- no data movement happens inside this function's timed
    region."""
    cap_note = ""
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                starts[i].record()
                _ = model(x)
                ends[i].record()
            torch.cuda.synchronize()
            times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            return times_ms, iters, cap_note

        # CPU: estimate one iteration's cost first, then cap --iters if the
        # full requested count would blow past the time budget.
        t0 = time.perf_counter()
        _ = model(x)
        t1 = time.perf_counter()
        per_iter_est_sec = t1 - t0

        actual_iters = iters
        projected_sec = per_iter_est_sec * iters
        if projected_sec > cpu_time_budget_sec:
            actual_iters = max(3, int(cpu_time_budget_sec / per_iter_est_sec))
            cap_note = (
                f"CPU timing capped: requested --iters={iters} projected to "
                f"~{projected_sec:.1f}s (> {cpu_time_budget_sec:.0f}s budget); "
                f"reduced to {actual_iters} iters."
            )
            print(f"    NOTE: {cap_note}")

        times_ms = [per_iter_est_sec * 1000.0]  # the estimation iteration counts as iter 1
        for _ in range(actual_iters - 1):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
        return times_ms, actual_iters, cap_note


def summarize_times(times_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "full_unet_timing.csv"
    md_path = output_dir / "full_unet_timing_summary.md"

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

    # -- Build synthetic configs --
    synthetic_configs = [
        {"label": "synthetic_256x256_batch1", "H": 256, "W": 256, "batch": 1},
        {"label": "synthetic_256x256_batch8", "H": 256, "W": 256, "batch": 8},
    ]
    if args.include_128:
        synthetic_configs.append({"label": "synthetic_128x128_batch1", "H": 128, "W": 128, "batch": 1})

    # -- Load real tiles (data loading -- NOT timed) --
    print("[2] Loading real UAVSAR tiles for real-tile timing mode "
          "(data loading/preprocessing here is NOT part of any timed region) ...")
    print(f"    Split CSV : {args.split_csv}")
    real_tiles = load_real_tiles(args.split_csv, args.data_root, max_tiles=8)
    print(f"    Loaded {len(real_tiles)} usable real tile(s).")

    real_configs = []
    if len(real_tiles) >= 1:
        real_configs.append({"label": "real_tile_batch1", "batch": 1, "tiles": real_tiles[:1]})
    if len(real_tiles) >= 8:
        real_configs.append({"label": "real_tile_batch8", "batch": 8, "tiles": real_tiles[:8]})
    elif len(real_tiles) > 1:
        print(f"    NOTE: only {len(real_tiles)} real tile(s) available; "
              f"skipping real_tile_batch8 (needs 8).")

    print("\n" + "=" * 78)
    print(f"Benchmarking full U-Net forward pass (warmup={args.warmup}, iters={args.iters})")
    print("Timing scope: MODEL FORWARD PASS ONLY (see docstring / summary notes)")
    print("=" * 78)

    rows: list[dict[str, object]] = []

    # -- Synthetic --
    for cfg in synthetic_configs:
        label, H, W, batch = cfg["label"], cfg["H"], cfg["W"], cfg["batch"]
        x = torch.randn(batch, 3, H, W, dtype=torch.float32, device=device)
        times_ms, actual_iters, cap_note = time_forward_pass(
            model, x, device, args.warmup, args.iters, args.cpu_time_budget_sec
        )
        stats = summarize_times(times_ms)
        images_per_sec = batch * 1000.0 / stats["mean_ms"]
        row = {
            "input_mode": "synthetic",
            "label": label,
            "H": H, "W": W, "batch": batch,
            "device": str(device), "gpu_name": gpu_name,
            "checkpoint": str(args.checkpoint.relative_to(REPO_ROOT)),
            "warmup_iters": args.warmup,
            "timed_iters": actual_iters,
            "requested_iters": args.iters,
            "mean_ms": stats["mean_ms"], "median_ms": stats["median_ms"],
            "min_ms": stats["min_ms"], "max_ms": stats["max_ms"], "std_ms": stats["std_ms"],
            "images_per_sec": images_per_sec,
            "timing_scope": TIMING_SCOPE_NOTE,
            "notes": cap_note,
        }
        rows.append(row)
        print(f"  {label:28s} batch={batch:2d}  mean={stats['mean_ms']:9.4f} ms  "
              f"median={stats['median_ms']:9.4f} ms  img/s={images_per_sec:9.2f}"
              + (f"  [{cap_note}]" if cap_note else ""))

    # -- Real tile --
    for cfg in real_configs:
        label, batch, tiles = cfg["label"], cfg["batch"], cfg["tiles"]
        x_np = np.stack(tiles, axis=0)  # (batch, 3, H, W)
        x = torch.from_numpy(x_np).float().to(device)  # data movement done BEFORE timing
        H, W = x.shape[2], x.shape[3]
        times_ms, actual_iters, cap_note = time_forward_pass(
            model, x, device, args.warmup, args.iters, args.cpu_time_budget_sec
        )
        stats = summarize_times(times_ms)
        images_per_sec = batch * 1000.0 / stats["mean_ms"]
        row = {
            "input_mode": "real_tile",
            "label": label,
            "H": H, "W": W, "batch": batch,
            "device": str(device), "gpu_name": gpu_name,
            "checkpoint": str(args.checkpoint.relative_to(REPO_ROOT)),
            "warmup_iters": args.warmup,
            "timed_iters": actual_iters,
            "requested_iters": args.iters,
            "mean_ms": stats["mean_ms"], "median_ms": stats["median_ms"],
            "min_ms": stats["min_ms"], "max_ms": stats["max_ms"], "std_ms": stats["std_ms"],
            "images_per_sec": images_per_sec,
            "timing_scope": TIMING_SCOPE_NOTE,
            "notes": (cap_note + "; " if cap_note else "")
            + f"real fp2-test tiles from {args.split_csv.name}, same tile(s) reused every "
              f"iteration (no per-iteration data loading)",
        }
        rows.append(row)
        print(f"  {label:28s} batch={batch:2d}  mean={stats['mean_ms']:9.4f} ms  "
              f"median={stats['median_ms']:9.4f} ms  img/s={images_per_sec:9.2f}"
              + (f"  [{cap_note}]" if cap_note else ""))

    # -- Merge with any prior run's rows (e.g. a previous --device cuda run
    # followed by a --device cpu run) so re-running this script for a
    # different device does not clobber the other device's results. Rows are
    # keyed by (input_mode, label, device); a rerun of the SAME key replaces
    # the prior row (latest measurement wins), other keys are preserved. --
    fieldnames = [
        "input_mode", "label", "H", "W", "batch", "device", "gpu_name", "checkpoint",
        "warmup_iters", "timed_iters", "requested_iters",
        "mean_ms", "median_ms", "min_ms", "max_ms", "std_ms", "images_per_sec",
        "timing_scope", "notes",
    ]
    merged_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for prior_row in csv.DictReader(handle):
                key = (prior_row["input_mode"], prior_row["label"], prior_row["device"])
                for numeric_field in ("H", "W", "batch", "warmup_iters", "timed_iters", "requested_iters"):
                    prior_row[numeric_field] = int(prior_row[numeric_field])
                for float_field in ("mean_ms", "median_ms", "min_ms", "max_ms", "std_ms", "images_per_sec"):
                    prior_row[float_field] = float(prior_row[float_field])
                merged_by_key[key] = prior_row
    for new_row in rows:
        key = (new_row["input_mode"], new_row["label"], new_row["device"])
        merged_by_key[key] = new_row
    merged_rows = list(merged_by_key.values())

    # -- Write CSV (merged across any prior run for a different device) --
    print(f"\n[3] Writing CSV: {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    # -- Write markdown summary (also from the merged row set) --
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
        f"| {r['input_mode']} | {r['label']} | {r['device']} | {r['H']}x{r['W']} | {r['batch']} | "
        f"{r['warmup_iters']} | {r['timed_iters']} | {r['mean_ms']:.4f} | "
        f"{r['median_ms']:.4f} | {r['min_ms']:.4f} | {r['max_ms']:.4f} | "
        f"{r['std_ms']:.4f} | {r['images_per_sec']:.2f} |"
        for r in sorted(rows, key=lambda r: (r["device"], r["input_mode"], r["label"]))
    )
    notes_rows = "\n".join(
        f"- **{r['label']}** ({r['device']}): {r['notes']}" for r in rows if r["notes"]
    )
    if not notes_rows:
        notes_rows = "(no capping or extra notes triggered for this run)"

    distinct_devices = sorted({str(r["device"]) for r in rows})
    multi_device_note = (
        f"\nThis file accumulates results across separate invocations of this "
        f"script; rows are present for device(s): {', '.join(distinct_devices)}. "
        f"The Environment table below reflects only the MOST RECENT invocation "
        f"({device}) -- each row's own `Device` column is authoritative for "
        f"that row's measurement.\n"
        if len(distinct_devices) > 1 else ""
    )

    md = f"""# Full U-Net PyTorch Inference Timing Benchmark

Generated by `scripts/benchmark_full_unet_inference.py`.

## Claim boundary (read this first)

- **All timings in this report are MEASURED** on this machine using
  `torch.cuda.Event(enable_timing=True)` (CUDA) or `time.perf_counter()`
  wall-clock (CPU), with warmup iterations and (for CUDA)
  `torch.cuda.synchronize()` before reading elapsed times.
- **Timing scope is MODEL FORWARD PASS ONLY.** Input tensors (synthetic or
  real-tile) are constructed/loaded and moved to the target device
  BEFORE the timed loop begins; data loading, file I/O, normalization,
  and host-to-device transfer are explicitly EXCLUDED from every
  mean/median/min/max/std figure in this report.
- This benchmarks the COMPLETE U-Net (all encoder/bottleneck/decoder
  stages), not just the first Conv-BN-ReLU layer. It does NOT measure
  segmentation accuracy (Dice/IoU/recall) and does NOT implement or
  estimate any FPGA timing -- it is a pure PyTorch/GPU (or CPU) software
  timing baseline, for comparison against separate FPGA feasibility
  analyses elsewhere in this repository.

## Environment
{multi_device_note}
| Field | Value |
|---|---|
| torch version | {torch.__version__} |
| CUDA available | {cuda_available} |
| GPU name | {gpu_name} |
| Device used | {device} |
| Checkpoint | `{args.checkpoint.relative_to(REPO_ROOT) if args.checkpoint.is_relative_to(REPO_ROOT) else args.checkpoint}` |
| Requested warmup iterations | {args.warmup} |
| Requested timed iterations | {args.iters} |
| Real-tile split CSV | `{args.split_csv.relative_to(REPO_ROOT) if args.split_csv.is_relative_to(REPO_ROOT) else args.split_csv}` |
| Input dtype | float32 |

## Timing table (measured, model forward pass only)

| Input mode | Label | Device | Input shape (HxW) | Batch | Warmup | Timed iters | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Std (ms) | Images/sec |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

`timed_iters` may differ from the requested iteration count only on CPU,
if the CPU time-budget cap (`--cpu-time-budget-sec`, default 60s) reduced
it -- see notes below.

## Notes / caps triggered

{notes_rows}

## Real-tile timing details

Real-tile rows use actual UAVSAR SAR tiles from the fp2 held-out test
split (`{args.split_csv.name}`), read with rasterio, first 3 bands only
(matching the model's 3-channel input), normalized with the SAME
per-tile percentile-clip + zero-mean/unit-variance normalization used by
`FloodTileDataset._normalize_per_tile` in
`scripts/train_unet_baseline_tuned.py` (reused verbatim, not reinvented).
The SAME loaded/normalized tile(s) are reused for every timed iteration
(no per-iteration disk I/O) -- this isolates model forward-pass cost from
data loading cost, per the task's instruction to separate the two where
practical.

## Safe interpretation

- These are software (PyTorch, GPU or CPU) inference timings for the
  COMPLETE U-Net, intended as the full-model reference point against
  which any FPGA first-layer feasibility discussion in this repository
  should be read -- the FPGA analyses elsewhere in this repo cover only
  the first Conv-BN-ReLU stage, not the complete model measured here.
- Batch-8 throughput (images/sec) is not simply 8x the batch-1 rate;
  larger batches typically amortize fixed kernel-launch/memory-bandwidth
  overhead better on GPU, so batch-8 images/sec is expected to exceed
  batch-1 images/sec on CUDA.
- Real-tile timings and synthetic-input timings at the same shape/batch
  are expected to be very close (the forward pass does the same amount of
  arithmetic regardless of input content), and any small difference is
  measurement noise, not a property of the data.

## Limitations

- **Forward pass only.** No accuracy metric (Dice/IoU/recall) is computed
  in this script.
- **No FPGA numbers.** This script does not estimate or claim any FPGA
  timing; see `scripts/benchmark_first_layer_gpu_vs_fpga_estimate.py` and
  `hardware/vhdl_conv3x3/` for the (first-layer-only) FPGA side of this
  comparison.
- **Real-tile mode reuses the same loaded tile(s) every iteration** (by
  design, to isolate forward-pass cost) -- this is NOT a streaming/
  dataset-iteration throughput benchmark with per-sample disk I/O.
- **CPU timing may be capped.** If `--device cpu` and the requested
  `--iters` would project (from a single measured iteration) to longer
  than `--cpu-time-budget-sec` (default 60s), the iteration count is
  automatically reduced and the cap is reported in the `notes` column and
  above.
- **Single machine, single GPU (if CUDA).** No multi-GPU, no mixed
  precision, no `torch.compile`, no TensorRT/ONNX export -- this is plain
  eager-mode PyTorch inference, matching how the tuned training script
  itself runs evaluation.
- **This does not retrain, fine-tune, or otherwise modify the checkpoint**;
  it only loads `model_state_dict` for inference-mode benchmarking.
"""
    md_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
