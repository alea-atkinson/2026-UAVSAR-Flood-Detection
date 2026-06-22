#!/usr/bin/env python3
"""
Benchmark U-Net inference under different pipeline configurations.

Separates data-loading bottlenecks from model-compute bottlenecks.
No retraining. No modification of split files. Analysis only.

Usage (run from project root):
    python scripts/benchmark_unet_pipeline_variants_for_hardware.py

Outputs:
    outputs/hardware_profile/unet_pipeline_variant_benchmark.csv
    outputs/hardware_profile/unet_pipeline_variant_benchmark.md
"""
from __future__ import annotations

import csv
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_scripts_dir = PROJECT_ROOT / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import train_unet_experiment as _te  # noqa: E402

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
TEST_CSV = (
    PROJECT_ROOT
    / "csv_splits"
    / "flood_splits_ieee_png_filtered_standard_strict_train_val"
    / "strict_no_overlap"
    / "heldout_fp2_test.csv"
)
DATA_ROOT = PROJECT_ROOT / "2025_Tile_Data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hardware_profile"

THRESHOLD = 0.5
WARMUP_BATCHES = 3
CPU_SUBSET_N = 20
BATCH_SIZES = [1, 2, 4, 8, 16]
WORKER_COUNTS = [0, 2, 4]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now() -> float:
    cuda_sync()
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchRow:
    variant: str
    device: str
    batch_size: int
    num_workers: int
    n_tiles: int
    data_load_ms: float   # mean ms/tile; -1.0 = N/A
    forward_ms: float     # mean ms/tile; -1.0 = failed
    postproc_ms: float    # mean ms/tile; -1.0 = N/A
    total_ms: float       # mean ms/tile; -1.0 = failed
    tiles_per_sec: float
    notes: str


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_to(device: torch.device) -> tuple[nn.Module, int, int]:
    """Return (model, param_count, base_channels)."""
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    bc = int(ckpt.get("args", {}).get("base_channels", 32))
    model = _te.UNet(in_channels=3, out_channels=1, base_channels=bc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, sum(p.numel() for p in model.parameters()), bc


# ---------------------------------------------------------------------------
# Benchmark: DataLoader pipeline (I/O + preprocess + forward + postproc)
# ---------------------------------------------------------------------------

def bench_dataloader_variant(
    variant: str,
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    n_limit: int | None = None,
) -> BenchRow:
    ds = Subset(dataset, list(range(min(n_limit, len(dataset))))) if n_limit else dataset

    try:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )
    except Exception as exc:
        return BenchRow(variant, str(device), batch_size, num_workers, 0,
                        -1, -1, -1, -1, 0.0, f"loader_init_error: {exc}")

    # Warmup with actual data
    try:
        warmup_iter = iter(loader)
        with torch.no_grad():
            for _ in range(WARMUP_BATCHES):
                try:
                    imgs, _ = next(warmup_iter)
                except StopIteration:
                    break
                model(imgs.to(device))
        cuda_sync()
        del warmup_iter
        gc.collect()
    except Exception:
        pass

    load_ms: list[float] = []
    fwd_ms: list[float] = []
    post_ms: list[float] = []
    n_done = 0

    try:
        t0 = now()
        for imgs, _ in loader:
            t1 = now()
            bs_actual = imgs.shape[0]
            load_ms.append((t1 - t0) * 1000.0 / bs_actual)

            imgs = imgs.to(device)

            t2 = now()
            with torch.no_grad():
                logits = model(imgs)
            t3 = now()
            fwd_ms.append((t3 - t2) * 1000.0 / bs_actual)

            t4 = now()
            _ = (torch.sigmoid(logits) > THRESHOLD).float().cpu()
            t5 = now()
            post_ms.append((t5 - t4) * 1000.0 / bs_actual)

            n_done += bs_actual
            t0 = now()

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return BenchRow(variant, str(device), batch_size, num_workers, n_done,
                            -1, -1, -1, -1, 0.0, "OOM")
        return BenchRow(variant, str(device), batch_size, num_workers, n_done,
                        -1, -1, -1, -1, 0.0, f"runtime_error: {exc}")
    except Exception as exc:
        return BenchRow(variant, str(device), batch_size, num_workers, n_done,
                        -1, -1, -1, -1, 0.0, f"error: {exc}")

    if not fwd_ms:
        return BenchRow(variant, str(device), batch_size, num_workers, 0,
                        -1, -1, -1, -1, 0.0, "no_data")

    skip = 1 if len(fwd_ms) > 2 else 0
    ml = float(np.mean(load_ms[skip:]))
    mf = float(np.mean(fwd_ms[skip:]))
    mp = float(np.mean(post_ms[skip:]))
    total = ml + mf + mp
    tps = 1000.0 / total if total > 0 else 0.0
    note = f"nw{num_workers}_prefetch" if num_workers > 0 else "sync_io"

    return BenchRow(variant, str(device), batch_size, num_workers, n_done,
                    ml, mf, mp, total, tps, note)


# ---------------------------------------------------------------------------
# Benchmark: preloaded tensors (model-only, no I/O)
# ---------------------------------------------------------------------------

def bench_preloaded_variant(
    model: nn.Module,
    tensors: torch.Tensor,  # (N, C, H, W) CPU float32
    device: torch.device,
    batch_size: int,
) -> BenchRow:
    n = tensors.shape[0]
    shape = (min(batch_size, n), *tensors.shape[1:])

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_BATCHES):
            model(torch.zeros(shape, device=device))
    cuda_sync()

    fwd_ms: list[float] = []
    post_ms: list[float] = []
    n_done = 0

    try:
        for start in range(0, n, batch_size):
            batch = tensors[start:start + batch_size].to(device)
            bs_actual = batch.shape[0]

            t0 = now()
            with torch.no_grad():
                logits = model(batch)
            t1 = now()
            fwd_ms.append((t1 - t0) * 1000.0 / bs_actual)

            t2 = now()
            _ = (torch.sigmoid(logits) > THRESHOLD).float().cpu()
            t3 = now()
            post_ms.append((t3 - t2) * 1000.0 / bs_actual)

            n_done += bs_actual

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return BenchRow("preloaded", str(device), batch_size, 0, n_done,
                            0.0, -1, -1, -1, 0.0, "OOM")
        raise

    skip = 1 if len(fwd_ms) > 2 else 0
    mf = float(np.mean(fwd_ms[skip:]))
    mp = float(np.mean(post_ms[skip:]))
    total = mf + mp
    tps = 1000.0 / total if total > 0 else 0.0

    return BenchRow("preloaded", str(device), batch_size, 0, n_done,
                    0.0, mf, mp, total, tps, f"model_only_bs{batch_size}")


# ---------------------------------------------------------------------------
# Benchmark: CPU forward only (small subset)
# ---------------------------------------------------------------------------

def bench_cpu_forward(model_cpu: nn.Module, dataset, n_tiles: int) -> BenchRow:
    fwd_ms: list[float] = []
    n_done = 0

    for i in range(min(n_tiles, len(dataset))):
        img, _ = dataset[i]
        img = img.unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            model_cpu(img)
        t1 = time.perf_counter()
        fwd_ms.append((t1 - t0) * 1000.0)
        n_done += 1

    skip = 1 if len(fwd_ms) > 2 else 0
    mf = float(np.mean(fwd_ms[skip:]))
    tps = 1000.0 / mf if mf > 0 else 0.0

    return BenchRow("cpu_subset", "cpu", 1, 0, n_done,
                    -1, mf, -1, mf, tps, f"cpu_fwd_only_first{n_tiles}tiles")


# ---------------------------------------------------------------------------
# Preload all tiles into CPU RAM
# ---------------------------------------------------------------------------

def preload_tensors(dataset) -> torch.Tensor:
    n = len(dataset)
    print(f"  Preloading {n} tiles into CPU RAM ...")
    tiles = []
    for i in range(n):
        img, _ = dataset[i]
        tiles.append(img)
        if (i + 1) % 30 == 0 or (i + 1) == n:
            print(f"    {i + 1}/{n}")
    result = torch.stack(tiles)
    mem_mb = result.numel() * 4 / 1024 / 1024
    print(f"  Preloaded: {tuple(result.shape)}  ({mem_mb:.1f} MB)")
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fv(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}" if v >= 0 else "N/A"


def _ftps(v: float) -> str:
    return f"{v:.1f}" if v > 0 else "N/A"


def _table_row(r: BenchRow, cols: list[str]) -> str:
    vals: dict = {
        "variant": r.variant, "device": r.device,
        "batch_size": r.batch_size, "num_workers": r.num_workers,
        "n_tiles": r.n_tiles,
        "data_load_ms": _fv(r.data_load_ms, 3),
        "forward_ms": _fv(r.forward_ms, 3),
        "postproc_ms": _fv(r.postproc_ms, 3),
        "total_ms": _fv(r.total_ms, 3),
        "tiles_per_sec": _ftps(r.tiles_per_sec),
        "notes": r.notes,
    }
    return "| " + " | ".join(str(vals[c]) for c in cols) + " |"


def _make_table(rows: list[BenchRow], cols: list[str]) -> str:
    hdr = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = "\n".join(_table_row(r, cols) for r in rows)
    return f"{hdr}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"GPU RAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    print("Loading model and dataset ...")
    model, param_count, base_channels = load_model_to(device)
    dataset = _te.FloodTileDataset(TEST_CSV, DATA_ROOT, augment=False)
    n_tiles = len(dataset)
    sample_img, _ = dataset[0]
    in_ch, H, W = sample_img.shape
    print(f"  Params: {param_count:,}  |  tiles: {n_tiles}  |  shape: (1,{in_ch},{H},{W})")
    print()

    rows: list[BenchRow] = []

    # ------------------------------------------------------------------
    # 1. Raw pipeline baseline (bs=1, nw=0)
    # ------------------------------------------------------------------
    print("[1/5] Raw pipeline baseline (bs=1, nw=0) ...")
    r_base = bench_dataloader_variant("raw_pipeline", model, dataset, device, 1, 0)
    rows.append(r_base)
    print(f"  {_fv(r_base.total_ms)} ms/tile  {_ftps(r_base.tiles_per_sec)} tps  [{r_base.notes}]")
    print()

    # ------------------------------------------------------------------
    # 2. Batch size scaling (nw=0)
    # ------------------------------------------------------------------
    print("[2/5] Batch size scaling (nw=0) ...")
    for bs in BATCH_SIZES:
        r = bench_dataloader_variant("batch_scale", model, dataset, device, bs, 0)
        rows.append(r)
        if r.notes == "OOM":
            print(f"  bs={bs:2d}: OOM")
        else:
            print(f"  bs={bs:2d}: {_fv(r.total_ms)} ms/tile  {_ftps(r.tiles_per_sec)} tps")
    print()

    # ------------------------------------------------------------------
    # 3. DataLoader worker scaling (bs=1)
    # ------------------------------------------------------------------
    print("[3/5] DataLoader worker scaling (bs=1) ...")
    for nw in WORKER_COUNTS:
        r = bench_dataloader_variant("worker_scale", model, dataset, device, 1, nw)
        rows.append(r)
        if r.tiles_per_sec > 0:
            print(f"  nw={nw}: {_fv(r.total_ms)} ms/tile  {_ftps(r.tiles_per_sec)} tps")
        else:
            print(f"  nw={nw}: FAILED — {r.notes}")
    print()

    # ------------------------------------------------------------------
    # 4. Preloaded tensor benchmark
    # ------------------------------------------------------------------
    print("[4/5] Preloaded tensor benchmark ...")
    preloaded = preload_tensors(dataset)
    for bs in BATCH_SIZES:
        r = bench_preloaded_variant(model, preloaded, device, bs)
        rows.append(r)
        if r.notes == "OOM":
            print(f"  bs={bs:2d}: OOM")
        else:
            print(f"  bs={bs:2d}: {_fv(r.forward_ms)} ms/tile fwd  {_ftps(r.tiles_per_sec)} tps")
    del preloaded
    gc.collect()
    print()

    # ------------------------------------------------------------------
    # 5. CPU subset
    # ------------------------------------------------------------------
    print(f"[5/5] CPU benchmark (first {CPU_SUBSET_N} tiles) ...")
    model_cpu, _, _ = load_model_to(torch.device("cpu"))
    r_cpu = bench_cpu_forward(model_cpu, dataset, CPU_SUBSET_N)
    rows.append(r_cpu)
    print(f"  {_fv(r_cpu.forward_ms)} ms/tile fwd  {_ftps(r_cpu.tiles_per_sec)} tps")
    del model_cpu
    print()

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    csv_path = OUTPUT_DIR / "unet_pipeline_variant_benchmark.csv"
    fieldnames = ["variant", "device", "batch_size", "num_workers", "n_tiles",
                  "data_load_ms", "forward_ms", "postproc_ms", "total_ms",
                  "tiles_per_sec", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "variant": r.variant, "device": r.device,
                "batch_size": r.batch_size, "num_workers": r.num_workers,
                "n_tiles": r.n_tiles,
                "data_load_ms": _fv(r.data_load_ms, 3),
                "forward_ms": _fv(r.forward_ms, 3),
                "postproc_ms": _fv(r.postproc_ms, 3),
                "total_ms": _fv(r.total_ms, 3),
                "tiles_per_sec": _ftps(r.tiles_per_sec),
                "notes": r.notes,
            })
    print(f"Wrote: {csv_path.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    batch_rows = sorted([r for r in rows if r.variant == "batch_scale"], key=lambda x: x.batch_size)
    worker_rows = sorted([r for r in rows if r.variant == "worker_scale"], key=lambda x: x.num_workers)
    preload_rows = sorted([r for r in rows if r.variant == "preloaded"], key=lambda x: x.batch_size)

    valid_batch = [r for r in batch_rows if r.tiles_per_sec > 0]
    valid_workers = [r for r in worker_rows if r.tiles_per_sec > 0]
    valid_preload = [r for r in preload_rows if r.forward_ms > 0]

    bs1_batch = next((r for r in valid_batch if r.batch_size == 1), None)
    pre_bs1 = next((r for r in valid_preload if r.batch_size == 1), None)
    nw0_worker = next((r for r in valid_workers if r.num_workers == 0), None)
    best_batch = max(valid_batch, key=lambda x: x.tiles_per_sec) if valid_batch else None
    best_pre = min(valid_preload, key=lambda x: x.forward_ms) if valid_preload else None

    # Q1: batching
    if best_batch and bs1_batch and best_batch.batch_size != 1:
        speedup = best_batch.tiles_per_sec / bs1_batch.tiles_per_sec
        if speedup > 1.1:
            q1 = (f"**Yes.** bs={best_batch.batch_size} achieves {best_batch.tiles_per_sec:.1f} tiles/sec "
                  f"vs {bs1_batch.tiles_per_sec:.1f} at bs=1 ({speedup:.1f}× speedup). "
                  f"GPU parallelism is underutilized at single-tile inference.")
        else:
            q1 = (f"**Marginal ({speedup:.1f}×).** Best at bs={best_batch.batch_size} "
                  f"({best_batch.tiles_per_sec:.1f} tps) vs bs=1 ({bs1_batch.tiles_per_sec:.1f} tps). "
                  f"The pipeline is likely I/O-bound — batching the model does not help if "
                  f"data loading is the bottleneck.")
    elif best_batch and bs1_batch:
        q1 = (f"No significant gain. Best config is bs={best_batch.batch_size} "
              f"({best_batch.tiles_per_sec:.1f} tps) ≈ bs=1 ({bs1_batch.tiles_per_sec:.1f} tps).")
    else:
        q1 = "Insufficient data (OOM on all batch sizes > 1)."

    # Q2: workers
    if len(valid_workers) > 1 and nw0_worker:
        best_nw = min(valid_workers, key=lambda x: x.total_ms)
        if best_nw.num_workers > 0 and nw0_worker.total_ms > 0:
            gain_pct = (nw0_worker.total_ms - best_nw.total_ms) / nw0_worker.total_ms * 100
            if gain_pct > 5:
                q2 = (f"**Yes.** nw={best_nw.num_workers} reduces pipeline time by {gain_pct:.0f}% "
                      f"({nw0_worker.total_ms:.2f} → {best_nw.total_ms:.2f} ms/tile). "
                      f"Background workers overlap I/O with the prior batch's GPU compute.")
            else:
                q2 = (f"**No significant improvement** (nw={best_nw.num_workers}: {best_nw.total_ms:.2f} ms "
                      f"vs nw=0: {nw0_worker.total_ms:.2f} ms, {gain_pct:.1f}% difference). "
                      f"Data loading is fast enough that worker spawn overhead offsets prefetch gains.")
        else:
            q2 = f"nw=0 was best ({nw0_worker.total_ms:.2f} ms/tile). Workers did not help or failed."
    else:
        q2 = "Insufficient data — only nw=0 succeeded."

    # Q3: model-only
    if pre_bs1:
        if best_pre and best_pre.batch_size != 1:
            q3 = (f"**{pre_bs1.forward_ms:.2f} ms/tile** at bs=1 ({pre_bs1.tiles_per_sec:.1f} tps). "
                  f"Best at bs={best_pre.batch_size}: {best_pre.forward_ms:.2f} ms/tile "
                  f"({best_pre.tiles_per_sec:.1f} tps). "
                  f"This is the theoretical minimum — no rasterio, no normalization, "
                  f"tensors already in CPU RAM.")
        else:
            q3 = (f"**{pre_bs1.forward_ms:.2f} ms/tile** at bs=1 ({pre_bs1.tiles_per_sec:.1f} tps). "
                  f"No I/O, no per-tile normalization — pure model forward.")
    else:
        q3 = "Preloaded benchmark failed or produced no valid results."

    # Q4: I/O bottleneck
    if r_base.data_load_ms >= 0 and pre_bs1 and pre_bs1.forward_ms > 0:
        io_ms = r_base.data_load_ms
        fwd_ms_v = pre_bs1.forward_ms
        io_pct = 100.0 * io_ms / (io_ms + fwd_ms_v)
        if io_pct > 50:
            q4 = (f"**Yes — raster I/O dominates.** Loading + normalize = {io_ms:.2f} ms/tile "
                  f"({io_pct:.0f}% of I/O+forward). Model forward = {fwd_ms_v:.2f} ms/tile. "
                  f"Pre-normalizing tiles offline would nearly double pipeline throughput.")
        elif io_pct > 25:
            q4 = (f"**Partial bottleneck.** I/O = {io_ms:.2f} ms/tile ({io_pct:.0f}% of I/O+forward). "
                  f"Model forward = {fwd_ms_v:.2f} ms/tile. Both I/O and compute are significant.")
        else:
            q4 = (f"**No — model compute dominates.** "
                  f"I/O = {io_ms:.2f} ms/tile ({io_pct:.0f}% of I/O+forward). "
                  f"Model forward = {fwd_ms_v:.2f} ms/tile. "
                  f"Accelerating Conv2d layers is the priority.")
    else:
        q4 = "Could not determine — missing baseline or preloaded results."

    # Q5: FPGA implications
    if pre_bs1 and pre_bs1.forward_ms > 0 and r_base.data_load_ms >= 0:
        if r_base.data_load_ms > pre_bs1.forward_ms * 0.5:
            q5 = ("The host CPU spends more time on rasterio/normalization than the model spends "
                  "on compute. An FPGA accelerator gains the most by **eliminating host-CPU I/O** — "
                  "either pre-process tiles to normalized float32 tensors offline, or implement "
                  "lightweight normalization on the FPGA front-end. The Conv2d pipeline itself is "
                  "fast; the bottleneck to remove first is the disk→CPU→normalize path.")
        else:
            q5 = ("Model compute is the bottleneck. An FPGA targeting the **DoubleConv (3×3 Conv2d)** "
                  "blocks would give the largest end-to-end speedup. BatchNorm folding removes "
                  "separate BN hardware. INT8 quantization halves DSP/memory utilization. "
                  "Fixed 256×256 input and known channel widths (32→512) enable fully static "
                  "resource allocation — no dynamic shape handling required. "
                  "Tile streaming with ping-pong BRAM buffers handles the bottleneck layer's "
                  "memory requirements.")
    else:
        q5 = ("Both I/O and compute contribute. An FPGA solution should co-design the "
              "pre-processing pipeline with the Conv2d accelerator, processing tiles as a "
              "stream from input raster to prediction mask without intermediate CPU involvement.")

    # Q6: next hardware step
    if device.type == "cuda" and pre_bs1 and r_base.data_load_ms >= 0:
        io_frac = r_base.data_load_ms / (r_base.data_load_ms + pre_bs1.forward_ms)
        best_bs_str = f"bs={best_batch.batch_size}" if best_batch else "bs=4"
        if io_frac > 0.4:
            q6 = (f"1. **Pre-normalize tiles offline** — save as `.npy` or `.pt` tensors, "
                  f"eliminating rasterio + percentile normalization at inference.\n"
                  f"2. **Increase batch size** to {best_bs_str} for best GPU throughput.\n"
                  f"3. **Export to ONNX** → benchmark with TensorRT for production GPU deployment.\n"
                  f"4. **FPGA path**: ONNX → INT8 quantization → Vitis-AI synthesis (Zynq UltraScale+).")
        else:
            q6 = ("1. **BatchNorm folding** — absorb BN γ/β/μ/σ into Conv2d weights (zero inference cost).\n"
                  "2. **INT8 quantization** — `torch.quantization` or ONNX quantization to halve memory bandwidth.\n"
                  "3. **ONNX export** → TensorRT (GPU) or Vitis-AI / hls4ml (FPGA).\n"
                  "4. **Profile on target FPGA** (Zynq UltraScale+ / Alveo U50) with hls4ml.\n"
                  "5. **Tile streaming** — design BRAM-aware pipeline for 256×256 × 512ch bottleneck.")
    elif device.type == "cuda":
        q6 = ("1. Export to ONNX and benchmark with TensorRT.\n"
              "2. Apply BatchNorm folding and INT8 quantization.\n"
              "3. Evaluate hls4ml or Vitis-AI for FPGA synthesis.")
    else:
        q6 = ("No GPU found — CPU baseline established.\n"
              "1. Test on a CUDA GPU for realistic throughput numbers.\n"
              "2. Export to ONNX for cross-platform evaluation.\n"
              "3. Explore OpenCL or HLS-based FPGA path.")

    # ------------------------------------------------------------------
    # Write markdown
    # ------------------------------------------------------------------
    ALL_COLS = ["variant", "device", "batch_size", "num_workers", "n_tiles",
                "data_load_ms", "forward_ms", "postproc_ms", "total_ms",
                "tiles_per_sec", "notes"]
    BATCH_COLS = ["batch_size", "n_tiles", "data_load_ms", "forward_ms",
                  "postproc_ms", "total_ms", "tiles_per_sec", "notes"]
    WORKER_COLS = ["num_workers", "n_tiles", "data_load_ms", "forward_ms",
                   "postproc_ms", "total_ms", "tiles_per_sec", "notes"]
    PRELOAD_COLS = ["batch_size", "n_tiles", "forward_ms", "postproc_ms",
                    "total_ms", "tiles_per_sec", "notes"]
    CPU_COLS = ["device", "batch_size", "n_tiles", "forward_ms", "total_ms",
                "tiles_per_sec", "notes"]

    gpu_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"

    md = f"""# U-Net Pipeline Variant Benchmark

**Checkpoint**: `{CHECKPOINT.name}`
**Test CSV**: `{TEST_CSV.name}` ({n_tiles} tiles)
**Device**: `{device}` — {gpu_str}
**Parameters**: {param_count:,}
**Input shape**: (1, {in_ch}, {H}, {W})

---

## 1. Raw Pipeline Baseline (bs=1, nw=0)

Full GeoTIFF load + per-tile normalize + GPU forward + sigmoid + threshold.

{_make_table([r_base], ALL_COLS)}

---

## 2. Batch Size Scaling (num_workers=0)

{_make_table(batch_rows, BATCH_COLS)}

---

## 3. DataLoader Worker Scaling (batch_size=1)

`data_load_ms` with nw>0 reflects main-thread wait time — background workers
overlap disk I/O with the previous batch's GPU compute, so the reported value
underestimates actual per-tile I/O cost but shows real end-to-end pipeline benefit.

{_make_table(worker_rows, WORKER_COLS)}

---

## 4. Preloaded Tensor Benchmark (model-only, no I/O)

All {n_tiles} tiles pre-loaded into CPU RAM as normalized float32 tensors.
Only forward pass + sigmoid + threshold measured. No rasterio, no percentile normalize.
Compare `forward_ms` here vs `forward_ms` in section 2 to confirm model timing consistency.

{_make_table(preload_rows, PRELOAD_COLS)}

---

## 5. CPU Subset (first {CPU_SUBSET_N} tiles, batch_size=1)

Forward pass only. Data loading (rasterio + normalize) included in wall time but not separately timed.

{_make_table([r_cpu], CPU_COLS)}

---

## Analysis

### 1. Does batching improve GPU throughput?

{q1}

### 2. Does num_workers reduce data loading bottleneck?

{q2}

### 3. How fast is model-only inference when tensors are already loaded?

{q3}

### 4. Is raster loading/preprocessing the real bottleneck?

{q4}

### 5. What does this imply for FPGA acceleration?

{q5}

### 6. What should the next hardware step be?

{q6}

---

*Generated by `scripts/benchmark_unet_pipeline_variants_for_hardware.py`*
"""

    md_path = OUTPUT_DIR / "unet_pipeline_variant_benchmark.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote: {md_path.relative_to(PROJECT_ROOT)}")

    # Console summary
    print()
    print("=" * 62)
    print("BENCHMARK SUMMARY")
    print("=" * 62)
    print(f"  Raw pipeline (bs=1) : {_fv(r_base.total_ms)} ms/tile  {_ftps(r_base.tiles_per_sec)} tps")
    if best_batch:
        print(f"  Best batch config   : bs={best_batch.batch_size}  {_fv(best_batch.total_ms)} ms/tile  {_ftps(best_batch.tiles_per_sec)} tps")
    if pre_bs1 and pre_bs1.forward_ms > 0:
        print(f"  Model-only (bs=1)   : {_fv(pre_bs1.forward_ms)} ms/tile fwd  {_ftps(pre_bs1.tiles_per_sec)} tps")
    if best_pre and best_pre != pre_bs1:
        print(f"  Model-only best     : bs={best_pre.batch_size}  {_fv(best_pre.forward_ms)} ms/tile fwd  {_ftps(best_pre.tiles_per_sec)} tps")
    print(f"  CPU fwd (first {CPU_SUBSET_N})  : {_fv(r_cpu.forward_ms)} ms/tile  {_ftps(r_cpu.tiles_per_sec)} tps")
    print()
    print(f"  Outputs: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
