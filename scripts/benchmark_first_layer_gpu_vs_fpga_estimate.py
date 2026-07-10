#!/usr/bin/env python3
"""
benchmark_first_layer_gpu_vs_fpga_estimate.py

GPU-measured vs FPGA cycle-estimated benchmark for the FIRST Conv-BN-ReLU
stage of the trained U-Net (enc1.block.0 Conv2d + enc1.block.1 BatchNorm2d
+ ReLU), compared against the RTL-measured and synthesis-level cycle
estimates from the resource-shared FPGA prototypes in
hardware/vhdl_conv3x3/.

---- Scope -----------------------------------------------------------------
This benchmarks ONLY the first Conv-BN-ReLU stage (3 -> 32 channels,
3x3 kernel), NOT the full U-Net. It is NOT a full U-Net benchmark and NOT
a board-measured FPGA speedup claim.

---- Claim boundary (IMPORTANT) ---------------------------------------------
GPU numbers in this script are ACTUALLY MEASURED on this machine's GPU
using torch.cuda.Event timing. FPGA numbers are NOT board-measured: the
5x5 toy-input cycle counts are RTL-measured (GHDL simulation cycle
counts from the existing, already-synthesized VHDL prototypes), and the
128x128 / 256x256 cycle counts are FORMULA-BASED ESTIMATES extrapolated
from that RTL-measured toy-input behavior and the existing
resource-shared latency-scaling formula (see
hardware/vhdl_conv3x3/resource_shared_first_layer_conv_bn_relu_plan.md
Section 13 and
hardware/vhdl_conv3x3/multi_lane_resource_shared_conv_bn_relu_design_memo.md
Section 6). This is a GPU-measured vs FPGA cycle-estimated comparison,
not a measured FPGA speedup and not board power.

---- Sources inspected before writing this script --------------------------
- scripts/train_unet_baseline_tuned.py / scripts/train_unet_baseline.py
  (DoubleConv / UNet architecture; enc1.block.0/1 layout)
- hardware/vhdl_conv3x3/first_layer_32out_bn_relu_2lane_resource_shared_summary.md
  (2-lane measured: LUTs 2,531, registers 5,878, DSP48E1 4/740,
  WNS +0.229 ns, power 0.176 W, 5,085 cycles for the 5x5x3 toy input)
- hardware/vhdl_conv3x3/first_layer_32out_bn_relu_resource_shared_summary.md
  (1-lane measured: LUTs 1,837, registers 5,588, DSP48E1 2/740,
  WNS +0.662 ns, power 0.165 W, 10,125 cycles for the 5x5x3 toy input)
- hardware/vhdl_conv3x3/multi_lane_resource_shared_conv_bn_relu_design_memo.md
  (latency-scaling formula: total_cycles(S,N,L) ~= S^2 + (S-2)^2*35*N/L)
- hardware/vhdl_conv3x3/first_layer_32out_bn_relu_resource_shared_pkg.vhd
  (confirms NUM_KERNELS=32, NUM_WINDOWS=9 for the 5x5 toy input, i.e.
  S=5 in the formula above)
- models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt
  (checkpoint under benchmark; enc1.block.0.weight is [32, 3, 3, 3],
  enc1.block.1.* are the BatchNorm2d parameters, confirmed by inspection)

---- Usage ------------------------------------------------------------------
    python3 scripts/benchmark_first_layer_gpu_vs_fpga_estimate.py

---- Outputs ------------------------------------------------------------
outputs/hardware_benchmarks/first_layer_gpu_vs_fpga_estimate/first_layer_gpu_timing.csv
outputs/hardware_benchmarks/first_layer_gpu_vs_fpga_estimate/first_layer_gpu_vs_fpga_estimate_summary.md
"""

from __future__ import annotations

import csv
import pathlib
import statistics
import time

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
CONV_KEY = "enc1.block.0.weight"        # [32, 3, 3, 3], bias=False
BN_PREFIX = "enc1.block.1"              # BatchNorm2d(32), eval-mode running stats
BN_EPS = 1e-5                            # torch.nn.BatchNorm2d default

OUT_DIR = REPO_ROOT / "outputs" / "hardware_benchmarks" / "first_layer_gpu_vs_fpga_estimate"
CSV_PATH = OUT_DIR / "first_layer_gpu_timing.csv"
MD_PATH = OUT_DIR / "first_layer_gpu_vs_fpga_estimate_summary.md"

WARMUP_ITERS = 20
TIMED_ITERS = 100

# [batch, 3, H, W] input configurations. Batch-8 sizes are the optional
# additions requested; all others are required.
INPUT_CONFIGS = [
    {"label": "5x5_batch1", "H": 5, "W": 5, "batch": 1, "required": True},
    {"label": "128x128_batch1", "H": 128, "W": 128, "batch": 1, "required": True},
    {"label": "256x256_batch1", "H": 256, "W": 256, "batch": 1, "required": True},
    {"label": "128x128_batch8", "H": 128, "W": 128, "batch": 8, "required": False},
    {"label": "256x256_batch8", "H": 256, "W": 256, "batch": 8, "required": False},
]

# ---------------------------------------------------------------------------
# FPGA cycle-estimate constants (from the existing RTL-measured/synthesized
# resource-shared prototypes -- NOT re-derived here, only reused).
# ---------------------------------------------------------------------------
FPGA_CLOCK_HZ = 100e6   # 100 MHz, matching every Vivado synthesis in this repo
FPGA_NUM_KERNELS = 32   # complete first Conv2d layer, N in the scaling formula

# RTL-measured (GHDL) cycle counts for the canonical 5x5x3 toy input (S=5).
MEASURED_1LANE_TOY_CYCLES = 10_125
MEASURED_2LANE_TOY_CYCLES = 5_085

# Direct-parallel 32-output design: only a rough, already-documented toy
# reference exists in this repo (~4-5 cycles). No documented formula exists
# for direct-parallel latency at larger tile sizes, so this script does NOT
# invent one -- see the markdown summary's FPGA estimate table.
DIRECT_PARALLEL_TOY_CYCLES_APPROX = (4, 5)


def fpga_resource_shared_cycles(spatial_size: int, num_kernels: int, lanes: int) -> float:
    """Resource-shared latency-scaling formula (SYNTHESIS-LEVEL ESTIMATE for
    spatial_size != 5; the formula's S=5 case is cross-checked against the
    RTL-measured toy cycle counts above, not used to override them).

    total_cycles(S, N, L) ~= S^2 + (S - 2)^2 * 35 * N / L

    From hardware/vhdl_conv3x3/resource_shared_first_layer_conv_bn_relu_plan.md
    Section 13 and
    hardware/vhdl_conv3x3/multi_lane_resource_shared_conv_bn_relu_design_memo.md
    Section 6.
    """
    s = spatial_size
    return s**2 + (s - 2) ** 2 * 35 * num_kernels / lanes


def cycles_to_ms(cycles: float) -> float:
    return cycles / FPGA_CLOCK_HZ * 1000.0


# ---------------------------------------------------------------------------
# Model: ONLY the first Conv-BN-ReLU stage (3 -> 32 channels), not the full
# U-Net. Weights loaded directly from the trained checkpoint's enc1.block
# tensors.
# ---------------------------------------------------------------------------
class FirstConvBNReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def load_first_layer(device: torch.device) -> FirstConvBNReLU:
    print(f"[1] Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    conv_w = state_dict[CONV_KEY]
    print(f"    Conv tensor key   : {CONV_KEY}  shape={list(conv_w.shape)}")
    print(f"    BatchNorm prefix  : {BN_PREFIX}")

    model = FirstConvBNReLU()
    with torch.no_grad():
        model.conv.weight.copy_(conv_w)
        model.bn.weight.copy_(state_dict[f"{BN_PREFIX}.weight"])
        model.bn.bias.copy_(state_dict[f"{BN_PREFIX}.bias"])
        model.bn.running_mean.copy_(state_dict[f"{BN_PREFIX}.running_mean"])
        model.bn.running_var.copy_(state_dict[f"{BN_PREFIX}.running_var"])
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# GPU timing
# ---------------------------------------------------------------------------
def benchmark_one_config(
    model: FirstConvBNReLU,
    device: torch.device,
    H: int,
    W: int,
    batch: int,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
) -> list[float]:
    """Returns per-iteration timings in milliseconds."""
    x = torch.randn(batch, 3, H, W, dtype=torch.float32, device=device)

    with torch.no_grad():
        # ---- Warmup (not timed): lets CUDA kernels JIT/select, caches warm.
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        if device.type == "cuda":
            # Record all start/end event pairs first, without synchronizing
            # in between, so per-iteration host-side sync overhead does not
            # contaminate the measured GPU time.
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                starts[i].record()
                _ = model(x)
                ends[i].record()
            torch.cuda.synchronize()
            times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        else:
            # CPU fallback: wall-clock timing (CUDA events are not
            # applicable). Only used if CUDA is unavailable on this machine.
            times_ms = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    return times_ms


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
    print("=" * 78)
    print("GPU environment")
    print("=" * 78)
    print(f"torch version    : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available   : {cuda_available}")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A (no CUDA device)"
    print(f"GPU name         : {gpu_name}")
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Device used      : {device}")
    print()

    model = load_first_layer(device)
    print(f"    Model moved to device: {device}\n")

    print("=" * 78)
    print("Benchmarking first Conv-BN-ReLU stage (NOT the full U-Net)")
    print(f"warmup_iters={WARMUP_ITERS}  timed_iters={TIMED_ITERS}")
    print("=" * 78)

    rows: list[dict[str, object]] = []
    for cfg in INPUT_CONFIGS:
        label, H, W, batch = cfg["label"], cfg["H"], cfg["W"], cfg["batch"]
        times_ms = benchmark_one_config(model, device, H, W, batch)
        stats = summarize_times(times_ms)

        note = ""
        if label == "5x5_batch1":
            note = (
                "5x5 GPU timing is dominated by kernel-launch/overhead and is "
                "NOT necessarily meaningful as a GPU throughput benchmark; "
                "included only for direct comparison against the FPGA "
                "prototypes' own 5x5x3 toy input."
            )

        row = {
            "input_label": label,
            "H": H,
            "W": W,
            "batch": batch,
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
            "device": str(device),
            "mean_ms": stats["mean_ms"],
            "median_ms": stats["median_ms"],
            "min_ms": stats["min_ms"],
            "max_ms": stats["max_ms"],
            "std_ms": stats["std_ms"],
            "note": note,
        }
        rows.append(row)
        print(
            f"  {label:16s} batch={batch}  "
            f"mean={stats['mean_ms']:.4f} ms  median={stats['median_ms']:.4f} ms  "
            f"min={stats['min_ms']:.4f} ms  max={stats['max_ms']:.4f} ms  "
            f"std={stats['std_ms']:.4f} ms"
        )

    # ---- Write CSV ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[2] Writing GPU timing CSV: {CSV_PATH}")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "input_label", "H", "W", "batch", "warmup_iters", "timed_iters",
            "device", "mean_ms", "median_ms", "min_ms", "max_ms", "std_ms", "note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- FPGA cycle estimates ----------------------------------------------
    print("\n" + "=" * 78)
    print("FPGA latency: RTL-measured (toy) / synthesis-level cycle estimate (larger tiles)")
    print("=" * 78)

    fpga_rows: list[dict[str, object]] = []

    # S=5 toy input: use the RTL-MEASURED (GHDL) cycle counts directly, not
    # the formula (the formula is cross-checked against these, not a
    # replacement for them).
    fpga_rows.append({
        "spatial_size": 5, "design": "1-lane resource-shared (measured)",
        "cycles": MEASURED_1LANE_TOY_CYCLES, "source": "RTL-measured (GHDL)",
        "ms": cycles_to_ms(MEASURED_1LANE_TOY_CYCLES),
    })
    fpga_rows.append({
        "spatial_size": 5, "design": "2-lane resource-shared (measured)",
        "cycles": MEASURED_2LANE_TOY_CYCLES, "source": "RTL-measured (GHDL)",
        "ms": cycles_to_ms(MEASURED_2LANE_TOY_CYCLES),
    })
    lo, hi = DIRECT_PARALLEL_TOY_CYCLES_APPROX
    fpga_rows.append({
        "spatial_size": 5, "design": "direct-parallel (approximate, documented)",
        "cycles": f"{lo}-{hi}", "source": "documented approximate reference",
        "ms": f"{cycles_to_ms(lo):.5f}-{cycles_to_ms(hi):.5f}",
    })

    # S=128, S=256: formula-based SYNTHESIS-LEVEL ESTIMATES, 1-lane and 2-lane only.
    for spatial_size in (128, 256):
        for lanes, design_name in ((1, "1-lane resource-shared (estimate)"),
                                    (2, "2-lane resource-shared (estimate)")):
            cyc = fpga_resource_shared_cycles(spatial_size, FPGA_NUM_KERNELS, lanes)
            fpga_rows.append({
                "spatial_size": spatial_size, "design": design_name,
                "cycles": round(cyc), "source": "formula estimate (S^2+(S-2)^2*35*N/L)",
                "ms": cycles_to_ms(cyc),
            })
        fpga_rows.append({
            "spatial_size": spatial_size, "design": "direct-parallel",
            "cycles": "not estimated", "source": "no documented formula in this repo",
            "ms": "not estimated",
        })

    for r in fpga_rows:
        ms_display = r["ms"] if isinstance(r["ms"], str) else f"{r['ms']:.5f}"
        print(f"  S={r['spatial_size']:4d}  {r['design']:42s}  cycles={r['cycles']!s:12s}  ms={ms_display}")

    # ---- Write markdown summary ---------------------------------------------
    print(f"\n[3] Writing markdown summary: {MD_PATH}")
    write_markdown_summary(rows, fpga_rows, cuda_available, gpu_name, device)
    print(f"    Written: {MD_PATH.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 78)
    print("Done. Outputs:")
    print(f"  {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)


def write_markdown_summary(
    gpu_rows: list[dict[str, object]],
    fpga_rows: list[dict[str, object]],
    cuda_available: bool,
    gpu_name: str,
    device: torch.device,
) -> None:
    gpu_row_by_label = {r["input_label"]: r for r in gpu_rows}

    gpu_table_rows = "\n".join(
        f"| {r['input_label']} | {r['H']} | {r['W']} | {r['batch']} | "
        f"{r['mean_ms']:.4f} | {r['median_ms']:.4f} | {r['min_ms']:.4f} | "
        f"{r['max_ms']:.4f} | {r['std_ms']:.4f} |"
        for r in gpu_rows
    )

    def fmt_ms(v: object) -> str:
        return v if isinstance(v, str) else f"{v:.5f}"

    fpga_table_rows = "\n".join(
        f"| {r['spatial_size']} | {r['design']} | {r['cycles']} | {r['source']} | {fmt_ms(r['ms'])} |"
        for r in fpga_rows
    )

    def gpu_ms_for(label: str) -> str:
        r = gpu_row_by_label.get(label)
        return f"{r['mean_ms']:.4f}" if r else "N/A"

    def fpga_ms_for(spatial_size: int, design_substr: str) -> str:
        for r in fpga_rows:
            if r["spatial_size"] == spatial_size and design_substr in r["design"]:
                return fmt_ms(r["ms"])
        return "N/A"

    comparison_rows = []
    for spatial_size, label in ((5, "5x5_batch1"), (128, "128x128_batch1"), (256, "256x256_batch1")):
        gpu_ms = gpu_ms_for(label)
        fpga_1lane_ms = fpga_ms_for(spatial_size, "1-lane")
        fpga_2lane_ms = fpga_ms_for(spatial_size, "2-lane")
        comparison_rows.append(
            f"| {spatial_size}x{spatial_size} | {gpu_ms} | {fpga_1lane_ms} | {fpga_2lane_ms} |"
        )
    comparison_table = "\n".join(comparison_rows)

    md = f"""\
# First Conv-BN-ReLU Stage: GPU-Measured vs FPGA Cycle-Estimated Comparison

Generated by `scripts/benchmark_first_layer_gpu_vs_fpga_estimate.py`.

## Claim boundary (read this first)

- **GPU numbers in this report are MEASURED** on this machine's GPU using
  `torch.cuda.Event(enable_timing=True)`, with warmup iterations and
  `torch.cuda.synchronize()` before reading timings.
- **FPGA numbers in this report are NOT board-measured.** The 5x5 toy
  -input cycle counts are **RTL-measured** (GHDL simulation cycle counts
  from the existing, Vivado-synthesized resource-shared VHDL prototypes).
  The 128x128 and 256x256 cycle counts are **synthesis-level estimates**
  extrapolated from that RTL-measured behavior using the existing
  resource-shared latency-scaling formula -- they have NOT been
  implemented, simulated, or synthesized at those sizes.
- This is a **GPU-measured vs FPGA cycle-estimated** comparison, not a
  measured FPGA speedup, not real-time FPGA acceleration, not board power,
  and not a full U-Net acceleration claim.
- Scope: the FIRST Conv-BN-ReLU stage only (3 -> 32 channels, 3x3 kernel,
  `enc1.block.0` Conv2d + `enc1.block.1` BatchNorm2d + ReLU). This is
  **not a full U-Net benchmark**.

## GPU environment

| Field | Value |
|---|---|
| torch version | {torch.__version__} |
| CUDA available | {cuda_available} |
| GPU name | {gpu_name} |
| Device used | {device} |
| Warmup iterations | {WARMUP_ITERS} |
| Timed iterations | {TIMED_ITERS} |
| Input dtype | float32 |
| Input shape | [batch, 3, H, W] |

## GPU timing table (measured)

| Input | H | W | Batch | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Std (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{gpu_table_rows}

**Note:** the 5x5 GPU timing is dominated by kernel-launch/dispatch
overhead and is **not necessarily meaningful as a GPU throughput
benchmark**. It is included only so it can be placed side-by-side with
the FPGA prototypes' own 5x5x3 toy input, which is the only input size
those prototypes have actually been simulated and synthesized against.

## FPGA latency table (RTL-measured toy / synthesis-level estimate for larger tiles)

FPGA clock: 100 MHz (matches every Vivado synthesis in this repo).
Formula for spatial sizes other than 5: `total_cycles(S, N, L) ~= S^2 + (S-2)^2 * 35 * N / L`,
N = 32 (complete first Conv2d layer), L = lane count (1 or 2).

| Spatial size (S) | Design | Cycles | Source | Time (ms) @ 100 MHz |
|---:|---|---|---|---:|
{fpga_table_rows}

Direct-parallel FPGA latency at 128x128 and 256x256 was **not estimated**
in this script: no documented latency-scaling formula for the
direct-parallel (non-resource-shared) 32-output design exists in this
repository. Only the toy-input approximate reference (~4-5 cycles,
already documented) is reported.

## Comparison table (GPU-measured vs FPGA cycle-estimated, batch=1)

| Spatial size | GPU mean (ms) | FPGA 1-lane (ms) | FPGA 2-lane (ms) |
|---:|---:|---:|---:|
{comparison_table}

GPU column is measured; FPGA columns are RTL-measured (S=5) or
synthesis-level estimates (S=128, S=256), NOT board-measured, as stated
in the claim boundary above.

## Safe interpretation

- The GPU executes the first Conv-BN-ReLU stage on 128x128 and 256x256
  inputs in a fraction of a millisecond (measured), because a modern GPU
  runs this stage as a small number of highly parallel, highly optimized
  CUDA kernels.
- The FPGA resource-shared prototypes, as actually built and measured
  today, process only a 5x5x3 toy input, one time-multiplexed window at
  a time, and their cycle counts scale to larger tiles only as an
  **estimate** via the existing scaling formula -- not as a measurement.
- At the toy-input scale, the FPGA resource-shared designs take
  RTL-measured cycle counts (10,125 for 1-lane, 5,085 for 2-lane) that
  correspond to roughly 0.1 ms and 0.05 ms respectively at 100 MHz --
  orders of magnitude slower in raw cycle count than the GPU's warmup
  -dominated 5x5 timing, but the two are not directly comparable in a
  meaningful throughput sense at this trivial input size (see the 5x5
  note above).
- At 128x128 and 256x256, the FPGA figures are **estimates only**: they
  have not been built, simulated, or synthesized at those sizes. Any gap
  between the GPU-measured and FPGA-estimated numbers at these sizes
  reflects the current resource-shared architecture's estimated scaling
  behavior, not a measured outcome, and should not be read as a proven
  speedup or slowdown in either direction.
- The direct-parallel FPGA design's toy-input cycle count (~4-5 cycles)
  is far lower than either resource-shared design's, consistent with it
  using dedicated per-kernel hardware (740 DSPs) instead of time
  -multiplexing -- but this repo has no documented way to extrapolate
  that design's latency to larger tiles, so no larger-tile comparison is
  offered for it here.

## Limitations

- This is the first Conv-BN-ReLU stage ONLY, not the full U-Net (which
  has an encoder, bottleneck, and decoder with several more stages).
- This is NOT a full U-Net benchmark.
- FPGA numbers beyond the 5x5x3 toy input are **synthesis-level
  estimates**, not measurements -- the resource-shared design has only
  been built, GHDL-simulated, and Vivado-synthesized for the 5x5x3 toy
  input to date.
- FPGA numbers are **not board-measured** at any input size: all FPGA
  figures in this report come from GHDL RTL simulation cycle counts and
  Vivado out-of-context synthesis reports, not from running on physical
  FPGA hardware.
- No FPGA board power figures are reported here; the resource-shared
  designs' *synthesis-estimated* on-chip power (not board power) is
  documented separately in
  `hardware/vhdl_conv3x3/first_layer_32out_bn_relu_resource_shared_summary.md`
  and
  `hardware/vhdl_conv3x3/first_layer_32out_bn_relu_2lane_resource_shared_summary.md`.
- GPU timings use `torch.cuda.Event` wall-clock kernel timing on a single
  GPU with random (not real UAVSAR) input tensors; they do not include
  data loading, preprocessing, or any full-model inference overhead.
- The 5x5 GPU timing is overhead-dominated and not a meaningful GPU
  throughput measurement (see note above); it is included only for
  side-by-side reference against the FPGA prototypes' own toy input size.
- Direct-parallel FPGA latency at 128x128/256x256 is not estimated in
  this report; no documented formula for it exists in this repository.
- This script does not retrain, fine-tune, or otherwise modify the
  checkpoint; it only loads `enc1.block.0` / `enc1.block.1` weights for
  inference-mode benchmarking.
"""
    MD_PATH.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
