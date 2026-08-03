#!/usr/bin/env python3
"""
benchmark_first_layer_16out_gpu_vs_fpga.py

GPU-vs-FPGA comparison for the 16-OUTPUT (of 32) folded Conv-BN-ReLU
first-layer subproblem: the SAME arithmetic the NEW, comfortably-fitting
FPGA artifact (commit `fe79e096`,
`hardware/vhdl_conv3x3/first_layer_16out_folded_arithmetic_core_streaming_front_end.vhd`)
was built, GHDL-verified, and Vivado-synthesized to compute -- measured
against a real GPU, at patch/tile sizes spanning tiny-launch-overhead
through large-workload-throughput regimes.

---- Why this script exists, and how it differs from the existing
     32-output GPU-vs-FPGA benchmark ---------------------------------------
`scripts/benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py` already
compares GPU vs. FPGA for the FULL 32-output folded Conv-BN-ReLU
arithmetic core -- the design that saturates the Artix-7 200T at 740/740
(100%) DSPs, with zero headroom. This script does NOT modify or replace
that comparison. It performs the SAME kind of comparison for the NEW
16-output subproblem (channels 0-15 of `enc1.block.0`), which commit
`fe79e096` already showed uses only 405/740 (54.73%) DSPs -- 335 DSPs
(45.27%) of real headroom -- while still meeting 100 MHz timing. This
script asks: for that SAME smaller, comfortably-fitting real-model
subproblem, is the FPGA side meaningfully throughput-competitive with a
real GPU, or is any FPGA advantage limited to small-workload launch
overhead?

---- What this benchmark measures ------------------------------------------
For a workload of N pre-extracted, already-in-memory 3x3x3 windows
(flattened to shape [N, 27]):
    output[N, 16] = ReLU( windows[N, 27] @ w_folded[27, 16] + b_folded[16] )
where `w_folded`/`b_folded` come from folding the REAL trained
`enc1.block.0`/`enc1.block.1` weights (BatchNorm folded into the Conv2d
weights and bias, identical formula used throughout this repo), keeping
ONLY output channels 0-15 of the 32 total. GPU timing is MEASURED
(`torch.cuda.Event`, normal dispatch + CUDA Graph replay). FPGA timing is
a cycle-count figure derived from the NEW 16-output streaming front end +
arithmetic core's GHDL-confirmed/Vivado-synthesized behavior (commit
`fe79e096`) -- NOT board-measured.

---- What "patch," "window," and "tile" mean in this script -----------------
- **patch_size**: the side length of a square H x W input image/tile in
  pixels (e.g. patch_size=128 means a 128x128 pixel input).
- **window**: one 3x3x3 (9 pixels x 3 input channels) VALID (no padding)
  convolution input, extracted at one output spatial position. A
  `patch_size x patch_size` image yields `(patch_size - 2) ** 2` valid
  windows.
- **tile**: used interchangeably with "patch" in this script's prose --
  the same square H x W input image. The realistic tile sizes used
  elsewhere in this repo's real-tile verification work are 128 and 256
  (see `hardware/vhdl_conv3x3/real_tile_stimulus_pkg.vhd`'s underlying
  256x256 `tile_16_42.tif` source tile).

---- FPGA timing model: which cycle formula, and why ------------------------
This script uses the STREAMING FRONT END's cycle formula, NOT the
arithmetic-core-only "windows + 6" formula the existing 32-output
benchmark script uses:
    total_cycles = (patch_size * patch_size) + 6
This is the GHDL-CONFIRMED formula for a full `H x W` image streamed
pixel-by-pixel through `window3x3_stream_3chan_flattened` into the folded
arithmetic core (see
`hardware/vhdl_conv3x3/reports/direct_parallel_line_buffer_feasibility_summary.md`,
Section 5: "Confirmed formula: total_cycles = (H * W) + 6", GHDL-measured
as 42 cycles for a real 6x6 image: 36 pixels + 6-cycle arithmetic-core
drain). It is used here (rather than "windows + 6") because this script
compares against the INTEGRATED streaming design (front end + core), which
must pay the cost of actually streaming every pixel in, not just the
valid windows -- a fairer, more realistic FPGA-side cost than assuming
windows arrive pre-extracted from nowhere.

- **100 MHz**: SYNTHESIS-SUPPORTED. This is the exact clock target the
  16-output integrated streaming design (commit `fe79e096`) was
  Vivado-synthesized against and closed timing on, at IMG_WIDTH=128 AND
  IMG_WIDTH=256 (WNS = +2.218 ns at both widths -- see
  `hardware/vhdl_conv3x3/reports/first_layer_16out_folded_arithmetic_core_streaming_front_end_summary.md`).
  This is still a CYCLE-COUNT estimate (`cycles / clock_hz`), not a
  board-measured latency -- but the clock rate and cycle formula are
  synthesis-confirmed, not assumed.
- **300 MHz**: a CLEARLY HYPOTHETICAL, speed-oriented target. NO
  synthesis run in this repo -- for the 16-output design or any other --
  has confirmed 300 MHz timing closure. Reported ONLY as an illustrative
  "what if" upper bound; the conclusion of this report is NOT centered on
  it.

---- What this benchmark EXCLUDES ------------------------------------------
- Disk I/O, DataLoader startup, checkpoint loading time (loaded once,
  before any timed region, purely to obtain the real trained weights).
- The rest of the U-Net (encoder stages 2-4, bottleneck, decoder) and
  output channels 16-31 of this same first layer.
- Any board-measured FPGA timing or power -- FPGA numbers here are
  cycle-count estimates derived from a GHDL-verified, Vivado-synthesized
  design, never a board measurement.
- Full FPGA U-Net acceleration of any kind.
- Any claim that FPGA is faster than GPU in general -- see Section 11 of
  the generated markdown summary for explicit claim boundaries.

---- BatchNorm folding (identical formula used throughout this repo) -------
    scale_bn[oc] = gamma[oc] / sqrt(running_var[oc] + eps)
    w_folded[oc] = w[oc] * scale_bn[oc]
    b_folded[oc] = beta[oc] - running_mean[oc] * scale_bn[oc]

---- Workload sweep -----------------------------------------------------
patch_size in {6, 16, 32, 64, 128, 256, 512, 1024} -- spanning tiny
launch-overhead-dominated sizes, the realistic 128/256 real-tile scale
already used throughout this repo's real-tile verification work, and
large-workload throughput-dominated sizes.

---- Usage ------------------------------------------------------------------
    conda activate flood-unet-gpu
    python scripts/benchmark_first_layer_16out_gpu_vs_fpga.py

---- Outputs -----------------------------------------------------------
outputs/hardware_benchmarks/first_layer_16out_gpu_vs_fpga/
    first_layer_16out_gpu_vs_fpga_timing.csv
    first_layer_16out_gpu_vs_fpga_summary.md
"""

from __future__ import annotations

import csv
import pathlib
import statistics
import time

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
CONV_KEY = "enc1.block.0.weight"        # [32, 3, 3, 3], bias=False
BN_PREFIX = "enc1.block.1"              # BatchNorm2d(32), eval-mode running stats
BN_EPS = 1e-5                            # torch.nn.BatchNorm2d default
NUM_OUT_CHANNELS = 16                    # ONLY channels 0-15 of 32 -- the comfortable subproblem

OUT_DIR = REPO_ROOT / "outputs" / "hardware_benchmarks" / "first_layer_16out_gpu_vs_fpga"
CSV_PATH = OUT_DIR / "first_layer_16out_gpu_vs_fpga_timing.csv"
MD_PATH = OUT_DIR / "first_layer_16out_gpu_vs_fpga_summary.md"

WARMUP_ITERS = 30
TIMED_ITERS = 300
CUDAGRAPH_WARMUP_ITERS = 10
CUDAGRAPH_REPLAY_ITERS = 300

# patch_size (square H x W image side length, pixels) -> swept sizes.
# Spans tiny launch-overhead-dominated sizes (6-32), the realistic
# real-tile scale used elsewhere in this repo (128, 256), and
# large-workload throughput-dominated sizes (512, 1024).
WORKLOAD_PATCH_SIZES = [6, 16, 32, 64, 128, 256, 512, 1024]

FPGA_CLOCKS_HZ = {"100mhz": 100e6, "300mhz": 300e6}
# GHDL-CONFIRMED drain/fill constant for the integrated streaming front
# end + folded arithmetic core (see module docstring): total_cycles =
# (H * W) + FPGA_STREAM_DRAIN_CYCLES. Measured directly by GHDL for a
# real 6x6 image (36 pixels + 6 = 42 cycles) in
# hardware/vhdl_conv3x3/reports/direct_parallel_line_buffer_feasibility_summary.md.
FPGA_STREAM_DRAIN_CYCLES = 6

# The NEW, comfortably-fitting 16-output folded Conv-BN-ReLU streaming
# first-layer core (commit `fe79e096`) -- REUSED here (not re-derived).
# Source: hardware/vhdl_conv3x3/reports/first_layer_16out_folded_arithmetic_core_streaming_front_end_summary.md
SYNTHESIZED_16OUT_CONTEXT = {
    "part": "xc7a200tsbg484-1",
    "clock_mhz": 100,
    "ghdl_pass": 256,
    "ghdl_total": 256,
    "core_only": {
        "luts": 1796, "registers": 1238, "dsps_used": 405, "dsps_available": 740,
        "wns_ns": 2.218, "power_w": 0.624,
    },
    "integrated_width128": {
        "luts": 16444, "registers": 10701, "dsps_used": 405, "dsps_available": 740,
        "wns_ns": 2.218, "power_w": 0.888,
    },
    "integrated_width256": {
        "luts": 30660, "registers": 20046, "dsps_used": 405, "dsps_available": 740,
        "wns_ns": 2.218, "power_w": 0.940,
    },
    "dsp_headroom_pct": (740 - 405) / 740 * 100.0,
    "source": "hardware/vhdl_conv3x3/reports/first_layer_16out_folded_arithmetic_core_streaming_front_end_summary.md",
}

# The prior, DSP-SATURATED 32-output folded core (commit `313558e4`) --
# cited here ONLY for narrative contrast in the markdown summary (why the
# 16-output subproblem is the "comfortable" one).
SYNTHESIZED_32OUT_CONTEXT = {
    "dsps_used": 740, "dsps_available": 740, "dsp_headroom_pct": 0.0,
    "source": "hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_summary.md",
}


def load_folded_weights_16out(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads the checkpoint, folds BatchNorm into the first Conv2d layer's
    weights/bias, keeps ONLY output channels 0-15, and returns
    (w_folded_flat[27, 16], b_folded[16]) on `device`. Checkpoint loading
    happens ONCE, here, OUTSIDE any timed region."""
    print(f"[1] Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    w = state_dict[CONV_KEY].float().numpy()  # (32, 3, 3, 3) = (oc, ic, kh, kw)
    bn_gamma = state_dict[f"{BN_PREFIX}.weight"].float().numpy()
    bn_beta = state_dict[f"{BN_PREFIX}.bias"].float().numpy()
    bn_mean = state_dict[f"{BN_PREFIX}.running_mean"].float().numpy()
    bn_var = state_dict[f"{BN_PREFIX}.running_var"].float().numpy()
    print(f"    Conv tensor key : {CONV_KEY}  shape={list(w.shape)}")
    print(f"    BatchNorm prefix: {BN_PREFIX}")

    print("\n[2] Folding BatchNorm into the first Conv2d layer's weights/bias, "
          f"then slicing to output channels 0-{NUM_OUT_CHANNELS - 1} of 32 ...")
    scale_bn = bn_gamma / np.sqrt(bn_var + BN_EPS)                 # (32,)
    w_folded = w * scale_bn[:, None, None, None]                   # (32, 3, 3, 3)
    b_folded = bn_beta - bn_mean * scale_bn                        # (32,)

    w_folded_16 = w_folded[:NUM_OUT_CHANNELS]                      # (16, 3, 3, 3)
    b_folded_16 = b_folded[:NUM_OUT_CHANNELS]                      # (16,)
    print(f"    scale_bn range (ch 0-15) : [{scale_bn[:NUM_OUT_CHANNELS].min():.6f}, "
          f"{scale_bn[:NUM_OUT_CHANNELS].max():.6f}]")
    print(f"    b_folded range (ch 0-15) : [{b_folded_16.min():.6f}, {b_folded_16.max():.6f}]")

    # Flatten each output channel's 27 weights (ic, kh, kw order) to match
    # a window flattened with the SAME (ic, kh, kw) order via .reshape(-1).
    w_folded_flat = w_folded_16.reshape(NUM_OUT_CHANNELS, 27).T    # (27, 16)

    w_t = torch.from_numpy(np.ascontiguousarray(w_folded_flat)).float().to(device)
    b_t = torch.from_numpy(b_folded_16).float().to(device)
    return w_t, b_t


def arithmetic_core_forward(windows: torch.Tensor, w_folded_flat: torch.Tensor,
                             b_folded: torch.Tensor) -> torch.Tensor:
    """windows: [N, 27]. Returns [N, 16] folded Conv-BN-ReLU outputs
    (channels 0-15 only)."""
    return torch.relu(windows @ w_folded_flat + b_folded)


def summarize_times_us(times_us: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(times_us),
        "median_us": statistics.median(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "p10_us": float(np.percentile(times_us, 10)),
        "p90_us": float(np.percentile(times_us, 90)),
    }


def benchmark_normal(
    windows: torch.Tensor,
    w_folded_flat: torch.Tensor,
    b_folded: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Measured, NORMAL PyTorch/CUDA dispatch timing. Includes real
    per-call Python/CUDA launch overhead, which is a REAL part of
    small-job latency, not an artifact to discard."""
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = arithmetic_core_forward(windows, w_folded_flat, b_folded)
        if device.type == "cuda":
            torch.cuda.synchronize()

        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(TIMED_ITERS)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(TIMED_ITERS)]
            for i in range(TIMED_ITERS):
                starts[i].record()
                _ = arithmetic_core_forward(windows, w_folded_flat, b_folded)
                ends[i].record()
            torch.cuda.synchronize()
            times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]  # ms -> us
        else:
            times_us = []
            for _ in range(TIMED_ITERS):
                t0 = time.perf_counter()
                _ = arithmetic_core_forward(windows, w_folded_flat, b_folded)
                t1 = time.perf_counter()
                times_us.append((t1 - t0) * 1e6)
    return summarize_times_us(times_us)


def benchmark_cudagraph(
    windows: torch.Tensor,
    w_folded_flat: torch.Tensor,
    b_folded: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float] | None, str]:
    """CUDA Graph capture + replay timing, if supported. Returns
    (stats_or_None, note). Replay collapses per-call CPU dispatch
    overhead into one graph-launch call -- this is a SENSITIVITY
    ANALYSIS of reduced-overhead timing, not a claim that ordinary
    PyTorch code runs this fast. Mirrors the SAME safe capture pattern
    already used by scripts/benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py
    (side-stream warmup before capture, per PyTorch's CUDA graph
    documentation). Falls back gracefully (returns None) if CUDA graphs
    are unavailable or capture fails for any reason."""
    if device.type != "cuda":
        return None, "CUDA not available on this device; CUDA Graph timing skipped."
    if not hasattr(torch.cuda, "CUDAGraph"):
        return None, "torch.cuda.CUDAGraph not available in this PyTorch build; skipped."

    try:
        static_windows = windows.clone()

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            with torch.no_grad():
                for _ in range(CUDAGRAPH_WARMUP_ITERS):
                    _ = arithmetic_core_forward(static_windows, w_folded_flat, b_folded)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_output = arithmetic_core_forward(static_windows, w_folded_flat, b_folded)

        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(CUDAGRAPH_REPLAY_ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(CUDAGRAPH_REPLAY_ITERS)]
        for i in range(CUDAGRAPH_REPLAY_ITERS):
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
        times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
        _ = static_output  # keep referenced; graph owns the memory
        return summarize_times_us(times_us), "OK"
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, f"CUDA Graph capture/replay failed: {exc!r}"


def fpga_stream_estimate_us(patch_size: int, clock_hz: float) -> tuple[int, float]:
    """Returns (total_cycles, latency_us) for the INTEGRATED streaming
    front end + 16-output folded arithmetic core, using the GHDL-CONFIRMED
    total_cycles = (H * W) + FPGA_STREAM_DRAIN_CYCLES formula (see module
    docstring). H = W = patch_size (square image)."""
    total_cycles = (patch_size * patch_size) + FPGA_STREAM_DRAIN_CYCLES
    latency_us = total_cycles / clock_hz * 1e6
    return total_cycles, latency_us


FPGA_LABEL_100MHZ = "FPGA (synthesis-supported)"
FPGA_LABEL_300MHZ = "FPGA (hypothetical estimate)"
GPU_LABEL_NORMAL = "GPU (measured, normal dispatch)"
GPU_LABEL_CUDAGRAPH = "GPU (measured, CUDA Graph replay)"


def winner(gpu_us: float, fpga_us: float, gpu_label: str, fpga_label: str) -> str:
    if gpu_us != gpu_us:  # NaN check
        return "N/A"
    return fpga_label if fpga_us < gpu_us else gpu_label


def main() -> None:
    print("=" * 78)
    print("Environment")
    print("=" * 78)
    print(f"torch version    : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available   : {cuda_available}")
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A (no CUDA device)"
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    print(f"GPU name         : {gpu_name}")
    print(f"CUDA version     : {cuda_version}")
    print(f"Device used      : {device}")
    print()

    w_folded_flat, b_folded = load_folded_weights_16out(device)
    print(f"    w_folded_flat shape: {tuple(w_folded_flat.shape)}  (on {device})")
    print(f"    b_folded shape     : {tuple(b_folded.shape)}  (on {device})")

    print("\n" + "=" * 78)
    print(f"Benchmarking 16-OUTPUT (channels 0-15) first-layer folded "
          f"Conv-BN-ReLU arithmetic (warmup={WARMUP_ITERS}, timed_iters={TIMED_ITERS})")
    print("Windows are pre-extracted and already resident on-device before timing "
          "(GPU side only -- the FPGA-side cycle model separately accounts for "
          "streaming every pixel in; see module docstring).")
    print("=" * 78)

    rows: list[dict[str, object]] = []
    cudagraph_note = None
    for patch_size in WORKLOAD_PATCH_SIZES:
        n_windows = (patch_size - 2) ** 2
        windows = torch.randn(n_windows, 27, dtype=torch.float32, device=device)

        normal_stats = benchmark_normal(windows, w_folded_flat, b_folded, device)
        cudagraph_stats, note = benchmark_cudagraph(windows, w_folded_flat, b_folded, device)
        if cudagraph_note is None:
            cudagraph_note = note

        gpu_normal_us = normal_stats["mean_us"]
        gpu_cudagraph_us = cudagraph_stats["mean_us"] if cudagraph_stats else float("nan")

        fpga_100_cycles, fpga_100_us = fpga_stream_estimate_us(patch_size, FPGA_CLOCKS_HZ["100mhz"])
        fpga_300_cycles, fpga_300_us = fpga_stream_estimate_us(patch_size, FPGA_CLOCKS_HZ["300mhz"])

        macs = n_windows * NUM_OUT_CHANNELS * 27

        gpu_windows_per_sec = n_windows / (gpu_normal_us * 1e-6)
        gpu_cudagraph_windows_per_sec = (
            n_windows / (gpu_cudagraph_us * 1e-6) if cudagraph_stats else float("nan")
        )
        fpga_100_windows_per_sec = n_windows / (fpga_100_us * 1e-6)
        fpga_300_windows_per_sec = n_windows / (fpga_300_us * 1e-6)

        gpu_macs_per_sec = macs / (gpu_normal_us * 1e-6)
        gpu_cudagraph_macs_per_sec = (
            macs / (gpu_cudagraph_us * 1e-6) if cudagraph_stats else float("nan")
        )
        fpga_100_macs_per_sec = macs / (fpga_100_us * 1e-6)
        fpga_300_macs_per_sec = macs / (fpga_300_us * 1e-6)

        row = {
            "patch_size": patch_size,
            "output_windows": n_windows,
            "gpu_normal_us": gpu_normal_us,
            "gpu_normal_median_us": normal_stats["median_us"],
            "gpu_normal_min_us": normal_stats["min_us"],
            "gpu_normal_max_us": normal_stats["max_us"],
            "gpu_cudagraph_us": gpu_cudagraph_us,
            "gpu_cudagraph_median_us": cudagraph_stats["median_us"] if cudagraph_stats else float("nan"),
            "fpga_100mhz_cycles": fpga_100_cycles,
            "fpga_100mhz_us": fpga_100_us,
            "fpga_300mhz_cycles": fpga_300_cycles,
            "fpga_300mhz_us_hypothetical": fpga_300_us,
            "gpu_windows_per_sec": gpu_windows_per_sec,
            "gpu_cudagraph_windows_per_sec": gpu_cudagraph_windows_per_sec,
            "fpga_100mhz_windows_per_sec": fpga_100_windows_per_sec,
            "fpga_300mhz_windows_per_sec_hypothetical": fpga_300_windows_per_sec,
            "macs": macs,
            "gpu_macs_per_sec": gpu_macs_per_sec,
            "gpu_cudagraph_macs_per_sec": gpu_cudagraph_macs_per_sec,
            "fpga_100mhz_macs_per_sec": fpga_100_macs_per_sec,
            "fpga_300mhz_macs_per_sec_hypothetical": fpga_300_macs_per_sec,
            "winner_100mhz_normal": winner(gpu_normal_us, fpga_100_us, GPU_LABEL_NORMAL, FPGA_LABEL_100MHZ),
            "winner_100mhz_cudagraph": winner(gpu_cudagraph_us, fpga_100_us, GPU_LABEL_CUDAGRAPH, FPGA_LABEL_100MHZ),
            "winner_300mhz_normal_hypothetical": winner(gpu_normal_us, fpga_300_us, GPU_LABEL_NORMAL, FPGA_LABEL_300MHZ),
            "winner_300mhz_cudagraph_hypothetical": winner(gpu_cudagraph_us, fpga_300_us, GPU_LABEL_CUDAGRAPH, FPGA_LABEL_300MHZ),
        }
        rows.append(row)

        cg_str = f"{gpu_cudagraph_us:.3f}" if cudagraph_stats else "N/A"
        print(f"  patch={patch_size:5d} windows={n_windows:8d}  "
              f"gpu_normal={gpu_normal_us:10.3f} us  gpu_cudagraph={cg_str:>10s} us  "
              f"fpga@100MHz={fpga_100_us:10.3f} us ({fpga_100_cycles} cycles)  "
              f"winner={row['winner_100mhz_normal']}")

    print(f"\nCUDA Graph status: {cudagraph_note}")

    # ---- Write CSV ----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[3] Writing CSV: {CSV_PATH}")
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- Write markdown summary ----------------------------------------------
    print(f"[4] Writing markdown summary: {MD_PATH}")
    write_markdown_summary(rows, cudagraph_note, cuda_available, gpu_name, cuda_version, device)
    print(f"    Written: {MD_PATH.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 78)
    print("Done. Outputs:")
    print(f"  {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)


def write_markdown_summary(
    rows: list[dict[str, object]],
    cudagraph_note: str,
    cuda_available: bool,
    gpu_name: str,
    cuda_version: str,
    device: torch.device,
) -> None:
    cudagraph_available = any(r["gpu_cudagraph_us"] == r["gpu_cudagraph_us"] for r in rows)  # not NaN

    def fmt(v: float, spec: str = ",.3f") -> str:
        return "N/A" if v != v else format(v, spec)  # NaN check

    ctx16 = SYNTHESIZED_16OUT_CONTEXT
    ctx32 = SYNTHESIZED_32OUT_CONTEXT

    timing_table_rows = "\n".join(
        f"| {r['patch_size']} | {r['output_windows']:,} | {fmt(r['gpu_normal_us'])} | "
        f"{fmt(r['gpu_cudagraph_us'])} | {r['fpga_100mhz_cycles']:,} | "
        f"{fmt(r['fpga_100mhz_us'])} | {fmt(r['fpga_300mhz_us_hypothetical'])} |"
        for r in rows
    )

    throughput_table_rows = "\n".join(
        f"| {r['patch_size']} | {r['output_windows']:,} | "
        f"{fmt(r['gpu_windows_per_sec'], ',.0f')} | {fmt(r['fpga_100mhz_windows_per_sec'], ',.0f')} | "
        f"{r['macs']:,} | {fmt(r['gpu_macs_per_sec'], ',.3e')} | {fmt(r['fpga_100mhz_macs_per_sec'], ',.3e')} |"
        for r in rows
    )

    winners_table_rows = "\n".join(
        f"| {r['patch_size']} | {r['output_windows']:,} | {r['winner_100mhz_normal']} | "
        f"{r['winner_100mhz_cudagraph']} | {r['winner_300mhz_normal_hypothetical']} | "
        f"{r['winner_300mhz_cudagraph_hypothetical']} |"
        for r in rows
    )

    def first_gpu_win(key: str):
        return next((r for r in rows if str(r[key]).startswith("GPU")), None)

    def crossover_sentence(crossover_row, clock_label, view_label, fpga_desc):
        if crossover_row is None:
            return (
                f"Under {view_label} at {clock_label}, the {fpga_desc} remains faster "
                f"across the ENTIRE tested workload range (up to "
                f"{rows[-1]['output_windows']:,} windows, patch_size={rows[-1]['patch_size']}) "
                f"-- no crossover observed."
            )
        return (
            f"Under {view_label} at {clock_label}, GPU becomes faster than the "
            f"{fpga_desc} at or before **{crossover_row['output_windows']:,} windows** "
            f"(patch_size={crossover_row['patch_size']})."
        )

    FPGA_DESC_100MHZ = "100 MHz synthesis-supported FPGA cycle timing"
    FPGA_DESC_300MHZ = "300 MHz hypothetical FPGA estimate"

    cudagraph_status_line = (
        f"CUDA Graph capture and replay **succeeded** on this run ({cudagraph_note})."
        if cudagraph_available else
        f"CUDA Graph capture/replay was **not available or did not succeed** on this run "
        f"({cudagraph_note}). The CUDA Graph columns below are reported as N/A; all other "
        f"analysis (measured normal GPU latency, FPGA cycle timing) is unaffected."
    )

    small_rows = [r for r in rows if r["patch_size"] <= 32]
    tile_rows = [r for r in rows if r["patch_size"] in (128, 256)]
    large_rows = [r for r in rows if r["patch_size"] >= 512]

    def regime_block(regime_rows: list[dict], label: str) -> str:
        lines = []
        for r in regime_rows:
            lines.append(
                f"- patch={r['patch_size']}: GPU normal = {fmt(r['gpu_normal_us'])} us, "
                f"FPGA@100MHz = {fmt(r['fpga_100mhz_us'])} us "
                f"({r['fpga_100mhz_cycles']:,} cycles) -- **{r['winner_100mhz_normal']} faster** "
                f"(normal GPU dispatch)."
            )
        return f"**{label}:**\n" + "\n".join(lines)

    md = f"""# First-Layer 16-Output Folded Core: GPU (Measured) vs. FPGA (Synthesis-Supported Streaming Cycle Timing)

Generated by `scripts/benchmark_first_layer_16out_gpu_vs_fpga.py`.

This report compares a REAL GPU (measured) against the NEW, comfortably-
fitting 16-output folded Conv-BN-ReLU streaming first-layer FPGA core
(commit `fe79e096`) for the EXACT SAME arithmetic: channels 0-15 of the
real trained `enc1.block.0` Conv2d weights, folded with `enc1.block.1`
BatchNorm, ReLU applied. It does NOT modify or replace the existing
32-output GPU-vs-FPGA benchmark
(`scripts/benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py`,
`outputs/hardware_benchmarks/first_layer_arithmetic_core_gpu_vs_fpga/`) --
it is a new, independent comparison for the smaller subproblem.

## 1. What exactly was compared

For N pre-extracted 3x3x3 input windows (flattened to `[N, 27]`):

```
output[N, 16] = ReLU( windows[N, 27] @ w_folded[27, 16] + b_folded[16] )
```

- `w_folded`/`b_folded`: real trained `enc1.block.0.weight` ([32, 3, 3,
  3]) folded with `enc1.block.1` (BatchNorm2d, eval-mode running stats),
  **keeping ONLY output channels 0-15 of 32** -- the identical folding
  formula used throughout this repo (`scale_bn = gamma / sqrt(running_var
  + eps)`, `w_folded = w * scale_bn`, `b_folded = beta - running_mean *
  scale_bn`).
- Checkpoint: `models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt`.
- GPU side: this exact matmul+bias+ReLU, MEASURED on-device with
  `torch.cuda.Event` timing (Section 5).
- FPGA side: cycle-count timing derived from the NEW 16-output integrated
  streaming front end + folded arithmetic core (commit `fe79e096`),
  GHDL-verified (256/256 outputs match real UAVSAR-derived golden
  vectors) and Vivado-synthesized at 100 MHz on Artix-7 200T (Section 6).

## 2. Definitions: "patch," "window," and "tile"

- **patch_size**: side length (pixels) of a square `H x W` input image
  (e.g. `patch_size=128` means a 128x128 pixel input region).
- **window**: one 3x3x3 (9 pixels x 3 input channels) VALID (no padding)
  convolution input, at one output spatial position. A `patch_size x
  patch_size` image yields `(patch_size - 2) ** 2` valid windows.
- **tile**: used interchangeably with "patch" here -- the same square
  `H x W` input image. 128 and 256 are the realistic tile sizes already
  used throughout this repo's real-tile verification work (the
  underlying source tile, `tile_16_42.tif`, is 256x256).

## 3. Why 16-output is "the comfortable FPGA subproblem"

The existing 32-output folded core (commit `313558e4`) saturates the
Artix-7 200T at **{ctx32['dsps_used']}/{ctx32['dsps_available']} DSP48E1
slices (100%)** -- zero headroom (source: `{ctx32['source']}`). The NEW
16-output core (commit `fe79e096`, channels 0-15 only, same real
checkpoint weights, same folded BatchNorm + ReLU, same Q.20 fixed-point
convention on the hardware side) uses only
**{ctx16['core_only']['dsps_used']}/{ctx16['core_only']['dsps_available']}
DSP48E1 slices ({100 - ctx16['dsp_headroom_pct']:.2f}%)** -- **{ctx16['dsp_headroom_pct']:.2f}%
({ctx16['core_only']['dsps_available'] - ctx16['core_only']['dsps_used']}
DSPs) of real, synthesis-measured headroom** on the SAME part, while
still meeting 100 MHz timing (WNS = +{ctx16['core_only']['wns_ns']} ns at
every width tested: core-only, IMG_WIDTH=128, IMG_WIDTH=256 -- source:
`{ctx16['source']}`). GHDL confirmed **{ctx16['ghdl_pass']}/{ctx16['ghdl_total']}**
outputs match real UAVSAR-derived golden vectors, streamed end-to-end
through the real streaming front end. This makes the 16-output design a
genuinely comfortably-fitting, real-model subproblem -- not maxed out --
and therefore a cleaner point of comparison against GPU than a design
that is already at its resource ceiling.

## 4. FPGA timing model: synthesis-supported vs. hypothetical

- **100 MHz -- SYNTHESIS-SUPPORTED.** The exact clock the 16-output
  integrated streaming design (front end + arithmetic core) was
  Vivado-synthesized against and closed timing on, at IMG_WIDTH=128 AND
  IMG_WIDTH=256 (source: `{ctx16['source']}`). Cycle count uses the
  GHDL-CONFIRMED streaming formula
  `total_cycles = (patch_size * patch_size) + {FPGA_STREAM_DRAIN_CYCLES}`
  (measured directly by GHDL for a real 6x6 image: 36 pixels + 6 = 42
  cycles -- source:
  `hardware/vhdl_conv3x3/reports/direct_parallel_line_buffer_feasibility_summary.md`).
  This is STILL a cycle-count estimate (`cycles / clock_hz`), **NOT a
  board-measured latency** -- but the clock rate, cycle formula, and
  "does this actually fit and close timing" question are synthesis-
  confirmed, not assumed.
- **300 MHz -- CLEARLY HYPOTHETICAL.** No synthesis run in this repo --
  for the 16-output design or any other -- has confirmed 300 MHz timing
  closure. Reported only as an illustrative upper bound; **the conclusion
  of this report is NOT centered on this number.**
- Why `(H*W) + 6`, not `windows + 6`: this report compares against the
  INTEGRATED streaming design, which must pay the real cost of streaming
  every pixel of the image in (not just the valid windows) before the
  last window's outputs are available -- the same correction already
  applied to the 32-output design's own tile-throughput reporting.

{cudagraph_status_line}

## 5. GPU timing method

- PyTorch on CUDA (falls back to CPU wall-clock timing if unavailable --
  see Section 6 for which was used in this run).
- Checkpoint loaded and BatchNorm folded ONCE, before any timed region.
- For each workload size N, a `[N, 27]` random input tensor is created and
  moved to the GPU BEFORE timing begins (no data movement inside the
  timed region).
- **Normal dispatch**: {WARMUP_ITERS} warmup iterations (untimed), then
  {TIMED_ITERS} timed iterations; `torch.cuda.Event` start/end pairs are
  recorded without synchronizing in between, then a single
  `torch.cuda.synchronize()` before reading elapsed times. Includes real
  per-call Python/CUDA dispatch overhead -- this is the number that
  answers "how long does one real, standalone small job actually take."
- **CUDA Graph replay**: {CUDAGRAPH_WARMUP_ITERS} warmup iterations on a
  side CUDA stream (per PyTorch's CUDA graph capture requirements), then
  the op is captured once via `torch.cuda.graph(...)` and replayed
  {CUDAGRAPH_REPLAY_ITERS} times with the same event-timing convention.
  Collapses per-call CPU dispatch overhead into one graph-launch cost --
  a SENSITIVITY ANALYSIS of reduced-overhead timing, not a claim that
  ordinary PyTorch code runs this fast for a one-off call.

## Environment (this run)

| Field | Value |
|---|---|
| torch version | {torch.__version__} |
| CUDA available | {cuda_available} |
| GPU name | {gpu_name} |
| CUDA version | {cuda_version} |
| Device used | {device} |
| Warmup iterations (normal) | {WARMUP_ITERS} |
| Timed iterations (normal) | {TIMED_ITERS} |
| CUDA Graph warmup / replay iterations | {CUDAGRAPH_WARMUP_ITERS} / {CUDAGRAPH_REPLAY_ITERS} |
| Input dtype | float32 |
| Input shape | [N, 27] -> output [N, 16] |

## 6. Timing table

| Patch | Windows | GPU normal (us) | GPU CUDA-Graph (us) | FPGA@100MHz cycles | FPGA@100MHz (us) | FPGA@300MHz (us, hypothetical) |
|---|---:|---:|---:|---:|---:|---:|
{timing_table_rows}

## 7. Throughput table: windows/sec and effective MACs/sec

`MACs = windows * {NUM_OUT_CHANNELS} * 27` (16 output channels x 27 taps
per window). MACs/sec computed from the SAME latency figures as Section 6
(GPU normal dispatch; FPGA at 100 MHz synthesis-supported cycle timing).

| Patch | Windows | GPU windows/sec | FPGA@100MHz windows/sec | MACs | GPU MACs/sec | FPGA@100MHz MACs/sec |
|---|---:|---:|---:|---:|---:|---:|
{throughput_table_rows}

## 8. Winner by view

| Patch | Windows | Winner @100MHz (GPU normal) | Winner @100MHz (GPU CUDA-Graph) | Winner @300MHz hypothetical (GPU normal) | Winner @300MHz hypothetical (GPU CUDA-Graph) |
|---|---:|---|---|---|---|
{winners_table_rows}

## 9. Crossover: is there a workload size where GPU becomes faster?

**Under measured, normal GPU dispatch latency** (the real small-job-latency view):

{crossover_sentence(first_gpu_win("winner_100mhz_normal"), "100 MHz", "measured normal GPU latency", FPGA_DESC_100MHZ)}

{crossover_sentence(first_gpu_win("winner_300mhz_normal_hypothetical"), "300 MHz", "measured normal GPU latency", FPGA_DESC_300MHZ)}

**Under CUDA Graph replay timing** (reduced-overhead sensitivity view):

{crossover_sentence(first_gpu_win("winner_100mhz_cudagraph"), "100 MHz", "CUDA Graph replay timing", FPGA_DESC_100MHZ)}

{crossover_sentence(first_gpu_win("winner_300mhz_cudagraph_hypothetical"), "300 MHz", "CUDA Graph replay timing", FPGA_DESC_300MHZ)}

## 10. Interpretation: tiny launch overhead vs. realistic tile scale vs. large-workload throughput

This section deliberately separates three regimes rather than reporting
one blended "FPGA wins" or "GPU wins" conclusion.

### 10a. Tiny-patch launch-overhead regime (patch_size <= 32)

{regime_block(small_rows, "patch 6-32")}

At this scale, GPU's fixed per-call Python/CUDA dispatch overhead is a
large fraction of total measured time, while the FPGA-side streaming
cycle model (`(H*W)+6` cycles at 100 MHz) has no comparable per-call
dispatch cost. Any FPGA advantage observed here is a launch-overhead
result, not a raw-arithmetic-throughput result.

### 10b. Realistic tile-scale regime (patch_size = 128, 256)

{regime_block(tile_rows, "patch 128/256")}

These are the SAME tile sizes already used throughout this repo's
real-tile verification and synthesis work (the 16-output integrated
design was Vivado-synthesized and closed 100 MHz timing at exactly these
two widths). This regime is the most representative of realistic
per-tile inference cost for this specific first-layer subproblem.

### 10c. Large-workload throughput regime (patch_size >= 512)

{regime_block(large_rows, "patch 512/1024")}

At large N, GPU's massively parallel matmul throughput amortizes its
fixed dispatch overhead across many windows simultaneously, while the
FPGA streaming model's total time grows almost linearly with
`patch_size^2` (one pixel per cycle, fundamentally sequential streaming
ingestion). This regime is the fairest test of raw throughput
competitiveness, with launch overhead minimized on the GPU side.

### 10d. Is this mostly a launch-overhead result, or a real throughput result?

Compare the FPGA's steady-state cycle rate (100,000,000 cycles/sec, i.e.
1 pixel/cycle at 100 MHz) against the GPU's measured windows/sec at the
LARGEST tested workload (patch_size={rows[-1]['patch_size']}): GPU
achieves **{fmt(rows[-1]['gpu_windows_per_sec'], ',.0f')} windows/sec**
(normal dispatch) versus the FPGA model's
**{fmt(rows[-1]['fpga_100mhz_windows_per_sec'], ',.0f')} windows/sec** at
100 MHz. Whichever side is larger at this workload size answers the
throughput-competitiveness question honestly, since dispatch overhead is
least influential here.

## 11. Claim boundaries

- **GPU timing is MEASURED** (normal dispatch, Section 5; CUDA Graph
  replay as a sensitivity view), on this specific GPU (see Environment
  table).
- **FPGA 100 MHz timing is cycle-count derived, but SYNTHESIS-SUPPORTED**:
  the underlying number is `cycles / clock_hz`, using the GHDL-confirmed
  `(H*W)+{FPGA_STREAM_DRAIN_CYCLES}` streaming cycle formula and the
  Vivado-confirmed 100 MHz timing closure of the 16-output integrated
  streaming design (WNS = +{ctx16['core_only']['wns_ns']} ns at
  IMG_WIDTH=128 and IMG_WIDTH=256). **This is NOT a board measurement.**
- **FPGA 300 MHz timing is purely hypothetical and UNSYNTHESIZED** -- no
  design in this repo has been synthesized or verified to close timing at
  300 MHz. It is included only as an illustrative upper bound and does
  NOT drive this report's conclusion.
- **This is not full U-Net FPGA inference** -- one Conv2d+BatchNorm+ReLU
  stage, 16 of its 32 output channels only.
- **No board-measured FPGA speedup or power is claimed anywhere in this
  report.**
- **This does not claim FPGA is faster than GPU in general** -- only that,
  for this specific comfortably-fitting first-layer subproblem, FPGA may
  be competitive or faster under normal GPU dispatch latency at small
  workload sizes, while GPU is expected to dominate raw throughput at
  large workload sizes (Section 10 reports the measured/derived numbers
  for both regimes rather than asserting a single winner).
- **This does not retrain, fine-tune, or otherwise modify the
  checkpoint.**
- **The existing 32-output GPU-vs-FPGA benchmark and its outputs are
  unmodified** by this script -- this is an independent comparison for
  the 16-output subproblem only.
"""
    MD_PATH.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
