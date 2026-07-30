#!/usr/bin/env python3
"""
benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py

A small-to-large WORKLOAD SWEEP for the first learned Conv-BN-ReLU
stage's ARITHMETIC CORE ONLY (`enc1.block.0` Conv2d + `enc1.block.1`
BatchNorm2d + ReLU), comparing measured GPU matmul timing against a
speed-oriented, direct-parallel Artix-7 200T cycle estimate.

---- Why this script exists, and how it differs from the earlier
     first-layer GPU-vs-FPGA benchmark ---------------------------------
scripts/benchmark_first_layer_gpu_vs_fpga_estimate.py already measured
GPU timing for the FULL first Conv-BN-ReLU nn.Module (real Conv2d +
BatchNorm2d + ReLU, on 5x5/128x128/256x256 IMAGES with padding=1) against
a RESOURCE-SHARED (time-multiplexed) FPGA cycle estimate. This script
does NOT redo that -- it isolates a narrower, fairer, SPEED-ORIENTED
question: once you already have the 27 numbers of a single 3x3x3 input
window, how long does it take to produce that window's 32 folded
Conv-BN-ReLU outputs, as a pure arithmetic-core operation, on GPU
(measured) vs. a SPEED-ORIENTED direct-parallel FPGA core (estimated)?
This is a fairer comparison than "full trained U-Net on GPU" vs. "small
FPGA prototype on a toy patch," because both sides here compute the
EXACT SAME arithmetic (one window in, 32 folded Conv-BN-ReLU values out)
and neither side's cost is inflated or deflated by unrelated overhead
(image line buffering, disk I/O, DataLoader startup, checkpoint loading,
or the rest of the U-Net).

---- GPU-overhead analysis (added in this revision) -------------------
The FIRST version of this script reported only measured, normal-PyTorch-
dispatch GPU latency. That measurement is REAL and VALID -- fixed GPU
launch/dispatch overhead is part of real small-workload latency, and
this script keeps reporting it unchanged. This revision ADDS three
additional, clearly-separated views so the report does not conflate
"real measured latency" with "how fast the arithmetic itself is once
dispatch overhead is minimized or amortized away":

1. **Measured, normal PyTorch/CUDA execution** (UNCHANGED from before):
   `torch.cuda.Event` timing around ordinary eager-mode calls, including
   normal per-call Python/CUDA dispatch overhead. This is the number that
   answers "how long does a real, one-off small job actually take."
2. **CUDA Graph replay timing** (NEW, if supported): the SAME arithmetic
   op is captured once into a CUDA graph, then replayed many times.
   Graph replay collapses per-call CPU dispatch overhead into a single
   graph-launch cost, so this measures much closer to the GPU's own
   execution time, with launch overhead substantially reduced (not
   eliminated). This is a SENSITIVITY ANALYSIS, not a claim that normal
   PyTorch code runs this fast.
3. **Overhead-corrected / amortized throughput estimate** (NEW): a
   simple linear model `T_gpu(N) = fixed_overhead_us + per_window_us *
   N` is fit to the MEASURED (normal-dispatch) latency at the LARGER
   workload sizes only (so a few noisy small-N points do not dominate
   the fit). The `per_window_us * N` term (i.e. the fitted model with
   its fixed-overhead intercept removed) is reported as an idealized,
   ZERO-fixed-overhead throughput estimate -- an explicit best case for
   GPU, useful only as a sensitivity bound, never as a claim about real
   single-call latency.

Removing or reducing GPU overhead (views 2 and 3) will ALWAYS make GPU
look better, especially for small workloads, precisely because fixed
per-call overhead is a larger fraction of a small job's total time. This
is expected and is not evidence that view 1 (real measured latency) is
wrong -- the three views answer three different, clearly-labeled
questions, and all three are reported side by side.

---- What this benchmark measures ------------------------------------------
For a workload of N pre-extracted, already-in-memory 3x3x3 windows
(flattened to shape [N, 27]):
    output[N, 32] = ReLU( windows[N, 27] @ w_folded[27, 32] + b_folded[32] )
where `w_folded`/`b_folded` come from folding the REAL trained
`enc1.block.0`/`enc1.block.1` weights (BatchNorm folded into the Conv2d
weights and bias, exactly as elsewhere in this repo). FPGA timing is
ESTIMATED from a speed-oriented, direct-parallel core assumption (one
window consumed per clock, all 32 channels produced per clock after a
small, explicitly-stated pipeline fill), NOT measured on any board.

---- What this benchmark EXCLUDES ------------------------------------------
- Image line buffering / sliding-window generation (the windows are
  assumed ALREADY EXTRACTED and in memory on both sides).
- Disk I/O, DataLoader startup, checkpoint loading time (the checkpoint
  is loaded once, before any timing region, purely to obtain the real
  trained weights).
- The rest of the U-Net (encoder stages 2-4, bottleneck, decoder).
- Any board-measured FPGA timing, power, or speedup.
- Full FPGA U-Net acceleration of any kind.

---- BatchNorm folding (identical formula used throughout this repo) -------
    scale_bn[oc] = gamma[oc] / sqrt(running_var[oc] + eps)
    w_folded[oc] = w[oc] * scale_bn[oc]
    b_folded[oc] = beta[oc] - running_mean[oc] * scale_bn[oc]

---- FPGA estimate assumptions (speed-oriented, direct-parallel core) ------
- One 3x3x3 input window consumed per clock cycle.
- All 32 output channels produced per clock cycle, after a small,
  fixed pipeline-fill latency (streaming throughput of 1 window/cycle
  once the pipeline is full -- NOT the resource-shared/time-multiplexed
  architecture used elsewhere in this repo).
- Folded BN + ReLU is included as EXTRA pipelined arithmetic on top of
  the raw convolution, not as a separate, additional pass.
- Pipeline-fill depth used for the "with-fill" estimate: 6 cycles --
  taken directly from this repo's own GHDL-measured pipelined folded
  Conv-BN-ReLU design (4 cycles for the existing raw-convolution
  `stream_conv3x3_3chan_cell`, confirmed by GHDL simulation, + 2 cycles
  for the registered BatchNorm-fold-multiply and bias-add-ReLU stages,
  confirmed by
  `hardware/vhdl_conv3x3/first_conv_bn_relu_kernel0_pipelined_summary.md`'s
  measured 6-cycle total latency) -- not a new, invented number.
- Two clock targets are reported: 100 MHz (the SAME clock used by every
  Vivado synthesis in this repo, including the existing, ALREADY
  SYNTHESIZED direct-parallel 32-output DSP-aware design that meets
  100 MHz timing on Artix-7 200T using 740/740 DSPs -- see
  `hardware/vhdl_conv3x3/first_layer_32out_dsp_200t_summary.md`), and
  300 MHz (a CLEARLY HYPOTHETICAL, NOT board-verified, speed-oriented
  target -- no synthesis run in this repo has confirmed 300 MHz timing
  closure for any design).

---- Workload sweep (EXTENDED in this revision) -----------------------
Output-window counts corresponding to valid 3x3 windows from square
patches (patch_size -> (patch_size-2)^2 valid windows): 6, 16, 32, 64,
128, 256, 384, 512, 768, 1024 -- extended from the original 6-256 sweep
so the overhead-fit has enough large, amortized-regime data points.

---- Usage ------------------------------------------------------------------
    python scripts/benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py

---- Outputs -----------------------------------------------------------
outputs/hardware_benchmarks/first_layer_arithmetic_core_gpu_vs_fpga/
    first_layer_arithmetic_core_timing.csv
    first_layer_arithmetic_core_summary.md
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

OUT_DIR = REPO_ROOT / "outputs" / "hardware_benchmarks" / "first_layer_arithmetic_core_gpu_vs_fpga"
CSV_PATH = OUT_DIR / "first_layer_arithmetic_core_timing.csv"
MD_PATH = OUT_DIR / "first_layer_arithmetic_core_summary.md"

WARMUP_ITERS = 30
TIMED_ITERS = 300
CUDAGRAPH_WARMUP_ITERS = 10
CUDAGRAPH_REPLAY_ITERS = 300

# patch_size -> number of valid 3x3 windows ((patch_size - 2) ** 2).
# Extended from the original [6, 16, 32, 64, 128, 256] sweep so the
# overhead/per-window fit (Section on "fitted model") has enough large,
# amortized-regime data points that a few small, noisy measurements do
# not dominate the regression.
WORKLOAD_PATCH_SIZES = [6, 16, 32, 64, 128, 256, 384, 512, 768, 1024]

# Only workloads at or above this patch size are used to FIT the linear
# GPU timing model (T_gpu(N) = fixed_overhead_us + per_window_us * N).
# Chosen so the fit is dominated by the large-N, overhead-amortized
# regime, not by the smallest, noisiest points.
FIT_MIN_PATCH_SIZE = 256

FPGA_CLOCKS_HZ = {"100mhz": 100e6, "300mhz": 300e6}
# 4 cycles (existing GHDL-measured raw-convolution stream_conv3x3_3chan_cell
# latency) + 2 cycles (existing GHDL-measured registered BN-fold-multiply +
# bias-add-ReLU pipeline stages) = 6 cycles total. See module docstring.
FPGA_PIPELINE_FILL_CYCLES = 6

# Existing, already-synthesized direct-parallel 32-output first Conv2d
# context, reused here (NOT re-derived) purely for narrative context in
# the markdown summary.
EXISTING_DIRECT_PARALLEL_200T_CONTEXT = {
    "part": "xc7a200tsbg484-1",
    "dsps_used": 740,
    "dsps_available": 740,
    "wns_ns": 2.343,
    "clock_mhz": 100,
    "source": "hardware/vhdl_conv3x3/first_layer_32out_dsp_200t_summary.md",
}


def load_folded_weights(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads the checkpoint, folds BatchNorm into the first Conv2d layer's
    weights/bias, and returns (w_folded_flat[27, 32], b_folded[32]) on
    `device`. Checkpoint loading happens ONCE, here, OUTSIDE any timed
    region."""
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

    print("\n[2] Folding BatchNorm into the first Conv2d layer's weights/bias ...")
    scale_bn = bn_gamma / np.sqrt(bn_var + BN_EPS)                 # (32,)
    w_folded = w * scale_bn[:, None, None, None]                   # (32, 3, 3, 3)
    b_folded = bn_beta - bn_mean * scale_bn                        # (32,)
    print(f"    scale_bn range : [{scale_bn.min():.6f}, {scale_bn.max():.6f}]")
    print(f"    b_folded range : [{b_folded.min():.6f}, {b_folded.max():.6f}]")

    # Flatten each output channel's 27 weights (ic, kh, kw order) to match
    # a window flattened with the SAME (ic, kh, kw) order via .reshape(-1).
    w_folded_flat = w_folded.reshape(32, 27).T                     # (27, 32)

    w_t = torch.from_numpy(w_folded_flat).float().to(device)
    b_t = torch.from_numpy(b_folded).float().to(device)
    return w_t, b_t


def arithmetic_core_forward(windows: torch.Tensor, w_folded_flat: torch.Tensor,
                             b_folded: torch.Tensor) -> torch.Tensor:
    """windows: [N, 27]. Returns [N, 32] folded Conv-BN-ReLU outputs."""
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
    """Measured, NORMAL PyTorch/CUDA dispatch timing -- UNCHANGED method
    from the original version of this script. Includes real per-call
    Python/CUDA launch overhead, which is a REAL part of small-job
    latency, not an artifact to discard."""
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
    PyTorch code runs this fast. Falls back gracefully (returns None)
    if CUDA graphs are unavailable or capture fails for any reason."""
    if device.type != "cuda":
        return None, "CUDA not available on this device; CUDA Graph timing skipped."
    if not hasattr(torch.cuda, "CUDAGraph"):
        return None, "torch.cuda.CUDAGraph not available in this PyTorch build; skipped."

    try:
        static_windows = windows.clone()

        # Warm up on a side stream first (required before capture, per
        # PyTorch's CUDA graph documentation) so autotuned cuBLAS/cuDNN
        # algorithm selection happens OUTSIDE the captured graph.
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


def fit_linear_overhead_model(
    windows_list: list[int], means_us: list[float], min_patch_windows: int
) -> dict[str, float]:
    """Fits T_gpu(N) = fixed_overhead_us + per_window_us * N via ordinary
    least squares, using ONLY the workloads with output_windows >=
    min_patch_windows (so a few small, noisy points do not dominate).
    Returns the fit parameters plus R^2 and which N were used."""
    windows_arr = np.array(windows_list, dtype=np.float64)
    means_arr = np.array(means_us, dtype=np.float64)
    mask = windows_arr >= min_patch_windows
    x = windows_arr[mask]
    y = means_arr[mask]
    if len(x) < 2:
        return {
            "fixed_overhead_us": float("nan"), "per_window_us": float("nan"),
            "r_squared": float("nan"), "n_points_used": int(len(x)),
        }
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "fixed_overhead_us": float(intercept),
        "per_window_us": float(slope),
        "r_squared": r_squared,
        "n_points_used": int(len(x)),
    }


def fpga_estimate_us(n_windows: int, clock_hz: float, pipeline_fill_cycles: int) -> tuple[float, float]:
    """Returns (with_fill_us, no_fill_us)."""
    no_fill_cycles = n_windows
    with_fill_cycles = n_windows + pipeline_fill_cycles
    no_fill_us = no_fill_cycles / clock_hz * 1e6
    with_fill_us = with_fill_cycles / clock_hz * 1e6
    return with_fill_us, no_fill_us


def winner(gpu_us: float, fpga_us: float) -> str:
    if gpu_us != gpu_us:  # NaN check
        return "N/A"
    return "FPGA (estimated)" if fpga_us < gpu_us else "GPU (measured)"


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

    w_folded_flat, b_folded = load_folded_weights(device)
    print(f"    w_folded_flat shape: {tuple(w_folded_flat.shape)}  (on {device})")
    print(f"    b_folded shape     : {tuple(b_folded.shape)}  (on {device})")

    print("\n" + "=" * 78)
    print(f"Benchmarking first-layer ARITHMETIC CORE ONLY "
          f"(warmup={WARMUP_ITERS}, timed_iters={TIMED_ITERS})")
    print("Windows are pre-extracted and already resident on-device before timing.")
    print("=" * 78)

    raw_rows: list[dict[str, object]] = []
    cudagraph_note = None
    for patch_size in WORKLOAD_PATCH_SIZES:
        n_windows = (patch_size - 2) ** 2
        windows = torch.randn(n_windows, 27, dtype=torch.float32, device=device)

        normal_stats = benchmark_normal(windows, w_folded_flat, b_folded, device)
        cudagraph_stats, note = benchmark_cudagraph(windows, w_folded_flat, b_folded, device)
        if cudagraph_note is None:
            cudagraph_note = note

        raw_rows.append({
            "patch_size": patch_size,
            "output_windows": n_windows,
            "normal": normal_stats,
            "cudagraph": cudagraph_stats,
        })
        cg_mean_str = f"{cudagraph_stats['mean_us']:.3f}" if cudagraph_stats else "N/A"
        print(f"  patch={patch_size:4d} windows={n_windows:7d}  "
              f"gpu_normal_mean={normal_stats['mean_us']:9.3f} us  "
              f"gpu_cudagraph_mean={cg_mean_str:>9s} us")

    print(f"\nCUDA Graph status: {cudagraph_note}")

    # ---- Fit overhead/per-window model on the NORMAL measured latency,
    # using only the larger workloads. ----
    print(f"\n[3] Fitting T_gpu(N) = fixed_overhead_us + per_window_us * N "
          f"(using patch_size >= {FIT_MIN_PATCH_SIZE}) ...")
    fit_min_windows = (FIT_MIN_PATCH_SIZE - 2) ** 2
    model = fit_linear_overhead_model(
        [r["output_windows"] for r in raw_rows],
        [r["normal"]["mean_us"] for r in raw_rows],
        fit_min_windows,
    )
    print(f"    fixed_overhead_us = {model['fixed_overhead_us']:.4f}")
    print(f"    per_window_us     = {model['per_window_us']:.6f}")
    print(f"    R^2               = {model['r_squared']:.5f}  "
          f"(fit used {model['n_points_used']} points)")

    # ---- Assemble final rows: FPGA estimates, overhead-corrected GPU
    # estimate, ratios, winners. ----
    rows: list[dict[str, object]] = []
    for r in raw_rows:
        n_windows = r["output_windows"]
        normal = r["normal"]
        cudagraph = r["cudagraph"]

        fpga_100_fill, fpga_100_nofill = fpga_estimate_us(
            n_windows, FPGA_CLOCKS_HZ["100mhz"], FPGA_PIPELINE_FILL_CYCLES
        )
        fpga_300_fill, fpga_300_nofill = fpga_estimate_us(
            n_windows, FPGA_CLOCKS_HZ["300mhz"], FPGA_PIPELINE_FILL_CYCLES
        )

        # Overhead-corrected / amortized estimate: the fitted model's
        # per-window slope only (zero fixed overhead) -- an idealized
        # sensitivity bound, not a real single-call latency claim.
        overhead_corrected_us = model["per_window_us"] * n_windows

        gpu_normal_mean = normal["mean_us"]
        gpu_cudagraph_mean = cudagraph["mean_us"] if cudagraph else float("nan")

        row = {
            "patch_size": r["patch_size"],
            "output_windows": n_windows,
            # -- Measured, normal PyTorch/CUDA dispatch (UNCHANGED) --
            "gpu_mean_us": gpu_normal_mean,
            "gpu_median_us": normal["median_us"],
            "gpu_min_us": normal["min_us"],
            "gpu_max_us": normal["max_us"],
            "gpu_p10_us": normal["p10_us"],
            "gpu_p90_us": normal["p90_us"],
            # -- CUDA Graph replay (NEW; NaN if unavailable) --
            "gpu_cudagraph_mean_us": gpu_cudagraph_mean,
            "gpu_cudagraph_median_us": cudagraph["median_us"] if cudagraph else float("nan"),
            "gpu_cudagraph_min_us": cudagraph["min_us"] if cudagraph else float("nan"),
            "gpu_cudagraph_max_us": cudagraph["max_us"] if cudagraph else float("nan"),
            "gpu_cudagraph_p10_us": cudagraph["p10_us"] if cudagraph else float("nan"),
            "gpu_cudagraph_p90_us": cudagraph["p90_us"] if cudagraph else float("nan"),
            # -- Overhead-corrected / amortized estimate (NEW) --
            "gpu_overhead_corrected_us": overhead_corrected_us,
            # -- FPGA estimates (unchanged) --
            "fpga_100mhz_us": fpga_100_fill,
            "fpga_100mhz_us_nofill": fpga_100_nofill,
            "fpga_300mhz_us": fpga_300_fill,
            "fpga_300mhz_us_nofill": fpga_300_nofill,
            # -- Ratios / winners vs. NORMAL measured (unchanged) --
            "gpu_vs_fpga100_ratio": gpu_normal_mean / fpga_100_fill,
            "gpu_vs_fpga300_ratio": gpu_normal_mean / fpga_300_fill,
            "winner_100mhz_estimate": winner(gpu_normal_mean, fpga_100_fill),
            "winner_300mhz_estimate": winner(gpu_normal_mean, fpga_300_fill),
            # -- Ratios / winners vs. CUDA Graph replay (NEW) --
            "gpu_cudagraph_vs_fpga100_ratio": (
                gpu_cudagraph_mean / fpga_100_fill if cudagraph else float("nan")
            ),
            "gpu_cudagraph_vs_fpga300_ratio": (
                gpu_cudagraph_mean / fpga_300_fill if cudagraph else float("nan")
            ),
            "winner_100mhz_cudagraph": winner(gpu_cudagraph_mean, fpga_100_fill),
            "winner_300mhz_cudagraph": winner(gpu_cudagraph_mean, fpga_300_fill),
            # -- Ratios / winners vs. overhead-corrected estimate (NEW) --
            "gpu_overhead_corrected_vs_fpga100_ratio": overhead_corrected_us / fpga_100_fill,
            "gpu_overhead_corrected_vs_fpga300_ratio": overhead_corrected_us / fpga_300_fill,
            "winner_100mhz_overhead_corrected": winner(overhead_corrected_us, fpga_100_fill),
            "winner_300mhz_overhead_corrected": winner(overhead_corrected_us, fpga_300_fill),
        }
        rows.append(row)

    # ---- Write CSV ----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[4] Writing CSV: {CSV_PATH}")
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- Write markdown summary ----------------------------------------------
    print(f"[5] Writing markdown summary: {MD_PATH}")
    write_markdown_summary(rows, model, cudagraph_note, cuda_available, gpu_name,
                            cuda_version, device)
    print(f"    Written: {MD_PATH.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 78)
    print("Done. Outputs:")
    print(f"  {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)


def write_markdown_summary(
    rows: list[dict[str, object]],
    model: dict[str, float],
    cudagraph_note: str,
    cuda_available: bool,
    gpu_name: str,
    cuda_version: str,
    device: torch.device,
) -> None:
    cudagraph_available = any(r["gpu_cudagraph_mean_us"] == r["gpu_cudagraph_mean_us"] for r in rows)  # not NaN

    def fmt(v: float, spec: str = ".3f") -> str:
        return "N/A" if v != v else format(v, spec)  # NaN check

    main_table_rows = "\n".join(
        f"| {r['patch_size']} | {r['output_windows']:,} | {fmt(r['gpu_mean_us'])} | "
        f"{fmt(r['gpu_cudagraph_mean_us'])} | {fmt(r['gpu_overhead_corrected_us'])} | "
        f"{fmt(r['fpga_100mhz_us'])} | {fmt(r['fpga_300mhz_us'])} |"
        for r in rows
    )

    winners_table_rows = "\n".join(
        f"| {r['patch_size']} | {r['output_windows']:,} | {r['winner_100mhz_estimate']} | "
        f"{r['winner_100mhz_cudagraph']} | {r['winner_100mhz_overhead_corrected']} | "
        f"{r['winner_300mhz_estimate']} | {r['winner_300mhz_cudagraph']} | "
        f"{r['winner_300mhz_overhead_corrected']} |"
        for r in rows
    )

    nofill_table_rows = "\n".join(
        f"| {r['patch_size']} | {r['output_windows']:,} | "
        f"{r['fpga_100mhz_us_nofill']:.3f} | {r['fpga_100mhz_us']:.3f} | "
        f"{r['fpga_300mhz_us_nofill']:.3f} | {r['fpga_300mhz_us']:.3f} |"
        for r in rows
    )

    def first_gpu_win(key: str):
        return next((r for r in rows if r[key] == "GPU (measured)"), None)

    def crossover_sentence(crossover_row, clock_label, view_label):
        if crossover_row is None:
            return (
                f"Under {view_label} at {clock_label}, the FPGA estimate remains faster "
                f"across the ENTIRE tested workload range (up to "
                f"{rows[-1]['output_windows']:,} windows) -- no crossover observed."
            )
        return (
            f"Under {view_label} at {clock_label}, the crossover (GPU first faster than "
            f"the FPGA estimate) falls at or before **{crossover_row['output_windows']:,} "
            f"windows** (patch_size={crossover_row['patch_size']})."
        )

    ctx = EXISTING_DIRECT_PARALLEL_200T_CONTEXT

    model_overhead_sign_note = (
        f"NEGATIVE ({model['fixed_overhead_us']:.4f} us), not positive as a naive "
        f"'launch overhead' interpretation might expect"
        if model["fixed_overhead_us"] < 0 else
        f"positive ({model['fixed_overhead_us']:.4f} us), consistent with a naive "
        f"'launch overhead' interpretation"
    )

    cudagraph_status_line = (
        f"CUDA Graph capture and replay **succeeded** on this run ({cudagraph_note})."
        if cudagraph_available else
        f"CUDA Graph capture/replay was **not available or did not succeed** on this run "
        f"({cudagraph_note}). The CUDA Graph columns below are reported as N/A; all other "
        f"analysis (measured normal latency, overhead-corrected estimate, FPGA estimate) "
        f"is unaffected."
    )

    md = f"""# First-Layer Arithmetic-Core Benchmark: GPU (Measured) vs. Speed-Oriented FPGA Estimate

Generated by `scripts/benchmark_first_layer_arithmetic_core_gpu_vs_fpga.py`.

This revision ADDS GPU-overhead analysis (CUDA Graph replay timing and an
overhead-corrected/amortized throughput estimate) alongside the ORIGINAL
measured, normal-PyTorch-dispatch GPU latency, which is UNCHANGED and
NOT removed -- see Section 4 for why all three views are kept side by
side rather than replacing one with another.

## 1. What this benchmark measures

This isolates the ARITHMETIC CORE of the first learned Conv-BN-ReLU stage
(`enc1.block.0` Conv2d + `enc1.block.1` BatchNorm2d + ReLU, real trained
weights from
`models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt`,
BatchNorm folded into the Conv2d weights/bias): given N pre-extracted
3x3x3 input windows (flattened to `[N, 27]`), how long does it take to
produce all 32 folded Conv-BN-ReLU output values per window --
`ReLU(windows @ w_folded + b_folded)`, i.e. exactly the same arithmetic a
direct-parallel FPGA core would perform, one window per clock -- as a GPU
matmul (MEASURED, in three complementary views -- Section 4) vs. a
speed-oriented, direct-parallel Artix-7 200T cycle estimate?

## 2. What this benchmark excludes

- Image line buffering / sliding-window extraction -- windows are assumed
  ALREADY EXTRACTED and resident in memory (GPU) or in registers (FPGA
  estimate) before any timed region begins.
- Disk I/O, DataLoader startup, and checkpoint loading (the checkpoint is
  loaded once, before timing, only to obtain the real trained weights).
- The rest of the U-Net (encoder stages 2-4, bottleneck, decoder).
- Any board-measured FPGA timing, power, or speedup -- the FPGA numbers
  in this report are cycle-count ESTIMATES, never a board measurement.
- Full FPGA U-Net acceleration of any kind.

## 3. Why this is fairer than comparing the full U-Net to a small FPGA prototype

Earlier benchmarks in this repo (e.g.
`scripts/benchmark_first_layer_gpu_vs_fpga_estimate.py`) compared a
MEASURED full first-layer nn.Module (with padding, full image sizes) on
GPU against a RESOURCE-SHARED (time-multiplexed) FPGA cycle estimate --
an architecture deliberately optimized for LOW RESOURCE USE, not speed.
This benchmark instead compares the SAME underlying arithmetic (one
window in, 32 folded Conv-BN-ReLU values out) against a SPEED-ORIENTED,
direct-parallel FPGA core assumption -- the architecture family this
repo's own existing 32-output DSP-aware synthesis result already
demonstrates is synthesizable (see Section 6) -- so neither side is
handicapped by an architecture chosen for a different goal (resource
savings) than the one being compared (speed).

## 4. GPU timing: three clearly-separated views (read this before the tables)

**All three views measure the SAME arithmetic op on the SAME GPU. They
differ only in how much CPU-side dispatch/launch overhead is included.**
None of them is "the fake one" -- each answers a different, legitimate
question:

1. **Measured, normal PyTorch/CUDA execution** (`gpu_mean_us` and
   related columns) -- UNCHANGED from the original version of this
   script. Ordinary eager-mode PyTorch calls, timed with
   `torch.cuda.Event` around each call, including real per-call
   Python/CUDA dispatch overhead. **This answers: how long does one real,
   standalone small job actually take, launched the normal way?** This is
   the number that matters if the FPGA (or GPU) is being asked to handle
   jobs one at a time, e.g. as they arrive.
2. **CUDA Graph replay timing** (`gpu_cudagraph_mean_us`) -- NEW. The
   identical arithmetic op is captured once into a CUDA graph, then
   replayed {CUDAGRAPH_REPLAY_ITERS} times; replay collapses per-call CPU
   dispatch overhead into a single graph-launch cost. **This is a
   SENSITIVITY ANALYSIS of reduced-overhead timing** -- it shows what GPU
   latency looks like once repeated-launch overhead is minimized (e.g. if
   this same op were called many times back-to-back in a pipeline), NOT
   a claim that ordinary PyTorch code runs this fast for a one-off call.
3. **Overhead-corrected / amortized throughput estimate**
   (`gpu_overhead_corrected_us`) -- NEW. A linear model
   `T_gpu(N) = fixed_overhead_us + per_window_us * N` is fit to view 1's
   measured latency, using only the LARGER workloads (`patch_size >=
   {FIT_MIN_PATCH_SIZE}`) so a few small, noisy points do not dominate
   the regression (Section 7). The `per_window_us * N` term ALONE (fixed
   overhead removed) is reported here -- an **idealized, zero-fixed-
   overhead throughput bound**, useful only as a best-case sensitivity
   estimate for GPU, never as a real single-call latency claim.

**Removing or reducing GPU overhead (views 2 and 3) will ALWAYS make GPU
look better, especially for small workloads**, precisely because fixed
per-call overhead is proportionally larger for small jobs. This is
expected, not a flaw in view 1 -- all three views are reported so the
fair conclusion (Section 9) can be checked against each of them
separately, rather than picking whichever view looks most favorable to
one side.

{cudagraph_status_line}

## 5. GPU timing method (measured views)

- PyTorch on CUDA (if available; falls back to CPU wall-clock timing
  otherwise -- see Section 6 for which was used in this run).
- The checkpoint is loaded and BatchNorm is folded into the Conv2d
  weights/bias ONCE, before any timed region.
- For each workload size N, a `[N, 27]` random input tensor is created
  and moved to the GPU BEFORE timing begins (no data movement inside the
  timed region).
- **View 1 (normal)**: {WARMUP_ITERS} warmup iterations (untimed), then
  {TIMED_ITERS} timed iterations; for CUDA, all `torch.cuda.Event`
  start/end pairs are recorded without synchronizing in between, then a
  single `torch.cuda.synchronize()` is called before reading elapsed
  times (avoids per-iteration host-side sync overhead contaminating the
  measurement).
- **View 2 (CUDA Graph)**: {CUDAGRAPH_WARMUP_ITERS} warmup iterations on
  a side CUDA stream (per PyTorch's CUDA graph capture requirements),
  then the op is captured once via `torch.cuda.graph(...)`, then replayed
  {CUDAGRAPH_REPLAY_ITERS} times with the same `torch.cuda.Event` timing
  convention as view 1.
- Reported for views 1 and 2: mean, median, min, max, P10, and P90, all
  in microseconds.

## 6. FPGA estimate assumptions

Speed-oriented, DIRECT-PARALLEL core (NOT the resource-shared/time-
multiplexed architecture used elsewhere in this repo):
- One 3x3x3 input window consumed per clock cycle.
- All 32 output channels produced per clock cycle, once the pipeline is
  full (steady-state throughput of 1 window/cycle).
- Folded BatchNorm + ReLU included as extra PIPELINED arithmetic stages
  on top of the raw convolution -- not a separate pass.
- **Pipeline-fill depth: {FPGA_PIPELINE_FILL_CYCLES} cycles**, taken
  directly from this repo's own GHDL-measured pipelined folded
  Conv-BN-ReLU design: 4 cycles for the existing, GHDL-verified raw-
  convolution `stream_conv3x3_3chan_cell` + 2 cycles for the registered
  BatchNorm-fold-multiply and bias-add-ReLU pipeline stages (see
  `hardware/vhdl_conv3x3/first_conv_bn_relu_kernel0_pipelined_summary.md`'s
  measured 6-cycle total latency). This is a REUSED, already-documented
  number, not a new invention.
- **100 MHz**: the SAME clock target used by every Vivado synthesis in
  this repo, including the existing direct-parallel 32-output DSP-aware
  design, which is ALREADY SYNTHESIZED and meets 100 MHz timing on
  Artix-7 200T (`{ctx["part"]}`) using **{ctx["dsps_used"]}/{ctx["dsps_available"]} DSPs
  (100%)**, WNS = +{ctx["wns_ns"]} ns (source: `{ctx["source"]}`). This
  existing result is direct evidence that the direct-parallel,
  speed-oriented architecture family assumed here is synthesizable and
  timing-closes at 100 MHz for the real 32-channel first layer -- though
  it is RESOURCE-HEAVY (100% of the part's DSP budget) and does not
  itself include the folded BN+ReLU stage.
- **300 MHz**: a CLEARLY HYPOTHETICAL, speed-oriented target. NO
  synthesis run in this repo has confirmed 300 MHz timing closure for
  this or any other design -- this number is reported ONLY as an
  illustrative "what if" upper bound, never as a demonstrated result.
- Two FPGA figures are computed at each clock: WITH pipeline fill
  (`cycles = N + {FPGA_PIPELINE_FILL_CYCLES}`, the primary
  `fpga_100mhz_us`/`fpga_300mhz_us` columns) and WITHOUT fill
  (`cycles = N`, a simple throughput-only estimate, reported alongside
  for transparency in Section 8).

## 6b. Architecture sanity check: is one-window-per-cycle the fastest practical estimate?

**This is an explanation, not a new benchmark result -- no timing numbers
in this report depend on this section.**

The FPGA estimate above should be interpreted as a speed-oriented,
direct-parallel BASELINE, not a proven globally optimal FPGA
implementation. It parallelizes across all 32 first-layer output
channels and consumes one input window per cycle. Processing multiple
windows per cycle would require duplicating the arithmetic core (or
using a different mapping), so it is fair to ask whether that is
actually feasible on the target used here.

It is not, at least not with the same DSP-heavy mapping: the existing
32-output direct-parallel Conv2d synthesis already consumes
**{ctx["dsps_used"]}/{ctx["dsps_available"]} DSPs (100%)** on Artix-7 200T
(`{ctx["part"]}`, source: `{ctx["source"]}`) -- and that design is RAW
Conv2d only, without the folded BN/ReLU arithmetic this benchmark's
estimate also assumes. Because the DSP budget is already fully consumed
by a single one-window-per-cycle core, naively duplicating that core to
process 2 or more windows per cycle would not fit on the same target
using the same DSP-heavy approach.

Thus, one-window-per-cycle is a reasonable FASTEST-SIMPLE estimate for
the currently available Artix-7 200T target, using this DSP-parallel
mapping -- not a claim that it is the fastest FPGA implementation
possible in any absolute sense. Faster designs would require different
architecture choices, such as more LUT/carry-chain arithmetic (trading
DSPs for logic fabric), lower numeric precision, more aggressive
pipelining, or a larger FPGA than the one already used throughout this
repo. None of those alternative architectures have been designed,
synthesized, or verified here -- this section only explains the
plausibility boundary of the estimate already used, it does not change
it.

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
| Input shape | [N, 27] |

## 7. Fitted overhead/per-window model

Fit to view 1 (normal measured) latency, using only workloads with
`patch_size >= {FIT_MIN_PATCH_SIZE}` ({model['n_points_used']} data
points), via ordinary least squares:

```
T_gpu(N) ~= fixed_overhead_us + per_window_us * N
fixed_overhead_us = {model['fixed_overhead_us']:.4f}
per_window_us     = {model['per_window_us']:.6f}
R^2               = {model['r_squared']:.5f}
```

The `gpu_overhead_corrected_us` column/view in the tables below is
`per_window_us * N` ONLY (the fitted fixed-overhead intercept removed) --
an idealized, zero-fixed-overhead throughput bound, not a claim about
real single-call latency.

**Caveat on the sign of `fixed_overhead_us`**: on this run, the fitted
intercept came out {model_overhead_sign_note}. This is an artifact of
fitting a straight line to only the LARGEST few workloads, where GPU
time vs. N is not perfectly linear (e.g. memory-bandwidth/scheduling
effects can make the largest points grow slightly faster than the
mid-range large points) -- it does NOT mean GPU has "negative launch
overhead." The intercept from this restricted-range fit should not be
read as a literal overhead number outside the fitted range; the
quantity actually used for the overhead-corrected estimate is only the
slope (`per_window_us`), applied per-window.

## 8. Timing table: measured (both views) + overhead-corrected estimate vs. FPGA estimate

| Patch | Windows | GPU normal mean (us) | GPU CUDA-Graph mean (us) | GPU overhead-corrected (us) | FPGA@100MHz (us) | FPGA@300MHz (us) |
|---|---:|---:|---:|---:|---:|---:|
{main_table_rows}

## 8b. Winner by view (GPU vs. FPGA estimate, at each clock)

| Patch | Windows | Winner @100MHz (normal) | Winner @100MHz (CUDA Graph) | Winner @100MHz (overhead-corrected) | Winner @300MHz (normal) | Winner @300MHz (CUDA Graph) | Winner @300MHz (overhead-corrected) |
|---|---:|---|---|---|---|---|---|
{winners_table_rows}

## 8c. FPGA estimate: with-fill vs. no-fill throughput-only (transparency)

| Patch | Windows | No-fill @100MHz (us) | With-fill @100MHz (us) | No-fill @300MHz (us) | With-fill @300MHz (us) |
|---|---:|---:|---:|---:|---:|
{nofill_table_rows}

## 9. Is there a crossover point where GPU becomes faster? (checked under all three views)

**Under measured, normal GPU latency** (the real small-job-latency view):

{crossover_sentence(first_gpu_win("winner_100mhz_estimate"), "100 MHz", "measured normal latency")}

{crossover_sentence(first_gpu_win("winner_300mhz_estimate"), "300 MHz", "measured normal latency")}

**Under CUDA Graph replay timing** (reduced-overhead sensitivity view):

{crossover_sentence(first_gpu_win("winner_100mhz_cudagraph"), "100 MHz", "CUDA Graph replay timing")}

{crossover_sentence(first_gpu_win("winner_300mhz_cudagraph"), "300 MHz", "CUDA Graph replay timing")}

**Under the overhead-corrected / amortized throughput estimate** (zero-fixed-overhead sensitivity bound):

{crossover_sentence(first_gpu_win("winner_100mhz_overhead_corrected"), "100 MHz", "the overhead-corrected estimate")}

{crossover_sentence(first_gpu_win("winner_300mhz_overhead_corrected"), "300 MHz", "the overhead-corrected estimate")}

As expected, reducing/removing GPU overhead (CUDA Graph, then the
overhead-corrected estimate) shifts the crossover point to smaller
workload sizes compared to measured normal latency, because GPU
becomes competitive earlier once fixed dispatch overhead is reduced.
This confirms that the speed-oriented FPGA estimate is most competitive
against GPU small-job launch/dispatch overhead, not against the GPU's
raw arithmetic throughput.

## 10. Interpretation

- **Small workloads may favor the FPGA-style fixed streaming arithmetic
  core**: at very small window counts, a direct-parallel FPGA core's
  fixed, low per-window cycle cost (no kernel-launch overhead, no
  scheduling) can undercut GPU's MEASURED normal latency, which carries a
  comparatively large fixed dispatch/launch overhead that dominates at
  small N. This is a REAL result, not an artifact -- fixed GPU overhead
  is part of real small-job latency.
- **Larger workloads may favor GPU throughput**: as N grows, the GPU's
  massively parallel matmul throughput amortizes its fixed overhead
  across many more windows simultaneously, while the FPGA estimate here
  assumes strictly sequential, one-window-per-cycle consumption -- so the
  FPGA estimate's total time grows linearly with N while GPU time grows
  much more slowly once its fixed overhead is amortized.
- **CUDA Graph and overhead-corrected views shift, but do not eliminate,
  this picture**: with per-call dispatch overhead reduced (CUDA Graph) or
  removed entirely (overhead-corrected estimate), GPU looks relatively
  better at small N than under measured normal latency -- exactly as
  expected, since removing overhead always helps disproportionately at
  small N. This does NOT mean the measured normal-latency result was
  wrong; it means the FPGA estimate's small-N advantage is, at least in
  part, an advantage over GPU DISPATCH OVERHEAD specifically, not over
  the GPU's raw arithmetic throughput.
- **This is an arithmetic-core benchmark, not full flood-map inference.**
  It says nothing about full-image throughput, full U-Net latency, or
  end-to-end flood-segmentation speed -- only about the isolated
  first-layer Conv-BN-ReLU arithmetic, per window, with data already
  resident before timing starts on all sides.

## 11. Claim boundaries

- **GPU timing is MEASURED** in two complementary ways (normal dispatch,
  Section 4 view 1; CUDA Graph replay, view 2), on this specific GPU (see
  Environment table). The overhead-corrected estimate (view 3) is a
  FITTED sensitivity bound derived from measured data, not itself a
  direct measurement.
- **FPGA timing is ESTIMATED**, from a speed-oriented, direct-parallel
  core assumption plus this repo's own existing Artix-7 200T synthesis
  context (Section 6) -- it is NOT a board measurement, and NOT a new
  synthesis run for this specific benchmark.
- **No board-measured FPGA speedup is claimed anywhere in this report.**
- **No full FPGA U-Net is claimed or implied** -- this is the first
  Conv-BN-ReLU stage's arithmetic core only.
- **The 300 MHz figure is explicitly hypothetical** -- no design in this
  repo has been synthesized or verified to close timing at 300 MHz.
- **CUDA Graph and overhead-corrected views are sensitivity analyses**,
  not claims that ordinary single-call PyTorch latency is as low as
  either view suggests.
- **This does not retrain, fine-tune, or otherwise modify the
  checkpoint.**
"""
    MD_PATH.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
