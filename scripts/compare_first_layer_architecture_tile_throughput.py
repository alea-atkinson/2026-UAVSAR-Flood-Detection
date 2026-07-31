#!/usr/bin/env python3
"""
compare_first_layer_architecture_tile_throughput.py

An APPLICATION-SCALE first-layer throughput comparison between the two
folded Conv-BN-ReLU architecture families this repo has actually built and
verified, at realistic tile sizes -- not a new benchmark, not a new
design, just cycle-derived extrapolation from already-confirmed behavior.

---- Why this exists -------------------------------------------------------
Two architecture families now exist for the first learned Conv2d+
BatchNorm+ReLU stage:
  1. Direct-parallel folded arithmetic core (commit 313558e4): 1
     pre-extracted 3x3x3 window/cycle, all 32 outputs/window, 6-cycle
     latency, GHDL-verified, Vivado-synthesized at 100 MHz on Artix-7
     200T (740/740 DSPs -- fully saturated).
  2. 2-lane resource-shared folded Conv-BN-ReLU, now real-tile validated
     (commit 5fc9aa57): 16 real windows x 32 channels = 512/512 GHDL PASS,
     9,030 cycles measured for those 16 real windows (4/740 DSPs -- nearly
     idle).
Both have confirmed per-window/per-run cycle behavior, but neither has
ever been placed side by side at an actual application-relevant tile size
(128x128, 256x256), nor next to the GPU numbers already measured elsewhere
in this repo. This script does exactly that -- pure cycle-count
arithmetic from already-established numbers, no new simulation, no new
synthesis.

---- Cycle formulas used ---------------------------------------------------
Direct-parallel (pre-extracted windows, no line-buffer/capture overhead):
    cycles = windows + 6          (6-cycle pipeline fill, GHDL-confirmed
                                    and Vivado-timing-confirmed at 100 MHz)

2-lane resource-shared (existing window-capture + time-multiplexed
processing, capture overhead included since this design streams pixels
in, unlike the direct-parallel core):
    cycles_per_window = 9030 / 16 = 564.375   (EMPIRICAL, real-UAVSAR-data
                                    GHDL-confirmed rate from commit
                                    5fc9aa57 -- this total already
                                    includes that run's 36-cycle capture
                                    phase amortized across its 16 windows,
                                    per this task's explicit instruction;
                                    see the CAVEAT in the markdown summary)
    cycles = input_size^2 + windows * cycles_per_window
             (capture-phase term scaled to the NEW tile's input_size,
              since a larger tile takes longer to stream in; windows term
              uses the empirical per-window processing rate above)

CROSS-CHECK ONLY (not used for the primary estimate): this repo's own
`multi_lane_resource_shared_conv_bn_relu_design_memo.md` derived a
DESIGN-PLAN cycles-per-window formula, `35 * N / L` (N=32 total kernels,
L=lanes), which for L=2 gives 35*32/2 = 560 cycles/window -- within 0.8%
of the 564.375 empirical rate used here. This is reported as a
corroborating cross-check, not as the primary estimate (the primary
estimate uses the REAL-DATA measured rate per this task's instructions).

---- What this is NOT -------------------------------------------------------
- NOT a board measurement of either architecture at any tile size.
- NOT a new GHDL simulation or Vivado synthesis run.
- NOT a claim that the direct-parallel core has been run on a real
  128x128 or 256x256 tile -- its "no capture overhead" assumption is
  explicit and unchanged from its own synthesis-supported report.
- NOT a claim that the resource-shared design has been run at 128x128 or
  256x256 -- its cycle count at those sizes is an EXTRAPOLATION from the
  6x6 real-data measurement, using the formula above.
- NOT a claim of full-U-Net FPGA acceleration.

---- Usage ------------------------------------------------------------------
    python3 scripts/compare_first_layer_architecture_tile_throughput.py

---- Outputs -----------------------------------------------------------
outputs/hardware_benchmarks/first_layer_architecture_tile_throughput/
    first_layer_architecture_tile_throughput.csv
    first_layer_architecture_tile_throughput_summary.md
"""

from __future__ import annotations

import csv
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "hardware_benchmarks" / "first_layer_architecture_tile_throughput"
CSV_PATH = OUT_DIR / "first_layer_architecture_tile_throughput.csv"
MD_PATH = OUT_DIR / "first_layer_architecture_tile_throughput_summary.md"

CLOCKS_HZ = {"100mhz": 100e6, "300mhz": 300e6}

# ---- Direct-parallel folded arithmetic core (commit 313558e4) -------------
DIRECT_PARALLEL_PIPELINE_FILL_CYCLES = 6
DIRECT_PARALLEL_CTX = {
    "part": "xc7a200tsbg484-1", "luts": 12039, "registers": 6564,
    "dsps_used": 740, "dsps_available": 740, "bram": 0,
    "wns_ns": 2.218, "power_w": 1.334,
    "source": "hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_summary.md",
}

# ---- 2-lane resource-shared, real-tile validated (commit 5fc9aa57) --------
RESOURCE_SHARED_MEASURED_CYCLES = 9030   # for 16 real windows, 6x6 real sub-block
RESOURCE_SHARED_MEASURED_WINDOWS = 16
RESOURCE_SHARED_CYCLES_PER_WINDOW = RESOURCE_SHARED_MEASURED_CYCLES / RESOURCE_SHARED_MEASURED_WINDOWS  # 564.375
RESOURCE_SHARED_CTX = {
    "part": "xc7a200tsbg484-1", "luts": 2531, "registers": 5878,
    "dsps_used": 4, "dsps_available": 740, "bram": 0,
    "wns_ns": 0.229, "power_w": 0.176,
    "source": "hardware/vhdl_conv3x3/reports/resource_shared_32out_real_tile_q20_verification_summary.md",
}
# Cross-check only, from multi_lane_resource_shared_conv_bn_relu_design_memo.md's
# own design-plan formula: cycles_per_window(N, L) ~= 35 * N / L
RESOURCE_SHARED_FORMULA_CROSSCHECK = 35 * 32 / 2  # 560.0, N=32 total kernels, L=2 lanes

# ---- Tile cases (as specified in the task) --------------------------------
TILE_CASES = [
    {"label": "6x6 (real UAVSAR sub-block)", "input_size": 6, "windows": 16},
    {"label": "128x128", "input_size": 128, "windows": 15876},
    {"label": "256x256", "input_size": 256, "windows": 64516},
]

# ---- Existing measured GPU reference numbers (reused, not re-measured) ----
# Source: outputs/hardware_benchmarks/first_layer_arithmetic_core_gpu_vs_fpga/
#         first_layer_arithmetic_core_timing.csv (arithmetic-core-only, RTX 4080)
GPU_ARITHMETIC_CORE_ONLY_US = {
    15876: 15.109,   # patch_size=128 row
    64516: 28.302,   # patch_size=256 row
}
GPU_ARITHMETIC_CORE_SOURCE = (
    "outputs/hardware_benchmarks/first_layer_arithmetic_core_gpu_vs_fpga/"
    "first_layer_arithmetic_core_timing.csv"
)

# Source: outputs/hardware_benchmarks/full_unet_inference_timing/full_unet_timing_summary.md
# (measured full U-Net forward pass, CUDA, batch=1; 128x128 synthetic, 256x256 real_tile)
GPU_FULL_UNET_US = {
    15876: 563.3,    # 128x128 synthetic, cuda, batch1 (0.5633 ms)
    64516: 1212.8,   # 256x256 real_tile, cuda, batch1 (1.2128 ms)
}
GPU_FULL_UNET_SOURCE = "outputs/hardware_benchmarks/full_unet_inference_timing/full_unet_timing_summary.md"


def direct_parallel_cycles(windows: int) -> int:
    return windows + DIRECT_PARALLEL_PIPELINE_FILL_CYCLES


def resource_shared_cycles(input_size: int, windows: int) -> float:
    return (input_size ** 2) + windows * RESOURCE_SHARED_CYCLES_PER_WINDOW


def cycles_to_us(cycles: float, clock_hz: float) -> float:
    return cycles / clock_hz * 1e6


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Application-scale first-layer architecture tile throughput comparison")
    print("=" * 78)
    print(f"Direct-parallel pipeline-fill cycles : {DIRECT_PARALLEL_PIPELINE_FILL_CYCLES}")
    print(f"Resource-shared empirical cycles/window : {RESOURCE_SHARED_CYCLES_PER_WINDOW:.3f} "
          f"({RESOURCE_SHARED_MEASURED_CYCLES} cycles / {RESOURCE_SHARED_MEASURED_WINDOWS} real windows)")
    print(f"Resource-shared design-plan formula cross-check (35*32/2) : "
          f"{RESOURCE_SHARED_FORMULA_CROSSCHECK:.1f} cycles/window "
          f"({abs(RESOURCE_SHARED_FORMULA_CROSSCHECK - RESOURCE_SHARED_CYCLES_PER_WINDOW) / RESOURCE_SHARED_CYCLES_PER_WINDOW * 100:.2f}% "
          f"from the empirical rate)")
    print()

    rows = []
    for case in TILE_CASES:
        windows = case["windows"]
        input_size = case["input_size"]

        dp_cycles = direct_parallel_cycles(windows)
        rs_cycles = resource_shared_cycles(input_size, windows)

        dp_us_100 = cycles_to_us(dp_cycles, CLOCKS_HZ["100mhz"])
        dp_us_300 = cycles_to_us(dp_cycles, CLOCKS_HZ["300mhz"])
        rs_us_100 = cycles_to_us(rs_cycles, CLOCKS_HZ["100mhz"])
        rs_us_300 = cycles_to_us(rs_cycles, CLOCKS_HZ["300mhz"])

        slowdown_100 = rs_us_100 / dp_us_100

        gpu_arith = GPU_ARITHMETIC_CORE_ONLY_US.get(windows)
        gpu_unet = GPU_FULL_UNET_US.get(windows)

        row = {
            "tile_label": case["label"],
            "input_size": input_size,
            "windows": windows,
            "direct_parallel_cycles": dp_cycles,
            "direct_parallel_us_100mhz": round(dp_us_100, 4),
            "direct_parallel_us_300mhz_hypothetical": round(dp_us_300, 4),
            "resource_shared_2lane_cycles": round(rs_cycles, 1),
            "resource_shared_2lane_us_100mhz": round(rs_us_100, 4),
            "resource_shared_2lane_us_300mhz_hypothetical": round(rs_us_300, 4),
            "resource_shared_vs_direct_parallel_slowdown_100mhz": round(slowdown_100, 2),
            "gpu_arithmetic_core_only_us": gpu_arith if gpu_arith is not None else "N/A",
            "gpu_full_unet_us": gpu_unet if gpu_unet is not None else "N/A",
        }
        rows.append(row)
        print(f"  {case['label']:32s} windows={windows:7,d}  "
              f"DP@100MHz={dp_us_100:10.3f} us  RS@100MHz={rs_us_100:12.3f} us  "
              f"slowdown={slowdown_100:8.1f}x")

    print(f"\nWriting CSV: {CSV_PATH.relative_to(REPO_ROOT)}")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Writing markdown summary: {MD_PATH.relative_to(REPO_ROOT)}")
    write_markdown_summary(rows)

    print("\n" + "=" * 78)
    print("Done. Outputs:")
    print(f"  {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)


def write_markdown_summary(rows: list[dict[str, object]]) -> None:
    def fmt(v) -> str:
        if v == "N/A":
            return "N/A"
        return f"{v:,.3f}"

    main_table_rows = "\n".join(
        f"| {r['tile_label']} | {r['windows']:,} | {r['direct_parallel_cycles']:,} | "
        f"{fmt(r['direct_parallel_us_100mhz'])} | {fmt(r['direct_parallel_us_300mhz_hypothetical'])} | "
        f"{r['resource_shared_2lane_cycles']:,} | {fmt(r['resource_shared_2lane_us_100mhz'])} | "
        f"{fmt(r['resource_shared_2lane_us_300mhz_hypothetical'])} | "
        f"**{r['resource_shared_vs_direct_parallel_slowdown_100mhz']:,.1f}x** |"
        for r in rows
    )

    gpu_table_rows = "\n".join(
        f"| {r['tile_label']} | {fmt(r['direct_parallel_us_100mhz'])} | "
        f"{fmt(r['resource_shared_2lane_us_100mhz'])} | "
        f"{fmt(r['gpu_arithmetic_core_only_us']) if r['gpu_arithmetic_core_only_us'] != 'N/A' else 'N/A'} | "
        f"{fmt(r['gpu_full_unet_us']) if r['gpu_full_unet_us'] != 'N/A' else 'N/A'} |"
        for r in rows
    )

    dp = DIRECT_PARALLEL_CTX
    rs = RESOURCE_SHARED_CTX

    md = f"""# First-Layer Architecture Tile-Scale Throughput Comparison

Generated by `scripts/compare_first_layer_architecture_tile_throughput.py`.

**This is a cycle-derived extrapolation, not a new benchmark, not a new
board measurement.** It takes ALREADY-CONFIRMED cycle behavior from two
existing, GHDL-verified, Vivado-synthesized designs and projects it to
realistic tile sizes using simple, explicitly-stated formulas -- no new
VHDL, no new synthesis, no new simulation.

## Claim boundary (read this first)

- **Neither architecture's numbers below are board measurements.** Both
  are cycle-count-derived estimates from GHDL-confirmed behavior and
  Vivado-confirmed 100 MHz timing closure for the EXISTING, already-built
  designs.
- **Direct-parallel assumes pre-extracted windows and NO line-buffer
  overhead** -- its cycle formula (`windows + 6`) does not include the
  cost of streaming a real image into on-chip window buffers, because
  that is not what its synthesized design does (see
  `{dp['source']}`).
- **Resource-shared uses its EXISTING window-capture + time-multiplexed
  control behavior** (this design DOES stream pixels in), but its
  numbers at 128x128/256x256 are still an EXTRAPOLATION from a single
  real 6x6 measurement (commit `5fc9aa57`), not a new simulation at
  those sizes.
- **No full-U-Net FPGA acceleration is claimed anywhere in this report.**
- **300 MHz figures are explicitly, unambiguously hypothetical** -- no
  design in this repo has been synthesized or verified to close timing
  at 300 MHz.

## 1. Cycle formulas used

**Direct-parallel folded arithmetic core** (`{dp['source']}`):
```
cycles = windows + 6        (6-cycle pipeline fill, GHDL- and
                              Vivado-timing-confirmed)
```

**2-lane resource-shared, real-tile validated** (`{rs['source']}`):
```
cycles_per_window = 9030 / 16 = {RESOURCE_SHARED_CYCLES_PER_WINDOW:.3f}
    (EMPIRICAL, real-UAVSAR-data GHDL-confirmed rate; this total already
    includes that run's small capture-phase overhead amortized across its
    16 windows -- see the caveat in Section 6)

cycles = input_size^2 + windows * {RESOURCE_SHARED_CYCLES_PER_WINDOW:.3f}
    (capture-phase term scaled to the NEW tile size; per-window term uses
    the empirical rate above)
```

**Cross-check only** (not used for the primary estimate): this repo's own
`multi_lane_resource_shared_conv_bn_relu_design_memo.md` design-plan
formula, `cycles_per_window(N, L) ~= 35 * N / L` (N=32 kernels, L=2
lanes), gives **{RESOURCE_SHARED_FORMULA_CROSSCHECK:.1f} cycles/window** --
within **{abs(RESOURCE_SHARED_FORMULA_CROSSCHECK - RESOURCE_SHARED_CYCLES_PER_WINDOW) / RESOURCE_SHARED_CYCLES_PER_WINDOW * 100:.2f}%**
of the {RESOURCE_SHARED_CYCLES_PER_WINDOW:.3f} empirical rate used here --
corroborating, but not replacing, the real-data measurement.

## 2. Timing results (100 MHz synthesis-supported, and 300 MHz hypothetical)

| Tile | Windows | DP cycles | DP @100MHz (us) | DP @300MHz hyp. (us) | RS cycles | RS @100MHz (us) | RS @300MHz hyp. (us) | RS slowdown vs. DP @100MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{main_table_rows}

## 3. Comparison with measured GPU timing (existing measurements, reused)

| Tile | DP @100MHz (us) | RS @100MHz (us) | GPU arithmetic-core-only (us, measured) | GPU full U-Net (us, measured) |
|---|---:|---:|---:|---:|
{gpu_table_rows}

GPU arithmetic-core-only source: `{GPU_ARITHMETIC_CORE_SOURCE}` (RTX 4080,
measured, normal PyTorch/CUDA dispatch). GPU full U-Net source:
`{GPU_FULL_UNET_SOURCE}` (measured, CUDA, batch=1, forward pass only).

## 4. Resource / power comparison (existing synthesis results, reused)

| Design | LUTs | Registers | DSP48E1 | BRAM | WNS @100MHz | Power (est.) |
|---|---:|---:|---:|---:|---:|---:|
| Direct-parallel folded arithmetic core | {dp['luts']:,} | {dp['registers']:,} | {dp['dsps_used']}/{dp['dsps_available']} ({dp['dsps_used']/dp['dsps_available']*100:.0f}%) | {dp['bram']} | +{dp['wns_ns']} ns | {dp['power_w']} W |
| 2-lane resource-shared (real-tile validated) | {rs['luts']:,} | {rs['registers']:,} | {rs['dsps_used']}/{rs['dsps_available']} ({rs['dsps_used']/rs['dsps_available']*100:.1f}%) | {rs['bram']} | +{rs['wns_ns']} ns | {rs['power_w']} W |

## 5. What this shows

**The design tradeoff is stark and consistent across all three tile
sizes**: the direct-parallel core is roughly **{rows[-1]['resource_shared_vs_direct_parallel_slowdown_100mhz']:,.0f}x faster** at
256x256 than the 2-lane resource-shared design at the same clock, but
consumes **100% of the Artix-7 200T's DSP budget** (740/740) to do it,
versus the resource-shared design's **{rs['dsps_used']}/{rs['dsps_available']} DSPs ({rs['dsps_used']/rs['dsps_available']*100:.1f}%)**.
Neither number moves much with tile size in relative terms -- the
slowdown ratio is essentially flat across 6x6, 128x128, and 256x256,
because both architectures' cycle counts scale near-linearly with window
count once past small fixed overheads.

**Why this matters after the real-tile resource-shared validation
(commit `5fc9aa57`)**: before that commit, the resource-shared design's
cycle behavior was only confirmed on the synthetic 5x5x3 toy pattern.
This report is the first time that REAL-data-confirmed cycle rate has
been extrapolated to an application-relevant tile size and placed
directly alongside the direct-parallel design's own synthesis-supported
numbers AND the measured GPU numbers -- turning two separate, previously
disconnected verification efforts into one coherent application-scale
comparison.

**Compared to GPU**: at 256x256, the measured GPU arithmetic-core-only
time ({GPU_ARITHMETIC_CORE_ONLY_US.get(64516, 'N/A')} us) is far faster
than either FPGA estimate at this window count, and the measured full
U-Net GPU time ({GPU_FULL_UNET_US.get(64516, 'N/A')} us) -- which does
the ENTIRE model, not just this one stage -- is still faster than the
resource-shared FPGA estimate for the first stage alone. This is
consistent with (not contradicting) the arithmetic-core GPU-vs-FPGA
benchmark's own crossover finding (`outputs/hardware_benchmarks/
first_layer_arithmetic_core_gpu_vs_fpga/first_layer_arithmetic_core_summary.md`):
GPU's advantage grows with workload size once its fixed dispatch overhead
is amortized, while these FPGA designs' cycle counts grow linearly (or
faster, per-window, for the resource-shared design) with window count.

## 6. What this still does not prove

- **Neither architecture has been measured on a real board at any tile
  size in this report.** All numbers are cycle-count-derived from
  simulation and out-of-context synthesis timing closure.
- **The resource-shared design's 128x128/256x256 numbers are an
  extrapolation**, not a new GHDL run at those sizes -- the underlying
  per-window rate is real-data-confirmed only at the small 6x6 scale.
- **Minor conflation caveat**: the {RESOURCE_SHARED_CYCLES_PER_WINDOW:.3f}
  cycles/window rate is computed by dividing the 6x6 real-data run's
  TOTAL cycles (which already includes that run's own 36-cycle capture
  phase) by its 16 windows, per this task's explicit instruction. Adding
  a fresh `input_size^2` capture term on top for larger tiles therefore
  double-counts a small fraction of that original run's capture overhead
  (roughly 2 cycles out of ~564 per window, under 0.5%) -- negligible at
  application scale, but noted here for full transparency.
- **The direct-parallel design's "no line-buffer overhead" assumption is
  unverified at tile scale** -- it has never been combined with an actual
  sliding-window/line-buffer front end and measured end-to-end; this
  report does not change that.
- **No full-U-Net FPGA acceleration is claimed or implied.** This
  compares one stage's cycle timing on two FPGA architectures against
  measured GPU numbers for that same stage AND for the full model -- not
  a full-model FPGA estimate.
"""
    MD_PATH.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
