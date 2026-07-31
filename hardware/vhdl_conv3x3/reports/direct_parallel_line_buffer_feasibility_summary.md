# Direct-Parallel First-Layer Streaming Front-End Feasibility

Date: July 31, 2026

This report answers the one question every prior direct-parallel
arithmetic-core benchmark and tile-throughput comparison explicitly
excluded: **what front end would be needed to stream real pixels from a
tile and hand the arithmetic core one flattened 27-value 3x3x3 window per
cycle -- and does it actually work?**

## Claim boundary (read this first)

- **Simulation-only.** No new Vivado synthesis was run for this task.
- **Not full-tile board behavior.** One real 6x6 UAVSAR-tile-derived
  sub-block (16 valid windows), not a full 128x128/256x256 tile.
- **Not full U-Net inference.** First Conv2d+BatchNorm+ReLU stage only.
- **No board testing, no board-measured speedup, no board-measured
  power** anywhere in this report.
- **No existing VHDL datapath file was modified.** The front end and
  integrated skeleton are new files built entirely from existing,
  unmodified sub-components (see Section 2).

## 1. Existing window/front-end module inspected

`hardware/vhdl_conv3x3/window3x3_stream.vhd` (existing, already used
throughout this repo) was inspected directly:

- **Input format**: ONE row-major pixel stream (`pixel_in`,
  `signed(7 downto 0)`), single channel only -- it does NOT support 3
  channels natively.
- **Output**: registered 3x3 window, 9 pixels (`p0`..`p8`, row-major).
- **`valid_out` behavior**: asserted once at least 2 full rows have been
  streamed AND the 3rd row has reached column index 2 (0-indexed) --
  i.e., after `2*IMG_WIDTH + 3` pixels (confirmed against
  `tb_window3x3_stream.vhd`'s own documented convention: a 5x5 image's
  first valid window is triggered by pixel 13 = 2*5+3).
- **Throughput**: 1 window/cycle once primed (no back-pressure support).
- **Latency**: registered output, effectively 1 cycle from a pixel being
  presented to its window being visible externally.
- **Channel support**: single channel. Every existing 3-channel design in
  this repo (`stream_conv3x3_3chan_cell.vhd`,
  `conv3x3_3chan_32out_bn_relu_time_mux.vhd`, etc.) already handles this
  by instantiating THREE `window3x3_stream` copies, one per channel,
  sharing `valid_in`/`clk`/`rst` -- this is the established repo
  convention this task's new front end reuses, not a new pattern.

**Conclusion**: the existing module does NOT by itself answer the
motivating question (it is single-channel), so a minimal 3-channel
wrapper was needed (Option B), built entirely from unmodified instances
of it.

## 2. What was built (and what was reused, unmodified)

| File | Status | Purpose |
|---|---|---|
| `window3x3_stream.vhd` | Existing, **unmodified** | Single-channel sliding-window primitive |
| `first_layer_32out_folded_arithmetic_core.vhd` | Existing, **unmodified** | Direct-parallel folded 32-output core (commit 313558e4) |
| `first_layer_32out_folded_bn_relu_real_tile_q20_pkg.vhd` | Existing, **unmodified** | Core's weight/scale/bias source |
| `real_tile_stimulus_pkg.vhd`, `first_conv_bn_relu_kernel{0..31}_real_tile_q20_pkg.vhd` | Existing, **unmodified** | Real 6x6 UAVSAR stimulus + golden vectors (reused a 3rd time) |
| `window3x3_stream_3chan_flattened.vhd` | **NEW** | 3-channel front-end wrapper: 3x unmodified `window3x3_stream`, port names matched exactly to the arithmetic core's 27 input ports |
| `first_layer_32out_folded_arithmetic_core_streaming_front_end.vhd` | **NEW** | Tiny integrated skeleton: pure port-map wiring of the new front end directly into the existing, unmodified arithmetic core |
| `tb_window3x3_stream_3chan_flattened.vhd` | **NEW** | End-to-end self-checking testbench |
| `run_ghdl_window3x3_stream_3chan_flattened.sh` | **NEW** | GHDL run script |

No packing/flattening logic was needed at the RTL level: VHDL does not
require a single bus for a "flattened 27-value window" -- 27 individually
named `signed(7 downto 0)` signals ARE the flattened window, and the
arithmetic core already expects exactly that port shape. The new wrapper
only maps the 3 existing window generators' outputs to those exact names.

## 3. Commands run

```bash
bash hardware/vhdl_conv3x3/run_ghdl_window3x3_stream_3chan_flattened.sh
git status -sb
```

## 4. GHDL result

**PASS: 512/512 outputs correct** (16 real windows x 32 kernels), streamed
through the NEW front end into the EXISTING, unmodified arithmetic core --
checked against the SAME real-tile Q.20 golden values already reused
twice before (direct-parallel single-shot testbench, resource-shared
real-tile testbench). Window 1's outputs (`y0=0 y1=62052 y31=0`) match
both prior verifications bit-for-bit, confirming the front end introduces
no data-path errors.

## 5. Measured latency / throughput

GHDL reported directly (not derived by hand):

```
MEASURED: first valid_out at cycle 21 (fill latency);
          last valid_out at cycle 42;
          total pixels streamed = 36;
          total cycles (stream + drain) = 42
```

- **Fill latency: 21 cycles** for a 6x6 image before the first useful
  32-output result appears (front end's own ~15-16-cycle window-buffer
  fill, plus the arithmetic core's 6-cycle pipeline).
- **Drain: exactly 6 cycles** after the last pixel is streamed (36th
  pixel) before the last window's outputs appear -- this exactly matches
  the arithmetic core's own documented 6-cycle latency, confirming the
  front end itself adds no EXTRA drain beyond what the core already
  requires.
- **Confirmed formula: `total_cycles = (H * W) + 6`** for a full `H x W`
  image (36 pixels + 6 = 42, exactly matching the measurement). This
  replaces the earlier "windows + 6" formula used when windows were
  assumed pre-extracted -- see
  `outputs/hardware_benchmarks/direct_parallel_line_buffer_feasibility/`
  for the corrected tile-scale timing table.
- **Throughput once primed**: 1 valid window/cycle, unchanged from the
  arithmetic core's own documented behavior -- the front end does not
  reduce steady-state throughput, only adds a one-time fill cost.

## 6. Does this support or change the one-window-per-cycle assumption?

**It supports the assumption, with a small, now-quantified correction.**
The direct-parallel core's own 1-window/cycle, 6-cycle-latency behavior
is UNCHANGED and confirmed again here. What changes is the TOTAL time to
process a full tile: it is no longer "windows + 6" (which implicitly
assumed the windows arrive from nowhere, already extracted) but
"H*W + 6" (paying for the actual pixel-streaming cost). At tiny scale
(6x6) this roughly DOUBLES the estimated time (22 -> 42 cycles, both
still sub-microsecond); at real tile scale (128x128, 256x256) the
correction is small (+3.2%, +1.6%) because the fixed fill/drain cost is
amortized over tens of thousands of windows.

## 7. Should tile-throughput estimates be updated?

**Yes, but the correction is minor at application scale.**
`outputs/hardware_benchmarks/first_layer_architecture_tile_throughput/`'s
direct-parallel numbers (which used `windows + 6`) should be read as a
slight underestimate -- by 1.6% at 256x256 and 3.2% at 128x128 -- rather
than an exact figure. The corrected numbers are provided in
`outputs/hardware_benchmarks/direct_parallel_line_buffer_feasibility/
direct_parallel_line_buffer_cycle_estimates.csv` for future use; that
report's own tables were not edited in place (per this task's
instructions), so both the original and corrected figures remain
available side by side, with the correction and its size explained here.

## 8. What this still does not prove

- **Not synthesized.** The integrated front-end + arithmetic-core
  skeleton has NOT been run through Vivado -- only GHDL-simulated. No
  claim is made about its LUT/register/DSP/timing/power footprint when
  synthesized, though the front end's own resource cost is expected to be
  small (3 unmodified `window3x3_stream` instances, each already
  characterized as lightweight in this repo's other reports).
- **Not full-tile board behavior.** Only a 6x6 real sub-block was
  streamed end-to-end; 128x128/256x256 timing above is a formula-based
  extrapolation, not a new simulation at those sizes.
- **Line buffers here are flip-flop-based** (via `window3x3_stream`'s
  existing shift-register rows), matching that module's own documented
  limitation that BRAM-backed line buffers would be needed for large
  `IMG_WIDTH` in a real synthesized design -- this was not evaluated here.
- **No claim of full U-Net FPGA acceleration, board-measured speedup, or
  board-measured power** anywhere in this report.
