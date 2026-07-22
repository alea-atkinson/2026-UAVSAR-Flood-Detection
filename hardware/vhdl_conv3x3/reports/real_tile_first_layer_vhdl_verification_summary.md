# Real-Tile First-Layer Folded Conv-BN-ReLU VHDL Verification

Date: July 22, 2026

This report documents a **real-data verification SUBSET**: the first
attempt in this repository to verify an existing folded Conv-BN-ReLU
VHDL design against a genuine, per-tile-normalized-and-quantized UAVSAR
patch instead of the synthetic 5x5x3 toy patch every other VHDL
prototype in this repo has used so far.

## Claim boundary (read this first)

- **This verifies kernel 0 and kernel 2 of the complete first Conv2d
  layer's 32 output channels -- NOT all 32 channels.** See "Why not full
  32-output real-tile verification" below.
- **No hardware architecture was invented and no already-verified VHDL
  datapath was modified.** Every new VHDL file is structurally IDENTICAL
  to the existing, already-verified
  `stream_conv3x3_3chan_kernel{0,2}_bn_relu_pipelined` designs (same
  `stream_conv3x3_3chan_cell` sub-component, same two-stage Q.16
  BatchNorm-fold + ReLU pipeline). The only thing that changed is which
  package supplies the SCALE_FX constant (computed from a real per-tile
  activation scale, not the toy patch's exact 1.0) and which pixel
  stimulus is driven into the testbench (a real UAVSAR sub-patch, not a
  synthetic ascending-integer pattern).
- **This is simulation-only (GHDL).** No synthesis, no board testing, no
  measured speedup, no measured power.
- **This does not implement padding.** Valid (no-padding) 3x3 windows
  only, same scope as every other VHDL prototype in this repo.
- **This is not a full U-Net or full Conv2d layer.** It verifies two
  individual output channels of the first Conv2d layer's folded
  Conv-BN-ReLU stage only.

## Purpose

We already had:
1. VHDL toy/small-patch verification for first-layer Conv-BN-ReLU
   prototypes (5x5x3 synthetic input, every design in
   `hardware/vhdl_conv3x3/`).
2. Python real-tile first-layer activation fidelity
   (`scripts/analyze_first_layer_real_activation_fidelity.py`).
3. Python full-U-Net segmentation impact across all 7 held-out flight
   paths (`scripts/analyze_first_layer_fixed_point_segmentation_impact.py`
   and its all-flight-path summary).

What was missing was a bridge between (1) and (2)/(3): **does the actual
VHDL datapath -- not just a Python re-implementation of its arithmetic --
produce bit-exact fixed-point integers when driven by real UAVSAR pixel
data?** This report answers that, for a deliberately small subset.

## Why kernel 0 + kernel 2, not the full 32-output design

The task allowed doing the full 32-output resource-shared design "if
practical," with an explicit fallback to the kernel0/kernel2 subset
otherwise. Before writing any VHDL, the 32-output resource-shared design
(`stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd`,
`first_layer_32out_bn_relu_resource_shared_pkg.vhd`) and its generator
script (`scripts/generate_resource_shared_32out_bn_relu_vectors.py`) were
inspected. Extending that design to real-tile input is a legitimately
larger undertaking than the kernel0/kernel2 subset, for concrete reasons:

- It requires all 32 kernels' SCALE_FX/BIAS_FX constants to be
  regenerated at once from a real per-tile scale_x (32 golden arrays
  instead of 2), multiplying the surface area for a golden-vector bug to
  hide in.
- Its window-scheduling FSM and package layout are more complex than the
  two-stage kernel0/kernel2 pipeline (time-multiplexed across kernels
  *and* windows), so re-verifying it against a non-5x5 image geometry
  carries materially more risk of a subtle indexing bug than reusing the
  already well-understood, already-parameterized (`IMG_WIDTH` generic)
  kernel0/kernel2 pipeline.
- The kernel0/kernel2 designs were already deliberately chosen in this
  repo's own history as the two most informative single-channel
  data points (kernel 0: baseline; kernel 2: the LARGEST folded-weight
  quantization scale and LARGEST float-vs-fixed-point error of all 32
  kernels, per
  `hardware/vhdl_conv3x3/test_vectors/conv_bn_relu_fidelity/first_conv_bn_relu_per_channel_metrics.csv`),
  so verifying real-tile behavior on exactly these two channels is a
  meaningful, non-arbitrary first real-data checkpoint, not merely "the
  easy path."

Given this, the kernel0/kernel2 subset was built first, exactly as the
task's fallback plan anticipated. **Extending to the full 32-output
design remains open** (see "Recommended next step").

## What was built (all new files; nothing existing was modified)

| File | Purpose |
|---|---|
| `scripts/generate_real_tile_bn_relu_fixed_point_vectors.py` | Python generator: loads the real tile, folds BatchNorm, quantizes (weights + real per-tile activation scale), computes exact-integer golden outputs, writes VHDL packages. |
| `hardware/vhdl_conv3x3/real_tile_stimulus_pkg.vhd` | Shared REAL 6x6x3 UAVSAR pixel stimulus (INT8), used by both kernel testbenches. |
| `hardware/vhdl_conv3x3/first_conv_bn_relu_kernel0_real_tile_pkg.vhd` | Kernel 0's folded INT8 weights (identical to the existing toy-patch package) + REAL-TILE SCALE_FX/BIAS_FX + 16-value golden array. |
| `hardware/vhdl_conv3x3/first_conv_bn_relu_kernel2_real_tile_pkg.vhd` | Same, for kernel 2. |
| `hardware/vhdl_conv3x3/stream_conv3x3_3chan_kernel0_bn_relu_real_tile.vhd` | New entity, structurally identical to the existing pipelined design, reading the new real-tile package. |
| `hardware/vhdl_conv3x3/stream_conv3x3_3chan_kernel2_bn_relu_real_tile.vhd` | Same, for kernel 2. |
| `hardware/vhdl_conv3x3/tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu.vhd` | New combined, self-checking testbench: drives both DUTs with the same real pixel stream, checks both against their golden arrays, reports PASS/FAIL count and max absolute integer difference. |
| `hardware/vhdl_conv3x3/run_ghdl_first_layer_real_tile_bn_relu.sh` | GHDL run script, following the exact conventions of every other `run_ghdl_*.sh` script in this repo. |
| `hardware/vhdl_conv3x3/real_tile_vectors/kernel{0,2}/*.{json,csv,md}` | Golden-vector artifacts (JSON/CSV/Markdown), analogous to `test_vectors/` elsewhere in this repo. |

**Nothing under `stream_conv3x3_3chan_cell.vhd`, `window3x3_stream.vhd`,
`conv3x3_dot_pipelined.vhd`, the existing toy-patch
`first_conv_bn_relu_kernel{0,2}_pkg.vhd` packages, or the existing
pipelined entities/testbenches was touched.**

## Real patch used

| Field | Value |
|---|---|
| Tile | `tile_16_42.tif` (first row of `heldout_fp2_test.csv`) |
| Checkpoint | `models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt` |
| Normalization | `FloodTileDataset._normalize_per_tile` (verbatim, `train_unet_baseline_tuned.py`) |
| Activation quantization | `scale_x = max(\|x_full_tile\|) / 127`, computed from the FULL 256x256 normalized tile -- identical convention to `analyze_first_layer_fixed_point_segmentation_impact.py` |
| Sub-block | 6x6 pixels, top-left offset (row=125, col=125), a direct slice of the ALREADY-quantized full-tile INT8 array (not independently re-quantized) |
| Valid 3x3 windows | 16 (4x4) per kernel |
| BatchNorm folding | Identical formula to `analyze_first_layer_fixed_point_segmentation_impact.py`: `scale_bn = gamma/sqrt(var+eps)`, `b_folded = beta - mean*scale_bn` |
| Weight quantization | Symmetric per-output-channel INT8, `scale_w_folded = max(\|w_folded\|)/127` |

## Computed constants (both kernels, this real tile)

| Kernel | scale_bn | b_folded | scale_w_folded | scale_x (full tile) | SCALE_FX (Q.16) | BIAS_FX (Q.16) | ReLU clamp exercised? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.515678 | -0.014275 | 0.00206228 | 0.01808654 | **2** | -936 | Yes (12/16 clamped) |
| 2 | 3.753419 | -0.011989 | 0.00575681 | 0.01808654 | **7** | -786 | Yes (4/16 clamped) |

The folded INT8 weight values match the existing toy-patch packages'
weights exactly (confirmed by inspection -- weights do not depend on the
input, only `SCALE_FX` changes since it bakes in the real `scale_x`).

## GHDL command run

```
bash hardware/vhdl_conv3x3/run_ghdl_first_layer_real_tile_bn_relu.sh
```

(analyzes `window3x3_stream.vhd`, `conv3x3_dot_pipelined.vhd`,
`stream_conv3x3_3chan_cell.vhd` -- all existing, unmodified -- then the
six new files listed above, in dependency order; elaborates and runs
`tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu` with
`--stop-time=1000ns --assert-level=failure`.)

## Result

```
=== REAL-TILE first-layer Conv-BN-ReLU verification summary ===
    Kernel 0: 16 / 16 outputs observed
    Kernel 2: 16 / 16 outputs observed
    PASS count = 32   FAIL count = 0
    Max absolute integer difference (Q.16 fixed-point) = 0
=== All tb_stream_conv3x3_3chan_first_layer_real_tile_bn_relu tests PASSED ===
  (32 / 32 outputs match Python Q.16 fixed-point golden vectors on REAL UAVSAR tile data)
```

**32 / 32 PASS. Max absolute integer difference = 0** -- the existing,
unmodified kernel-0 and kernel-2 folded Conv-BN-ReLU pipelined VHDL
datapath produces bit-exact Q.16 fixed-point integers, matching the
Python golden-vector generator exactly, on this real UAVSAR tile's
patch. ReLU clamping was genuinely exercised by real data (12/16 outputs
clamped for kernel 0, 4/16 for kernel 2), not just theoretically possible.

## An honest finding from doing this with real data

Using a REAL per-tile `scale_x` (0.0181, vs. the toy patch's exact 1.0)
revealed something the toy-patch tests could never show: the existing
Q.16 fixed-point format, which was implicitly calibrated around
`scale_x = 1.0`, gives a very COARSE `SCALE_FX` for real, normalized SAR
data -- **`SCALE_FX = 2` for kernel 0 and `SCALE_FX = 7` for kernel 2**,
i.e. only 1-3 bits of usable precision in the scale constant itself
(`combined_scale = scale_x * scale_w_folded` was `0.0000373` and
`0.0001041` respectively, both far too small for Q.16 to represent with
more than a couple of significant bits). This does **not** cause a
VHDL-vs-Python mismatch (both sides round `SCALE_FX` identically, so the
comparison in this report is still bit-exact) -- but it does mean the
CURRENT Q.16 format, if used verbatim for a real-tile hardware
implementation, would introduce a coarser scale-quantization step than
the toy-patch tests ever exposed. A higher `FRAC_BITS` value (e.g. Q.24
or Q.28) would likely be needed for a real-tile hardware implementation
to retain adequate `SCALE_FX` precision; this report only surfaces the
issue via the `--frac-bits` check already built into the generator
script, it does not change the format or re-verify at a different
`FRAC_BITS`.

## Limitations

- **2 of 32 output channels verified** (kernel 0 and kernel 2), not the
  complete first Conv2d layer.
- **One tile, one 6x6 sub-block location** (`tile_16_42.tif`, offset
  125,125). A different tile or offset was not tried.
- **Valid (no-padding) convolution only** -- matches every VHDL prototype
  in this repo; padding is not implemented anywhere in VHDL.
- **Simulation-only (GHDL).** No synthesis was re-run for these new
  designs (the underlying `stream_conv3x3_3chan_cell` sub-component and
  pipeline structure were already synthesized as part of the existing
  toy-patch designs; the new top-level entities differ from those only in
  which package's constants they reference, so no new synthesis risk is
  expected, but none was performed here to confirm timing).
- **No board testing, no measured speedup, no measured power.**
- **`SCALE_FX` precision finding (above) is a discovered limitation of
  the existing Q.16 convention for real-scale data, not a bug in this
  verification** -- both VHDL and Python round it identically, so this
  report's PASS result is unaffected; it only means a REAL hardware
  deployment (not toy-patch verification) may want more fractional bits.

## Recommended next step

Given the kernel0/kernel2 subset passed bit-exact with zero max
difference, two follow-ons are natural, in order of increasing scope:

1. **Try a second real tile and/or a different sub-block offset** with
   the same generator script (`--tile-name`, `--row-offset`,
   `--col-offset` are already CLI-configurable) to build confidence this
   result generalizes beyond one specific 6x6 patch, before committing to
   the larger 32-output extension.
2. **Extend to the full 32-output resource-shared design**
   (`stream_conv3x3_3chan_32out_bn_relu_time_mux.vhd`), which was
   assessed but deliberately deferred here (see "Why kernel 0 + kernel 2,
   not the full 32-output design" above) -- this would need a real-tile
   variant of `first_layer_32out_bn_relu_resource_shared_pkg.vhd` with 32
   real-scale-based `SCALE_FX`/`BIAS_FX` pairs and 32 golden arrays
   (512 total checks for a 16-window real patch), reusing this task's
   generator-script pattern generalized across all 32 kernels at once.
3. **Re-run this same real-tile generator at a higher `--frac-bits`**
   (e.g. 24 or 28) to check whether the `SCALE_FX` precision finding
   above meaningfully changes the fixed-point golden values, before
   considering a `FRAC_BITS` change for any future real-tile-targeted
   VHDL design.
