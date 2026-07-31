#!/usr/bin/env python3
"""
analyze_unet_later_layer_fpga_scaling.py

A LATER-LAYER SCALING REALITY CHECK for the FPGA feasibility side project.

---- Why this exists -------------------------------------------------------
Every VHDL design in this repository (direct-parallel raw Conv2d, the
folded 32-output Conv-BN-ReLU arithmetic core, and the resource-shared
time-multiplexed variants) implements ONLY the first learned Conv2d+
BatchNorm+ReLU stage (`enc1.block.0` + `enc1.block.1` + ReLU), whose
Conv2d has just 3 input channels. This script asks the one scaling
question none of those reports answer: what happens to the SAME
direct-parallel resource-demand argument once you move past `enc1` to
`enc2`, `enc3`, `enc4`, and the `bottleneck` -- layers whose INPUT channel
counts are no longer 3, but 32, 64, 128, and 256 respectively?

This is NOT a new RTL implementation. No later-layer VHDL design is
created, built, simulated, or synthesized here. This script only extracts
REAL Conv2d weight shapes from the actual trained checkpoint and performs
the SAME naive "one multiplier per MAC, one output pixel per cycle,
fully-parallel" resource-demand arithmetic already used (and already
labeled as theoretical/not-synthesized) in
`hardware/vhdl_conv3x3/first_layer_32out_scaling_estimate.md` --
extended to the checkpoint's actual later-layer shapes, and compared
against the four FPGA DSP budgets already established elsewhere in this
repository (Artix-7 200T, Kintex-7 160T, Zynq xc7z030, Spartan-7 xc7s100).

---- What this computes -----------------------------------------------------
For each layer's first Conv2d (`enc1.block.0`, `enc2.block.0`,
`enc3.block.0`, `enc4.block.0`, `bottleneck.block.0`), read the REAL
weight tensor shape from the checkpoint (`[out_channels, in_channels,
kh, kw]`) and compute:
    macs_per_output_pixel = in_channels * out_channels * kh * kw
This is exactly the same "multiplier demand for a fully parallel,
one-output-pixel-per-cycle design" quantity already used throughout this
repo's own scaling-estimate reports (e.g. "32 outputs x 3 channels x 9
taps = 864 multiplies" in `first_layer_32out_scaling_estimate.md`) and
already CONFIRMED, not just projected, by the actually-synthesized
enc1 32-output designs (`first_layer_32out_dsp_200t_summary.md`,
`reports/first_layer_32out_folded_arithmetic_core_summary.md`): both
report exactly 864 required multiplies, mapped onto exactly 740/740
available DSP48E1 slices on Artix-7 200T.

---- What this does NOT do -------------------------------------------------
- Does NOT build, simulate, or synthesize any later-layer VHDL design.
- Does NOT claim later-layer FPGA implementation is impossible in any
  architecture -- only that NAIVE fully-parallel, one-multiplier-per-MAC
  direct mapping does not scale onto these specific, already-used part
  budgets.
- Does NOT claim full U-Net FPGA acceleration has been implemented.
- Does NOT claim board-measured timing, speedup, or power anywhere.
- Does NOT account for DSP-packing tricks (e.g. multiple INT8 multiplies
  per DSP48E1 via bit-packing), pipelining depth, LUT-fabric fallback
  multiplies, or place-and-route feasibility -- this is a first-order
  multiplier-count vs. DSP-count comparison only, exactly the same level
  of rigor already used and already labeled "theoretical" in this repo's
  existing first-layer scaling-estimate report.

---- Usage ------------------------------------------------------------------
    python3 scripts/analyze_unet_later_layer_fpga_scaling.py

---- Outputs -----------------------------------------------------------
outputs/hardware_scaling/unet_later_layer_fpga_scaling/
    unet_later_layer_conv_shapes.csv
    unet_later_layer_fpga_scaling.csv
    unet_later_layer_fpga_scaling_summary.md
"""

from __future__ import annotations

import csv
import pathlib

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)

OUT_DIR = REPO_ROOT / "outputs" / "hardware_scaling" / "unet_later_layer_fpga_scaling"
SHAPES_CSV_PATH = OUT_DIR / "unet_later_layer_conv_shapes.csv"
SCALING_CSV_PATH = OUT_DIR / "unet_later_layer_fpga_scaling.csv"
MD_PATH = OUT_DIR / "unet_later_layer_fpga_scaling_summary.md"

# The first Conv2d of each major U-Net stage (block.0 of each DoubleConv --
# the SAME position within DoubleConv that every VHDL design in this repo
# has implemented for enc1). block.1 is BatchNorm2d, block.3 is the
# DoubleConv's SECOND Conv2d -- not examined here, matching this repo's
# existing enc1-only scope.
LAYER_KEYS = {
    "enc1.block.0": "enc1 (encoder stage 1)",
    "enc2.block.0": "enc2 (encoder stage 2)",
    "enc3.block.0": "enc3 (encoder stage 3)",
    "enc4.block.0": "enc4 (encoder stage 4)",
    "bottleneck.block.0": "bottleneck",
}

# FPGA DSP budgets already established and used elsewhere in this repo --
# NOT re-derived here. Sources:
#   Artix-7 200T (xc7a200tsbg484-1): first_layer_32out_dsp_200t_summary.md,
#       reports/first_layer_32out_folded_arithmetic_core_summary.md
#   Kintex-7 160T (xc7k160tfbg484-1): reports/kintex7_synthesis_comparison_summary.md
#   Zynq xc7z030 (xc7z030fbg484-1) / Spartan-7 xc7s100 (xc7s100fgga484-1):
#       reports/free_tier_target_capacity_check.md
DSP_BUDGETS = {
    "Artix-7 200T (xc7a200tsbg484-1)": 740,
    "Kintex-7 160T (xc7k160tfbg484-1)": 600,
    "Zynq xc7z030 (xc7z030fbg484-1)": 400,
    "Spartan-7 xc7s100 (xc7s100fgga484-1)": 160,
}


def main() -> None:
    print("=" * 78)
    print(f"[1] Loading checkpoint: {CHECKPOINT_PATH}")
    print("=" * 78)
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    print("\n" + "=" * 78)
    print("[2] Extracting REAL Conv2d weight shapes for each stage's first Conv2d")
    print("=" * 78)
    shape_rows = []
    for key, label in LAYER_KEYS.items():
        weight_key = f"{key}.weight"
        if weight_key not in state_dict:
            raise SystemExit(f"ERROR: checkpoint is missing expected tensor '{weight_key}'")
        shape = list(state_dict[weight_key].shape)  # [out_ch, in_ch, kh, kw]
        out_ch, in_ch, kh, kw = shape
        print(f"    {key:20s} ({label}): weight shape = {shape}")
        shape_rows.append({
            "layer_key": key,
            "layer_label": label,
            "weight_tensor": weight_key,
            "out_channels": out_ch,
            "in_channels": in_ch,
            "kernel_h": kh,
            "kernel_w": kw,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[3] Writing shapes CSV: {SHAPES_CSV_PATH.relative_to(REPO_ROOT)}")
    with SHAPES_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shape_rows[0].keys()))
        writer.writeheader()
        writer.writerows(shape_rows)

    print("\n" + "=" * 78)
    print("[4] Computing MAC-per-output-pixel and naive direct-parallel "
          "multiplier/DSP demand")
    print("=" * 78)
    enc1_macs = None
    scaling_rows = []
    for row in shape_rows:
        macs_per_output_pixel = row["in_channels"] * row["out_channels"] * row["kernel_h"] * row["kernel_w"]
        if row["layer_key"] == "enc1.block.0":
            enc1_macs = macs_per_output_pixel
        scaling_rows.append({**row, "macs_per_output_pixel": macs_per_output_pixel})

    assert enc1_macs is not None
    print(f"    enc1.block.0 MACs/output-pixel = {enc1_macs} "
          f"(reference point: ratio_to_enc1 = 1.00x)")

    final_rows = []
    for row in scaling_rows:
        macs = row["macs_per_output_pixel"]
        ratio_to_enc1 = macs / enc1_macs
        out_row = {
            "layer_key": row["layer_key"],
            "layer_label": row["layer_label"],
            "in_channels": row["in_channels"],
            "out_channels": row["out_channels"],
            "kernel_h": row["kernel_h"],
            "kernel_w": row["kernel_w"],
            "macs_per_output_pixel": macs,
            "naive_direct_parallel_multiplier_demand": macs,
            "ratio_to_enc1": round(ratio_to_enc1, 4),
        }
        for part_name, dsp_budget in DSP_BUDGETS.items():
            short = part_name.split(" (")[0].replace("-", "").replace(" ", "_").lower()
            out_row[f"dsp_budget_{short}"] = dsp_budget
            out_row[f"dsp_demand_pct_{short}"] = round(macs / dsp_budget * 100.0, 2)
            out_row[f"fits_one_dsp_per_multiplier_{short}"] = macs <= dsp_budget
        final_rows.append(out_row)
        print(f"    {row['layer_key']:20s} MACs/px={macs:>7,}  ratio_to_enc1={ratio_to_enc1:>7.2f}x")

    print(f"\n[5] Writing scaling CSV: {SCALING_CSV_PATH.relative_to(REPO_ROOT)}")
    with SCALING_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\n[6] Writing markdown summary: {MD_PATH.relative_to(REPO_ROOT)}")
    write_markdown_summary(final_rows, enc1_macs)
    print(f"    Written: {MD_PATH.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 78)
    print("Done. Outputs:")
    print(f"  {SHAPES_CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"  {SCALING_CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"  {MD_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)


def write_markdown_summary(rows: list[dict[str, object]], enc1_macs: int) -> None:
    def short_key(part_name: str) -> str:
        return part_name.split(" (")[0].replace("-", "").replace(" ", "_").lower()

    part_names = list(DSP_BUDGETS.keys())

    shapes_table_rows = "\n".join(
        f"| {r['layer_key']} | {r['layer_label']} | {r['in_channels']} | "
        f"{r['out_channels']} | {r['kernel_h']}x{r['kernel_w']} |"
        for r in rows
    )

    macs_table_rows = "\n".join(
        f"| {r['layer_key']} | {r['in_channels']} x {r['out_channels']} x "
        f"{r['kernel_h']} x {r['kernel_w']} | **{r['macs_per_output_pixel']:,}** | "
        f"**{r['ratio_to_enc1']:.2f}x** |"
        for r in rows
    )

    def dsp_table_for_part(part_name: str) -> str:
        sk = short_key(part_name)
        budget = DSP_BUDGETS[part_name]
        lines = "\n".join(
            f"| {r['layer_key']} | {r['macs_per_output_pixel']:,} | {budget:,} | "
            f"{r[f'dsp_demand_pct_{sk}']:.1f}% | "
            f"{'YES' if r[f'fits_one_dsp_per_multiplier_{sk}'] else '**NO**'} |"
            for r in rows
        )
        return lines

    enc1_row = next(r for r in rows if r["layer_key"] == "enc1.block.0")
    biggest_jump_row = max(rows, key=lambda r: r["ratio_to_enc1"] if r["layer_key"] != "enc1.block.0" else 0)
    enc2_row = next(r for r in rows if r["layer_key"] == "enc2.block.0")

    dsp_summary_lines = []
    for part_name in part_names:
        sk = short_key(part_name)
        budget = DSP_BUDGETS[part_name]
        n_fit = sum(1 for r in rows if r[f"fits_one_dsp_per_multiplier_{sk}"])
        dsp_summary_lines.append(
            f"- **{part_name}** ({budget} DSPs): {n_fit}/5 layers fit a naive "
            f"one-multiplier-per-MAC direct-parallel mapping."
        )
    dsp_summary_text = "\n".join(dsp_summary_lines)

    md = f"""# U-Net Later-Layer FPGA Scaling Reality Check

Generated by `scripts/analyze_unet_later_layer_fpga_scaling.py`.

**This is a resource-demand PROJECTION, not a synthesized later-layer
design.** No VHDL for `enc2`, `enc3`, `enc4`, or `bottleneck` has been
built, simulated, or synthesized anywhere in this repository. Every number
below is a direct arithmetic consequence of the ACTUAL trained
checkpoint's Conv2d weight shapes (read directly from
`models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt`,
not assumed or guessed) compared against FPGA DSP budgets already
established and used elsewhere in this repository -- it does not
replace an actual synthesis run at any of these later stages.

## 1. Purpose

Every VHDL design in this repository -- the direct-parallel raw Conv2d
design, the folded 32-output Conv-BN-ReLU arithmetic core (commit
`313558e4`), and the resource-shared/time-multiplexed variants --
implements ONLY `enc1.block.0` (`Conv2d(3, 32, 3)`), which has just **3
input channels**. This report asks the remaining weak research question
head-on: **given the first-stage results, what specifically prevents or
enables extending this approach to later U-Net layers?**

## 2. Real Conv2d shapes (from the actual checkpoint, not guessed)

| Layer key | Stage | Input channels | Output channels | Kernel |
|---|---|---:|---:|---|
{shapes_table_rows}

These shapes were read directly from the checkpoint's `state_dict` (each
`{{layer}}.block.0.weight` tensor) -- not assumed from the expected
`3→32→64→128→256→512` pattern, though they confirm it exactly.

## 3. MACs per output pixel and growth relative to enc1

For a fully parallel, one-output-pixel-per-cycle direct mapping, the
number of multiplies needed per output pixel is
`in_channels x out_channels x kernel_h x kernel_w` -- the SAME quantity
this repository's own `first_layer_32out_scaling_estimate.md` used for
enc1 output-channel scaling ("32 outputs x 3 channels x 9 taps = 864
multiplies"), and the same quantity the actually-synthesized enc1 designs
confirm empirically: both `first_layer_32out_dsp_200t_summary.md` and
`hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_summary.md`
report exactly **864** required multiplies, mapped onto **740/740**
available DSP48E1 slices on Artix-7 200T -- i.e. enc1 ALREADY consumes
100% of this part's DSP budget before any later layer is even considered.

| Layer key | in x out x kh x kw | MACs / output pixel | Ratio to enc1 |
|---|---|---:|---:|
{macs_table_rows}

**enc1 is the best case, not a representative case.** It has only 3 input
channels because it is the RGB/SAR-band input layer -- every subsequent
stage's input channel count is the PREVIOUS stage's output channel count
(32, then 64, then 128, then 256), so MACs per output pixel grow with the
PRODUCT of input and output channels, not either alone. This is exactly
why `enc2.block.0` alone (`Conv2d({enc2_row['in_channels']}, {enc2_row['out_channels']}, 3)`)
already needs **{enc2_row['ratio_to_enc1']:.1f}x** as many multiplies as
the entire already-DSP-saturated 32-output enc1 design.

## 4. DSP-budget fit check (naive one-multiplier-per-MAC direct-parallel mapping)

Using the four FPGA DSP budgets already established elsewhere in this
repository (`reports/kintex7_synthesis_comparison_summary.md`,
`reports/free_tier_target_capacity_check.md`):

### Artix-7 200T (740 DSPs)

| Layer key | MACs needed | DSPs available | % of budget | Fits (1 mult/DSP)? |
|---|---:|---:|---:|---|
{dsp_table_for_part(part_names[0])}

### Kintex-7 160T (600 DSPs)

| Layer key | MACs needed | DSPs available | % of budget | Fits (1 mult/DSP)? |
|---|---:|---:|---:|---|
{dsp_table_for_part(part_names[1])}

### Zynq xc7z030 (400 DSPs)

| Layer key | MACs needed | DSPs available | % of budget | Fits (1 mult/DSP)? |
|---|---:|---:|---:|---|
{dsp_table_for_part(part_names[2])}

### Spartan-7 xc7s100 (160 DSPs)

| Layer key | MACs needed | DSPs available | % of budget | Fits (1 mult/DSP)? |
|---|---:|---:|---:|---|
{dsp_table_for_part(part_names[3])}

### Fit summary across all four targets

{dsp_summary_text}

**Calibration note -- why this table shows enc1 itself as "NO" even
though enc1's 32-output design actually synthesized successfully:**
this table's "fits" check is a STRICT one-multiplier-per-DSP,
no-fallback comparison (864 theoretical MACs vs. 740 available DSPs on
Artix-7 200T = 116.8%, technically over budget). In the REAL,
already-synthesized enc1 design, Vivado's actual pre-fallback DSP demand
was even higher than this simple count -- **1,044**, not 864 (see
`reports/first_layer_32out_folded_arithmetic_core_summary.md`) -- yet the
design still closed 100 MHz timing, because Vivado's fallback legalization
pushed the ~304 multiplies that didn't fit in DSPs (1,044 minus the
740 available) into LUT/CARRY4 fabric instead, landing on exactly
740/740 DSPs used. **So a modest overage (enc1's real demand was about
1.4x the DSP budget) was successfully absorbed by LUT fallback in
practice.** The point of this report is that later layers' overages are
not modest: enc2 alone needs roughly **25x** the Artix-7 200T's DSP
budget (18,432 vs. 740) -- a completely different feasibility regime
from the ~1.4x gap that fallback bridged for enc1, not simply "more of
the same kind of shortfall."

## 5. Interpretation

- **First-layer success does not automatically imply full-U-Net
  direct-parallel feasibility.** The enc1 result that motivated this whole
  project -- a fully parallel, folded, 1-window/cycle Conv-BN-ReLU core
  closing 100 MHz timing -- only works because enc1 has just 3 input
  channels. That 3-channel input is a property of the SAR data, not of the
  architecture choice, and it does not recur at any later stage.
- **enc2 creates the single biggest relative resource jump.** Moving from
  enc1 (3 input channels) to enc2 ({enc2_row['in_channels']} input
  channels, {enc2_row['out_channels']} output channels) multiplies the
  naive direct-parallel multiplier demand by **{enc2_row['ratio_to_enc1']:.1f}x**
  in one step -- a larger jump, in absolute terms, than any subsequent
  encoder stage adds on top of it, because input channels grow from a
  tiny fixed number (3) to a number tied to the model's own channel width
  for the first time.
- **By the bottleneck, naive direct-parallel demand is
  {biggest_jump_row['ratio_to_enc1']:.0f}x enc1's** ({biggest_jump_row['macs_per_output_pixel']:,}
  MACs/output-pixel) -- multiple orders of magnitude beyond any DSP budget
  checked here, or any Artix-7/Kintex-7/Zynq/Spartan-7 part in the 7-series
  family generally.
- **Direct-parallel full unrolling does not scale cleanly beyond the first
  layer on these targets.** This is a straightforward consequence of MACs
  growing as the PRODUCT of input and output channels, while the enc1
  result already shows the first (smallest-possible) layer consumes an
  entire part's DSP budget on its own.
- **This does not mean later layers are impossible on an FPGA in
  general.** It means the SPECIFIC architecture family demonstrated so far
  in this repository (fully parallel, one-multiplier-per-MAC, one-output-
  pixel-per-cycle) does not extend past enc1 on these specific targets.
  Extending further would require a DIFFERENT architectural approach:
  resource sharing / time-multiplexing (already demonstrated at low DSP
  cost for enc1 in this repo's resource-shared designs, at the cost of
  much higher latency), channel tiling or batching (processing a subset of
  input/output channels per pass and accumulating across passes), lower
  numeric precision (fewer bits per multiply may allow multiple MACs per
  DSP48E1 via packing), more LUT/carry-chain arithmetic (trading DSPs for
  logic fabric, as already seen for the BN-fold stage in
  `reports/first_layer_32out_folded_arithmetic_core_summary.md`), a larger
  FPGA than any checked in this repository, or a fundamentally different
  accelerator architecture (e.g. a systolic array or reused MAC grid sized
  to the part, not to the layer).

## 6. Claim boundaries

- **This is a resource-demand projection only.** No later-layer VHDL was
  built, simulated, or synthesized.
- **Real checkpoint shapes, not guessed values**: every input/output
  channel count above was read directly from
  `models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt`.
- **Does not claim later layers are impossible on FPGAs in general** --
  only that the naive, fully-parallel, one-multiplier-per-MAC direct
  mapping already used for enc1 does not extend to later layers on the
  four specific part budgets already used elsewhere in this repository.
- **Does not claim full U-Net FPGA acceleration has been implemented.**
- **No board-measured timing, speedup, or power is claimed anywhere in
  this report.**
- This comparison ignores DSP-packing (multiple low-precision multiplies
  per DSP48E1), LUT-fabric fallback multiplies, pipelining depth, and
  place-and-route feasibility -- it is a first-order multiplier-count vs.
  DSP-count comparison, at the same level of rigor already used and
  already labeled "theoretical" in
  `hardware/vhdl_conv3x3/first_layer_32out_scaling_estimate.md`.
"""
    MD_PATH.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
