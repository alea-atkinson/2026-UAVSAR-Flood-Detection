# Streaming Front-End Width Sweep: Resource Scaling (100 MHz, Artix-7 200T)

Source data: `streaming_front_end_width_sweep.csv`, from
`hardware/vhdl_conv3x3/reports/first_layer_32out_folded_arithmetic_core_streaming_front_end_width_sweep_summary.md`
(full derivation, GHDL/Vivado provenance, and claim boundaries there).

**Not a board measurement.** Vivado out-of-context synthesis results only.

## Resource / timing / power vs. IMG_WIDTH

| Design | IMG_WIDTH | LUTs | Registers | DSP48E1 | BRAM | WNS (ns) | Power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Core only (no front end) | N/A | 12,039 | 6,564 | 740/740 | 0 | +2.218 | 1.334 |
| Integrated front end + core | 6 | 13,713 | 7,154 | 740/740 | 0 | +2.218 | 1.441 |
| Integrated front end + core | 128 | 30,120 | 16,156 | 740/740 | 0 | +2.063 | 1.695 |
| Integrated front end + core | 256 | 47,542 | 25,585 | 740/740 | 0 | +2.063 | 1.745 |

## Front-end-only LUT/register cost (total minus core-only baseline)

| IMG_WIDTH | Front-end LUTs | Front-end registers | LUTs per pixel of width (approx.) |
|---:|---:|---:|---:|
| 6 | 1,674 | 590 | 279.0 |
| 128 | 18,081 | 9,592 | 141.3 |
| 256 | 35,503 | 19,021 | 138.7 |

Fitted linear trend (from the 128/256 points): **front-end LUTs ≈ 659 +
136 x IMG_WIDTH** -- consistent with the line buffers' own linear
(3 rows x IMG_WIDTH pixels x 3 channels) storage requirement, not a
superlinear blow-up.

## Bottom line

DSPs remain pinned at 740/740 (100%) regardless of IMG_WIDTH -- the front
end never touches a multiplier. But at a REAL tile width (256), the front
end alone costs **35,503 LUTs (26.4% of the part)**, pushing the combined
design to **47,542 LUTs (35.3%)** -- no longer "small" in absolute terms,
though timing still closes at 100 MHz with positive margin (+2.063 ns).
