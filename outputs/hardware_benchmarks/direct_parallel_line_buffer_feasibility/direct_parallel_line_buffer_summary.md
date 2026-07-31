# Direct-Parallel First-Layer Timing: Corrected for Streaming Front-End Overhead

Source data: `direct_parallel_line_buffer_cycle_estimates.csv`, derived from
the GHDL-measured formula confirmed in
`hardware/vhdl_conv3x3/reports/direct_parallel_line_buffer_feasibility_summary.md`:
**`total_cycles = (H * W) + 6`** (streaming a real `H x W` image in,
through the new 3-channel front end, into the existing direct-parallel
folded arithmetic core) -- replacing the earlier, front-end-free
`windows + 6` formula used in
`outputs/hardware_benchmarks/first_layer_architecture_tile_throughput/`.

**Not a board measurement.** Cycle-count-derived from GHDL-confirmed
front-end + arithmetic-core simulation behavior (see the hardware report
for the full derivation and caveats).

## Corrected vs. previous (no-front-end) direct-parallel timing @ 100 MHz

| Tile | Old (`windows+6`) cycles | Corrected (`H*W+6`) cycles | Old (us) | Corrected (us) | Increase |
|---|---:|---:|---:|---:|---:|
| 6x6 | 22 | 42 | 0.220 | 0.420 | +90.9% |
| 128x128 | 15,882 | 16,390 | 158.82 | 163.90 | +3.2% |
| 256x256 | 64,522 | 65,542 | 645.22 | 655.42 | +1.6% |

300 MHz figures (hypothetical, unsynthesized) are in the CSV for
completeness.

## Interpretation

Streaming-front-end overhead is a **large relative correction at tiny
scale** (6x6: +91%, since fixed fill/drain cost dominates a nearly-empty
pipeline) but a **small correction at realistic tile scale** (128x128:
+3.2%; 256x256: +1.6%). The earlier tile-throughput report's
"pre-extracted windows, no line-buffer overhead" direct-parallel estimate
was NOT substantially misleading at application scale, but it should be
labeled as a slight underestimate, not an exact figure, going forward.
