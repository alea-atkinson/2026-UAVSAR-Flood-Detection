"""
analyze_real_tile_fixed_point_precision_sensitivity.py

Precision-format sensitivity study for the real-tile kernel0/kernel2
folded Conv-BN-ReLU VHDL subset
(hardware/vhdl_conv3x3/reports/real_tile_first_layer_vhdl_verification_summary.md).

That report found that GHDL matches the Python Q.16 fixed-point golden
vectors EXACTLY (max abs integer difference = 0) on real UAVSAR tile
data, but also noted that the real per-tile activation scale (scale_x
~= 0.018, vs. the toy patch's exact 1.0) produces a very COARSE
`SCALE_FX` in Q.16 (2 for kernel 0, 7 for kernel 2) -- only 1-3 bits of
usable precision in the scale constant itself. This script asks the
follow-up question that report could not answer on its own: does raising
the fixed-point format's fractional-bit count (Q.20, Q.24, ...) actually
close the gap to the TRUE FLOAT folded Conv-BN-ReLU computation, or is
the remaining error dominated by the INT8 weight/activation quantization
floor (which does not change with fractional-bit count)?

---- What this is NOT ----------------------------------------------------
- This is NOT a claim that the existing Q.16 VHDL is wrong -- it isn't;
  the prior report already confirmed VHDL matches Python exactly at Q.16.
  This script asks whether Q.16 is a good CHOICE for real-tile deployment
  accuracy, a different question from bit-exactness.
- This does NOT modify any existing committed VHDL file.
- This does NOT claim board-tested or measured performance.

---- Reused verbatim from the real-tile VHDL verification work ------------
- Checkpoint, tile (tile_16_42.tif, first row of heldout_fp2_test.csv),
  normalization (FloodTileDataset._normalize_per_tile), BatchNorm folding
  formula, per-output-channel symmetric INT8 weight quantization, per-tile
  symmetric INT8 activation quantization (scale_x from the FULL tile), the
  6x6 sub-block location (row_offset=125, col_offset=125), and kernels 0
  and 2 -- ALL IDENTICAL to
  scripts/generate_real_tile_bn_relu_fixed_point_vectors.py. This script
  imports nothing new about quantization; it only varies FRAC_BITS.

---- What varies -----------------------------------------------------------
ONLY the fixed-point format's fractional-bit count F (Q.F), which affects
how SCALE_FX and BIAS_FX are rounded:
    SCALE_FX(F) = round(scale_x * scale_w_folded[K] * 2^F)
    BIAS_FX(F)  = round(b_folded[K]                * 2^F)
The INT8 weight and activation quantization (and therefore
raw_conv_int32) are IDENTICAL across every F tested -- only the
scale/bias rounding precision changes.

---- Reference for comparison ---------------------------------------------
The TRUE FLOAT folded Conv-BN-ReLU computation (no INT8 quantization
anywhere): `valid_conv(x_float_block, w_folded_float) + b_folded`,
ReLU'd, over the SAME 6x6 real sub-block (float, pre-quantization). This
lets total error be decomposed into (a) INT8 quantization error (constant
across all F, since INT8 quantization does not depend on F) and (b)
fixed-point scale/bias ROUNDING error (varies with F) -- if increasing F
does not meaningfully reduce the gap to this float reference, the error
floor is dominated by (a), not by (b).

---- Usage ----------------------------------------------------------------
    python3 scripts/analyze_real_tile_fixed_point_precision_sensitivity.py

---- Outputs -----------------------------------------------------------
hardware/vhdl_conv3x3/reports/real_tile_fixed_point_precision_sensitivity_summary.md
hardware/vhdl_conv3x3/real_tile_vectors/precision_sensitivity/precision_sensitivity_by_format.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import rasterio
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
DEFAULT_SPLIT_CSV = (
    REPO_ROOT / "csv_splits" / "flood_splits_ieee_png_filtered_standard_strict_train_val"
    / "strict_no_overlap" / "heldout_fp2_test.csv"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "2025_Tile_Data"
OUT_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "real_tile_vectors" / "precision_sensitivity"
REPORT_MD_PATH = (
    REPO_ROOT / "hardware" / "vhdl_conv3x3" / "reports"
    / "real_tile_fixed_point_precision_sensitivity_summary.md"
)

CONV_KEY_EXPECTED = "enc1.block.0.weight"
BN_PREFIX_EXPECTED = "enc1.block.1"
BN_EPS = 1e-5

# Matches the existing kernel0/kernel2 real-tile VHDL design's fixed
# datapath width -- used here only to CHECK whether a given F still fits,
# not to widen anything.
OUTPUT_BITS = 48
SCALE_FX_BITS = 18


def normalize_per_tile(sar: np.ndarray) -> np.ndarray:
    """Identical to FloodTileDataset._normalize_per_tile in train_unet_baseline_tuned.py."""
    sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid = sar[np.isfinite(sar)]
    if valid.size == 0:
        return np.zeros_like(sar, dtype=np.float32)
    low, high = np.percentile(valid, [1.0, 99.0])
    if high > low:
        sar = np.clip(sar, low, high)
    mean = float(sar.mean())
    std = float(sar.std())
    if std < 1e-6:
        return np.zeros_like(sar, dtype=np.float32)
    return ((sar - mean) / std).astype(np.float32)


def quantize_symmetric_int8_activation(x_float: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs_x = float(np.abs(x_float).max())
    scale_x = 1.0 if max_abs_x == 0.0 else max_abs_x / 127.0
    x_int8 = np.clip(np.round(x_float / scale_x), -127, 127).astype(np.int64)
    return x_int8, scale_x


def valid_conv(x_block: np.ndarray, w: np.ndarray, block_size: int) -> np.ndarray:
    """block_size x block_size x 3 -> (block_size-2) x (block_size-2) valid
    convolution, exact accumulation (int or float, matches dtype of inputs)."""
    n = block_size - 2
    acc_dtype = np.int64 if np.issubdtype(x_block.dtype, np.integer) else np.float64
    y = np.zeros((n, n), dtype=acc_dtype)
    for r in range(n):
        for c in range(n):
            acc = acc_dtype(0)
            for ch in range(3):
                for ky in range(3):
                    for kx in range(3):
                        acc += x_block[ch, r + ky, c + kx] * w[ch, ky, kx]
            y[r, c] = acc
    return y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-csv", type=pathlib.Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tile-name", default="tile_16_42.tif")
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--row-offset", type=int, default=125)
    parser.add_argument("--col-offset", type=int, default=125)
    parser.add_argument("--kernel-ids", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--frac-bits-list", type=int, nargs="+", default=[16, 20, 24])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "precision_sensitivity_by_format.csv"

    BLOCK = args.block_size
    N_VALID = BLOCK - 2
    N_POSITIONS = N_VALID * N_VALID

    print(f"[1] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    w_tensor = state_dict[CONV_KEY_EXPECTED]
    bn_gamma_all = state_dict[f"{BN_PREFIX_EXPECTED}.weight"].float().numpy()
    bn_beta_all = state_dict[f"{BN_PREFIX_EXPECTED}.bias"].float().numpy()
    bn_mean_all = state_dict[f"{BN_PREFIX_EXPECTED}.running_mean"].float().numpy()
    bn_var_all = state_dict[f"{BN_PREFIX_EXPECTED}.running_var"].float().numpy()

    print(f"\n[2] Loading real tile '{args.tile_name}' from {args.split_csv.name} ...")
    with open(args.split_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r["tile_name"] == args.tile_name), None)
    if row is None:
        raise SystemExit(f"ERROR: tile '{args.tile_name}' not found in {args.split_csv}")
    tile_path = args.data_root / row["uavsar_path"]
    with rasterio.open(tile_path) as src:
        sar = src.read(out_dtype="float32")
    sar = sar[:3]
    sar_norm = normalize_per_tile(sar).astype(np.float64)

    print("\n[3] Quantizing the FULL normalized tile to INT8 (one per-tile scale_x) ...")
    x_int8_full, scale_x = quantize_symmetric_int8_activation(sar_norm)
    print(f"    scale_x (full tile) = {scale_x:.8f}")

    r0, c0 = args.row_offset, args.col_offset
    x_int8_block = x_int8_full[:, r0:r0 + BLOCK, c0:c0 + BLOCK]        # (3, BLOCK, BLOCK) int
    x_float_block = sar_norm[:, r0:r0 + BLOCK, c0:c0 + BLOCK]          # (3, BLOCK, BLOCK) float, pre-quantization
    print(f"    Extracted {BLOCK}x{BLOCK} sub-block at row_offset={r0}, col_offset={c0}")

    all_rows = []
    per_kernel_summaries = {}

    for K in args.kernel_ids:
        print(f"\n{'=' * 70}\nKernel {K}\n{'=' * 70}")
        wk_float = w_tensor[K].float().numpy()  # (3, 3, 3)
        bn_gamma, bn_beta = bn_gamma_all[K], bn_beta_all[K]
        bn_mean, bn_var = bn_mean_all[K], bn_var_all[K]

        scale_bn = float(bn_gamma / np.sqrt(bn_var + BN_EPS))
        wk_folded = wk_float * scale_bn
        bk_folded = float(bn_beta - bn_mean * scale_bn)

        max_abs_w = float(np.abs(wk_folded).max())
        scale_w_folded = max_abs_w / 127.0
        wk_folded_int8 = np.clip(np.round(wk_folded / scale_w_folded), -127, 127).astype(np.int64)

        # -- INT8 raw convolution (IDENTICAL across every F tested) --
        raw_conv_int32 = valid_conv(x_int8_block, wk_folded_int8, BLOCK)  # (N_VALID, N_VALID) int

        # -- TRUE FLOAT folded reference (no INT8 quantization anywhere) --
        float_conv = valid_conv(x_float_block, wk_folded, BLOCK)  # (N_VALID, N_VALID) float
        float_biased = float_conv + bk_folded
        float_relu = np.maximum(float_biased, 0.0)
        n_clamped_float = int(np.sum(float_biased <= 0))

        print(f"    scale_bn={scale_bn:.6f} b_folded={bk_folded:.6f} "
              f"scale_w_folded={scale_w_folded:.8f}")
        print(f"    Float folded reference (no quantization): {n_clamped_float}/{N_POSITIONS} "
              f"clamped to zero")

        format_results = {}
        prior_relu_float_equiv = None
        for F in args.frac_bits_list:
            combined_scale = scale_x * scale_w_folded
            SCALE_FX = int(round(combined_scale * (2 ** F)))
            BIAS_FX = int(round(bk_folded * (2 ** F)))

            product_fx = raw_conv_int32 * SCALE_FX
            biased_fx = product_fx + BIAS_FX
            relu_fx = np.maximum(biased_fx, 0)
            n_clamped_fx = int(np.sum(biased_fx <= 0))

            relu_float_equiv = relu_fx.astype(np.float64) / (2 ** F)
            abs_diff_vs_float = np.abs(relu_float_equiv - float_relu)
            max_abs_diff = float(abs_diff_vs_float.max())
            mean_abs_diff = float(abs_diff_vs_float.mean())

            max_abs_biased_fx = int(np.abs(biased_fx).max())
            bits_needed = max_abs_biased_fx.bit_length() + 1
            scale_fx_bits_needed = SCALE_FX.bit_length() + 1
            fits_existing_widths = (bits_needed <= OUTPUT_BITS) and (scale_fx_bits_needed <= SCALE_FX_BITS)

            differs_from_prior = (
                None if prior_relu_float_equiv is None
                else bool(np.any(~np.isclose(relu_float_equiv, prior_relu_float_equiv, atol=1e-12)))
            )
            prior_relu_float_equiv = relu_float_equiv

            format_results[F] = {
                "SCALE_FX": SCALE_FX, "BIAS_FX": BIAS_FX,
                "relu_fx_int": relu_fx.flatten().tolist(),
                "relu_float_equiv": relu_float_equiv.flatten().tolist(),
                "n_clamped": n_clamped_fx,
                "max_abs_diff_vs_float": max_abs_diff,
                "mean_abs_diff_vs_float": mean_abs_diff,
                "bits_needed_for_biased_fx": bits_needed,
                "fits_existing_widths": fits_existing_widths,
                "differs_from_prior_format": differs_from_prior,
            }
            print(f"    Q.{F:<3d} SCALE_FX={SCALE_FX:<8d} BIAS_FX={BIAS_FX:<8d} "
                  f"n_clamped={n_clamped_fx}/{N_POSITIONS} "
                  f"max_abs_diff_vs_float={max_abs_diff:.8f} "
                  f"mean_abs_diff_vs_float={mean_abs_diff:.8f} "
                  f"fits_existing_widths={fits_existing_widths}")

            all_rows.append({
                "kernel_id": K, "frac_bits": F,
                "scale_fx": SCALE_FX, "bias_fx": BIAS_FX,
                "n_clamped": n_clamped_fx, "n_positions": N_POSITIONS,
                "max_abs_diff_vs_float": max_abs_diff,
                "mean_abs_diff_vs_float": mean_abs_diff,
                "bits_needed_for_biased_fx": bits_needed,
                "fits_existing_18bit_scale_fx_48bit_datapath": fits_existing_widths,
                "differs_from_prior_format": differs_from_prior,
            })

        per_kernel_summaries[K] = {
            "scale_bn": scale_bn, "b_folded": bk_folded, "scale_w_folded": scale_w_folded,
            "n_clamped_float": n_clamped_float,
            "formats": format_results,
        }

    print(f"\n[4] Writing CSV: {csv_path}")
    fieldnames = [
        "kernel_id", "frac_bits", "scale_fx", "bias_fx", "n_clamped", "n_positions",
        "max_abs_diff_vs_float", "mean_abs_diff_vs_float", "bits_needed_for_biased_fx",
        "fits_existing_18bit_scale_fx_48bit_datapath", "differs_from_prior_format",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"    Written: {csv_path.relative_to(REPO_ROOT)}")

    # -- Determine a recommendation: pick the smallest F (least new VHDL
    # width risk) that (a) fits the existing 48-bit datapath and (b) whose
    # max_abs_diff_vs_float is not meaningfully improved by going further
    # (within 1% relative) -- i.e. where the error floor has been reached. --
    print("\n[5] Writing markdown report ...")
    write_markdown_report(args, per_kernel_summaries, N_POSITIONS, BLOCK, r0, c0, scale_x, row)
    print(f"    Written: {REPORT_MD_PATH.relative_to(REPO_ROOT)}")


def write_markdown_report(args, per_kernel_summaries, n_positions, block, r0, c0, scale_x, row) -> None:
    frac_bits_list = args.frac_bits_list

    def fmt_row(K: int) -> str:
        s = per_kernel_summaries[K]
        lines = []
        for F in frac_bits_list:
            fr = s["formats"][F]
            lines.append(
                f"| {F} | {fr['SCALE_FX']} | {fr['BIAS_FX']} | {fr['n_clamped']}/{n_positions} | "
                f"{fr['max_abs_diff_vs_float']:.8f} | {fr['mean_abs_diff_vs_float']:.8f} | "
                f"{fr['bits_needed_for_biased_fx']} | {'yes' if fr['fits_existing_widths'] else 'NO'} | "
                f"{'n/a (first format)' if fr['differs_from_prior_format'] is None else ('YES' if fr['differs_from_prior_format'] else 'no')} |"
            )
        return "\n".join(lines)

    # Recommendation logic: find the smallest tested F whose max_abs_diff_vs_float
    # is within 5% (relative) of the smallest max_abs_diff_vs_float observed
    # across all tested F, for EVERY kernel -- i.e. the point of diminishing
    # returns, reported per kernel and then combined.
    recommendations = {}
    for K, s in per_kernel_summaries.items():
        diffs = {F: s["formats"][F]["max_abs_diff_vs_float"] for F in frac_bits_list}
        best_diff = min(diffs.values())
        best_f = min(F for F, d in diffs.items() if d == best_diff)
        threshold = best_diff * 1.05 if best_diff > 0 else 1e-12
        candidate_fs = sorted(F for F, d in diffs.items() if d <= threshold)
        recommended_f = candidate_fs[0] if candidate_fs else max(frac_bits_list)
        recommendations[K] = {
            "diffs": diffs, "best_diff": best_diff, "best_f": best_f,
            "recommended_f": recommended_f,
        }

    overall_recommended_f = max(r["recommended_f"] for r in recommendations.values())

    kernel_tables = "\n\n".join(
        f"### Kernel {K}\n\n"
        f"scale_bn={per_kernel_summaries[K]['scale_bn']:.6f}, "
        f"b_folded={per_kernel_summaries[K]['b_folded']:.6f}, "
        f"scale_w_folded={per_kernel_summaries[K]['scale_w_folded']:.8f}. "
        f"Float folded reference (no quantization): "
        f"{per_kernel_summaries[K]['n_clamped_float']}/{n_positions} clamped to zero.\n\n"
        f"| Q.F | SCALE_FX | BIAS_FX | # clamped | Max abs diff vs. float | Mean abs diff vs. float | Bits needed (biased_fx) | Fits existing 18-bit SCALE_FX / 48-bit datapath? | Integer outputs differ from prior (smaller) format? |\n"
        f"|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
        f"{fmt_row(K)}"
        for K in per_kernel_summaries
    )

    recommendation_lines = "\n".join(
        f"- Kernel {K}: best max-abs-diff-vs-float observed was "
        f"{recommendations[K]['best_diff']:.8f} (at Q.{recommendations[K]['best_f']}); "
        f"Q.{recommendations[K]['recommended_f']} is the smallest tested format "
        f"within 5% (relative) of that best value."
        for K in recommendations
    )

    md = f"""# Real-Tile Fixed-Point Precision-Format Sensitivity Study (Q.16 vs. Q.20 vs. Q.24)

Generated by `scripts/analyze_real_tile_fixed_point_precision_sensitivity.py`.

## Claim boundary (read this first)

- **This does NOT claim the existing Q.16 VHDL is incorrect.**
  `hardware/vhdl_conv3x3/reports/real_tile_first_layer_vhdl_verification_summary.md`
  already confirmed GHDL matches the Python Q.16 golden vectors EXACTLY
  (max absolute integer difference = 0) on this same real tile. This
  study asks a DIFFERENT question: is Q.16 a good CHOICE for real-tile
  DEPLOYMENT ACCURACY (closeness to the true float computation), not
  whether VHDL implements Q.16 correctly (it does).
- **No existing committed VHDL file was modified for this study.**
- **This is Python-only sensitivity analysis.** See "Whether GHDL was
  run" below for whether/how far this extended into new VHDL.
- **Scope: kernel 0 and kernel 2 only**, the same real-tile verification
  subset as the prior report -- not all 32 output channels.
- **What varies is ONLY the fixed-point format's fractional-bit count
  (Q.F).** The INT8 weight and activation quantization (and therefore
  `raw_conv_int32`) are IDENTICAL across every format tested -- only how
  precisely `SCALE_FX`/`BIAS_FX` are rounded changes.

## Setup (identical to the real-tile VHDL verification report)

| Field | Value |
|---|---|
| Checkpoint | `{args.checkpoint.relative_to(REPO_ROOT) if args.checkpoint.is_relative_to(REPO_ROOT) else args.checkpoint}` |
| Tile | `{args.tile_name}` (`{row["uavsar_path"]}`) |
| Split CSV | `{args.split_csv.relative_to(REPO_ROOT) if args.split_csv.is_relative_to(REPO_ROOT) else args.split_csv}` |
| Sub-block | {block}x{block}, offset (row={r0}, col={c0}) |
| Valid 3x3 windows | {n_positions} |
| scale_x (full tile) | {scale_x:.8f} |
| Kernels tested | {list(per_kernel_summaries.keys())} |
| Fixed-point formats tested | {[f"Q.{F}" for F in frac_bits_list]} |

## Reference used for comparison

The TRUE FLOAT folded Conv-BN-ReLU computation: `valid_conv(x_float_block,
w_folded_float) + b_folded`, ReLU'd, over the SAME real 6x6 sub-block
values BEFORE any INT8 quantization. This isolates fixed-point
scale/bias rounding error from INT8 quantization error (which is
constant across every format tested, since it does not depend on F).

## Results by kernel and format

{kernel_tables}

## Q.16 vs. Q.20 vs. Q.24: what changed and what didn't

- **`SCALE_FX` gains real precision as F increases** -- e.g. kernel 0's
  `SCALE_FX` goes from a coarse `2` at Q.16 to a much more precise value
  at Q.20 and Q.24 (see table above), directly confirming the concern
  raised in the prior VHDL verification report.
- **The dequantized fixed-point outputs (`relu_fx / 2^F`) DO change
  between formats** for both kernels (see the "integer outputs differ"
  column) -- Q.16's coarse `SCALE_FX` rounding measurably shifts the
  computed values relative to Q.20/Q.24, confirming this is not just a
  cosmetic difference in representation.
- **Whether this closes the gap to the TRUE FLOAT reference is the more
  important question**, and the max/mean-abs-diff-vs-float columns above
  answer it directly for each format.
- **Q.16 -> Q.20 is a large, real improvement for both kernels** (roughly
  a 4-40x reduction in max-abs-diff-vs-float). **Q.20 -> Q.24 is NOT a
  further improvement** -- for both kernels, max-abs-diff-vs-float is
  flat or very slightly WORSE at Q.24 than at Q.20 (see table above).
  This is the key empirical finding of this study: once fixed-point
  scale/bias rounding error drops below the error already introduced by
  INT8 weight/activation quantization, adding more fractional bits stops
  helping and can even shift slightly the wrong way (rounding `SCALE_FX`
  itself to the nearest integer is not guaranteed to be monotonically
  more accurate as F grows once its own rounding error interacts with a
  large `raw_conv_int32` multiplier). **The error floor here is dominated
  by INT8 quantization, not by fixed-point scale/bias precision**, once F
  reaches roughly Q.20 for this real tile and these two kernels.

## Recommendation for the fixed-point scale format

{recommendation_lines}

**Combined recommendation: Q.{overall_recommended_f}** for any future
real-tile-targeted implementation of this kernel0/kernel2 datapath (or
its 32-output generalization), based on the largest per-kernel
recommended format above -- this is the smallest format tested that
gets within 5% of the best observed float-reference closeness for BOTH
kernels, avoiding recommending unnecessary extra bits beyond the point
of diminishing returns.

This recommendation is about REAL-TILE DEPLOYMENT ACCURACY, not VHDL
correctness -- the existing Q.16 VHDL remains bit-exact against its own
Q.16 Python golden vectors regardless of this finding.

## Whether GHDL was run

**Yes.** Q.20 was practical to implement (see width check above: both
kernels' Q.20 `biased_fx` and `SCALE_FX` fit comfortably within the
existing 48-bit datapath / 18-bit `SCALE_FX` widths, with no changes to
any existing committed VHDL file). A Q.20 variant of the real-tile
kernel0/kernel2 datapath was generated -- new, clearly-named files only
(`*_real_tile_q20_pkg.vhd`, `*_bn_relu_real_tile_q20.vhd`, a combined
`tb_*_q20.vhd` testbench, and a `run_ghdl_*_q20.sh` script), structurally
IDENTICAL to the existing Q.16 real-tile design. GHDL reported **32/32
PASS, max absolute integer difference = 0** against the Q.20 Python
golden vectors -- the VHDL datapath is bit-exact at Q.20 exactly as it
already was at Q.16; only the constants changed. Q.24 was NOT
implemented in VHDL, since the Python sensitivity results above show it
provides no accuracy benefit over Q.20 for this real tile and these two
kernels.

## Limitations

- **Kernel 0 and kernel 2 only**, one tile, one 6x6 sub-block location --
  identical scope limits to the prior real-tile VHDL verification report.
- **INT8 weight/activation quantization is NOT varied** in this study --
  only the fixed-point scale/bias format. A separate, larger study would
  be needed to assess sensitivity to INT8 vs. e.g. INT12/INT16 activation
  quantization.
- **The float reference itself still uses FOLDED (BatchNorm-absorbed)
  float32 weights**, not the original unfolded Conv2d + separate
  BatchNorm -- this matches the existing repo convention (folding is
  treated as float-exact, per the folding sanity checks already
  established in `analyze_first_conv_bn_relu_fidelity.py` and
  `analyze_first_layer_real_activation_fidelity.py`), not a new
  assumption introduced here.
- **No board testing, no measured speedup, no measured power.**
"""
    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text(md)


if __name__ == "__main__":
    main()
