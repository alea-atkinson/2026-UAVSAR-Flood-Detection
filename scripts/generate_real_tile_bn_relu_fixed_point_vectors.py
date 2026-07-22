"""
generate_real_tile_bn_relu_fixed_point_vectors.py

Generates a GENUINE fixed-point (integer-only) golden reference, and the
VHDL package that carries it, for the existing kernel-0/kernel-2 folded
Conv-BN-ReLU VHDL prototypes -- using a REAL patch cut from a REAL
held-out UAVSAR tile, instead of the canonical synthetic 5x5x3 toy patch
every other VHDL vector-generation script in this repo has used so far.

---- Why this exists ------------------------------------------------------
Every VHDL Conv-BN-ReLU prototype in this repo (unpipelined, pipelined,
kernel0, kernel2, resource-shared, DSP-aware, ...) has only ever been
verified against the canonical synthetic 5x5x3 toy patch (or, for the
quantization/segmentation-impact fidelity studies, real tiles evaluated
in PYTHON only, never fed through GHDL). This script closes that gap for
a small, tractable real-data subset: it extracts a genuine small patch of
a real fp2 held-out tile's per-tile-normalized SAR values, quantizes it
with the SAME convention used for the full-model fidelity studies, and
generates VHDL golden vectors from it -- so a GHDL testbench can confirm
the EXISTING, UNMODIFIED kernel0/kernel2 folded Conv-BN-ReLU pipelined
datapath produces bit-exact fixed-point integers on real sensor-derived
data, not just synthetic integers.

---- Scope note (read this before trusting the numbers) --------------------
This generates vectors for TWO of the 32 output channels (kernel 0 and
kernel 2) only, reusing the existing
`stream_conv3x3_3chan_kernel{0,2}_bn_relu_pipelined` ARCHITECTURE
unchanged (same `stream_conv3x3_3chan_cell` sub-component, same two-stage
Q.16 BN-fold + ReLU pipeline) -- it is a REAL-DATA VERIFICATION SUBSET,
not full 32-output real-tile verification, and not a new hardware
architecture. Extending to the complete 32-output resource-shared design
would additionally require adapting that design's window-scheduling FSM
to a non-5x5 image geometry and regenerating all 32 kernels' real-tile
golden vectors at once; this was assessed and deliberately deferred (see
the accompanying report under hardware/vhdl_conv3x3/reports/) in favor of
first landing a smaller, well-understood, low-risk real-data subset.

---- Quantization / BN-folding convention -----------------------------
IDENTICAL formulas to scripts/analyze_first_layer_fixed_point_segmentation_impact.py
(itself identical to scripts/analyze_first_layer_real_activation_fidelity.py):
    scale_bn[oc]       = gamma[oc] / sqrt(running_var[oc] + eps)
    w_folded[oc]       = w[oc] * scale_bn[oc]
    b_folded[oc]       = beta[oc] - running_mean[oc] * scale_bn[oc]
    scale_w_folded[oc] = max(|w_folded[oc]|) / 127                     (per-output-channel)
    w_folded_int8[oc]  = round(w_folded[oc] / scale_w_folded[oc]), clipped [-127, 127]
    scale_x            = max(|x_full_tile|) / 127                       (PER FULL TILE, not
                          per sub-block -- see "Real patch extraction" below)
    x_int8_full_tile   = round(x_full_tile / scale_x), clipped [-127, 127]

This is the SAME per-tile activation-quantization convention used by the
real-tile Python fidelity/segmentation-impact studies -- NOT a new
"official" hardware convention, and no VHDL in this repo quantized real
image input before this script.

---- Fixed-point (Q.16) datapath, EXACTLY what the existing VHDL computes --
Same formula as scripts/generate_bn_relu_fixed_point_vectors.py, just with
a REAL (not 1.0) scale_x:
    SCALE_FX = round(scale_x * scale_w_folded[oc] * 2^16)   -- unsigned constant
    BIAS_FX  = round(b_folded[oc]                * 2^16)    -- signed constant
    raw_conv_int32 = sum over 3 channels, 9 taps of (x_int8 * w_folded_int8)  -- exact
    product_fx     = raw_conv_int32 * SCALE_FX                -- exact, Q.16
    biased_fx      = product_fx + BIAS_FX                      -- exact, Q.16
    relu_fx        = biased_fx if biased_fx > 0 else 0          -- exact, Q.16

---- Real patch extraction ------------------------------------------------
1. Load ONE real fp2 held-out tile (default: tile_16_42.tif, the first row
   of heldout_fp2_test.csv), first 3 SAR bands.
2. Normalize the FULL 256x256 tile with FloodTileDataset._normalize_per_tile
   (verbatim from scripts/train_unet_baseline_tuned.py).
3. Quantize the FULL normalized tile to INT8 using ONE per-tile scale_x
   (matching analyze_first_layer_fixed_point_segmentation_impact.py's
   full-tile convention exactly -- NOT a smaller scale computed from just
   the sub-block, so these vectors are a literal, traceable EXCERPT of
   what that Python study already computed for this tile).
4. Extract a small square BLOCK_SIZE x BLOCK_SIZE (default 6x6, giving
   (6-2)^2 = 16 valid 3x3 windows) sub-block of that ALREADY-QUANTIZED
   INT8 tile, at a fixed, documented (--row-offset, --col-offset) location
   (default: near tile center, comfortably inside the 256x256 tile with no
   boundary handling needed).
5. This BLOCK_SIZE x BLOCK_SIZE INT8 sub-block is streamed through the
   existing `stream_conv3x3_3chan_cell` streaming interface exactly like
   the synthetic 5x5 toy patch was -- same interface, same valid (no
   padding) semantics, just IMG_WIDTH=BLOCK_SIZE instead of 5, and real
   pixel values instead of synthetic 1..25/2x/-1x values.

---- Do NOT do here ---------------------------------------------------------
- Does NOT modify stream_conv3x3_3chan_cell, stream_conv3x3_3chan_kernel{0,2}
  _bn_relu_pipelined, or their existing toy-patch packages/testbenches --
  everything here is NEW, separately-named output.
- Does NOT invent a new hardware architecture -- reuses the identical
  existing two-stage Q.16 fold+ReLU pipeline structure.
- Does NOT implement padding -- valid (no-padding) 3x3 windows only, same
  scope as every other VHDL prototype in this repo.
- Does NOT claim board-tested or measured performance.

---- Usage ------------------------------------------------------------------
    python3 scripts/generate_real_tile_bn_relu_fixed_point_vectors.py --kernel-id 0
    python3 scripts/generate_real_tile_bn_relu_fixed_point_vectors.py --kernel-id 2

---- Outputs (kernel-id substituted for <K>) ------------------------------
hardware/vhdl_conv3x3/real_tile_vectors/kernel<K>_real_tile_bn_relu_fixed_point_vectors.json
hardware/vhdl_conv3x3/real_tile_vectors/kernel<K>_real_tile_bn_relu_fixed_point_vectors.csv
hardware/vhdl_conv3x3/real_tile_vectors/kernel<K>_real_tile_bn_relu_fixed_point_summary.md
hardware/vhdl_conv3x3/first_conv_bn_relu_kernel<K>_real_tile_pkg.vhd
hardware/vhdl_conv3x3/real_tile_stimulus_pkg.vhd   (shared pixel-stimulus package,
    written identically regardless of --kernel-id, since both kernels see
    the SAME real patch -- safe to (re)generate, deterministic given the
    same --tile-name/--row-offset/--col-offset)
"""

from __future__ import annotations

import argparse
import csv
import json
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
OUT_VECTORS_DIR = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "real_tile_vectors"
STIMULUS_PKG_PATH = REPO_ROOT / "hardware" / "vhdl_conv3x3" / "real_tile_stimulus_pkg.vhd"

CONV_KEY_EXPECTED = "enc1.block.0.weight"
BN_PREFIX_EXPECTED = "enc1.block.1"
BN_EPS = 1e-5  # torch.nn.BatchNorm2d default

# Widths already used by the existing kernel0/kernel2 pipelined VHDL design
# -- this script CHECKS a real patch's constants against these, it does
# not silently widen anything.
SCALE_FX_BITS = 18   # signed(17 downto 0)
OUTPUT_BITS = 48     # signed(47 downto 0)


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
    """IDENTICAL convention to analyze_first_layer_fixed_point_segmentation_impact.py:
    ONE scale computed from the array's own max magnitude."""
    max_abs_x = float(np.abs(x_float).max())
    scale_x = 1.0 if max_abs_x == 0.0 else max_abs_x / 127.0
    x_int8 = np.clip(np.round(x_float / scale_x), -127, 127).astype(np.int64)
    return x_int8, scale_x


def valid_conv_int32(x_i8_block: np.ndarray, w_i8: np.ndarray, block_size: int) -> np.ndarray:
    """block_size x block_size x 3 -> (block_size-2) x (block_size-2) valid
    convolution, exact INT accumulation (matches VHDL's raw_conv_int32)."""
    n = block_size - 2
    y = np.zeros((n, n), dtype=np.int64)
    for r in range(n):
        for c in range(n):
            acc = 0
            for ch in range(3):
                for ky in range(3):
                    for kx in range(3):
                        acc += int(x_i8_block[ch, r + ky, c + kx]) * int(w_i8[ch, ky, kx])
            y[r, c] = acc
    return y


def vhdl_int_array(vals) -> str:
    return "(" + ", ".join(str(int(v)) for v in vals) + ")"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", type=int, required=True, choices=[0, 2],
                         help="Output channel index of enc1.block.0.weight to process. "
                         "Only 0 and 2 are supported, matching the existing kernel0/kernel2 "
                         "folded Conv-BN-ReLU pipelined VHDL prototypes.")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-csv", type=pathlib.Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tile-name", default="tile_16_42.tif",
                         help="Tile name (matching the split CSV's tile_name column). "
                         "Default: tile_16_42.tif, the first row of heldout_fp2_test.csv.")
    parser.add_argument("--block-size", type=int, default=6,
                         help="Square sub-block size in pixels (default: 6, giving "
                         "(6-2)^2 = 16 valid 3x3 windows).")
    parser.add_argument("--row-offset", type=int, default=125,
                         help="Row offset of the sub-block's top-left corner within the "
                         "256x256 tile (default: 125, near tile center).")
    parser.add_argument("--col-offset", type=int, default=125,
                         help="Column offset of the sub-block's top-left corner (default: 125).")
    parser.add_argument("--frac-bits", type=int, default=16,
                         help="Fixed-point fractional bits (Q.F format). Default 16, same "
                         "as every other folded Conv-BN-ReLU VHDL prototype in this repo.")
    parser.add_argument("--variant-suffix", default="",
                         help="Optional suffix (e.g. '_q20') inserted into the output "
                         "vectors directory, package filename, and VHDL package name, so a "
                         "non-default --frac-bits run NEVER overwrites the default Q.16 "
                         "outputs. Leave empty (default) to reproduce the exact existing "
                         "Q.16 file paths/names unchanged.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    K = args.kernel_id
    FRAC_BITS = args.frac_bits
    BLOCK = args.block_size
    N_VALID = BLOCK - 2
    N_POSITIONS = N_VALID * N_VALID

    SUFFIX = args.variant_suffix

    out_dir = OUT_VECTORS_DIR / f"kernel{K}{SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"kernel{K}{SUFFIX}_real_tile_bn_relu_fixed_point_vectors.json"
    csv_path = out_dir / f"kernel{K}{SUFFIX}_real_tile_bn_relu_fixed_point_vectors.csv"
    md_path = out_dir / f"kernel{K}{SUFFIX}_real_tile_bn_relu_fixed_point_summary.md"
    pkg_path = REPO_ROOT / "hardware" / "vhdl_conv3x3" / f"first_conv_bn_relu_kernel{K}_real_tile{SUFFIX}_pkg.vhd"

    # -- 1. Load checkpoint, extract kernel K's Conv2d weight + BN params --
    print(f"[1] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    w_tensor = state_dict[CONV_KEY_EXPECTED]
    wk_float = w_tensor[K].float().numpy()  # (3, 3, 3)
    bn_gamma = state_dict[f"{BN_PREFIX_EXPECTED}.weight"].float().numpy()[K]
    bn_beta = state_dict[f"{BN_PREFIX_EXPECTED}.bias"].float().numpy()[K]
    bn_mean = state_dict[f"{BN_PREFIX_EXPECTED}.running_mean"].float().numpy()[K]
    bn_var = state_dict[f"{BN_PREFIX_EXPECTED}.running_var"].float().numpy()[K]
    print(f"    Kernel {K}: gamma={bn_gamma:.6f} beta={bn_beta:.6f} "
          f"running_mean={bn_mean:.6f} running_var={bn_var:.6f}")

    # -- 2. Fold BatchNorm into kernel K's weights/bias --
    print(f"\n[2] Folding BatchNorm into kernel {K}'s weights/bias ...")
    scale_bn = float(bn_gamma / np.sqrt(bn_var + BN_EPS))
    wk_folded = wk_float * scale_bn
    bk_folded = float(bn_beta - bn_mean * scale_bn)
    print(f"    scale_bn[{K}] = {scale_bn:.6f}   b_folded[{K}] = {bk_folded:.6f}")

    # -- 3. Symmetric per-channel INT8 quantization of folded weights --
    max_abs_w = float(np.abs(wk_folded).max())
    scale_w_folded = max_abs_w / 127.0
    wk_folded_int8 = np.clip(np.round(wk_folded / scale_w_folded), -127, 127).astype(np.int64)
    print(f"\n[3] scale_w_folded[{K}] = {scale_w_folded:.8f}")
    print(f"    INT8 folded weights ch0/1/2:\n{wk_folded_int8[0]}\n{wk_folded_int8[1]}\n{wk_folded_int8[2]}")

    # -- 4. Load and normalize the REAL tile (full 256x256), same
    # normalization as train_unet_baseline_tuned.py --
    print(f"\n[4] Loading real tile '{args.tile_name}' from {args.split_csv.name} ...")
    with open(args.split_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r["tile_name"] == args.tile_name), None)
    if row is None:
        raise SystemExit(f"ERROR: tile '{args.tile_name}' not found in {args.split_csv}")
    tile_path = args.data_root / row["uavsar_path"]
    if not tile_path.exists():
        raise SystemExit(f"ERROR: tile file not found at {tile_path}")
    with rasterio.open(tile_path) as src:
        sar = src.read(out_dtype="float32")
    sar = sar[:3]
    sar_norm = normalize_per_tile(sar)
    print(f"    Tile shape (post-normalize): {sar_norm.shape}")

    # -- 5. Quantize the FULL tile to INT8 with ONE per-tile scale_x
    # (matches analyze_first_layer_fixed_point_segmentation_impact.py's
    # convention exactly), THEN extract the sub-block from the
    # already-quantized array. --
    print("\n[5] Quantizing the FULL normalized tile to INT8 (one per-tile scale_x) ...")
    x_int8_full, scale_x = quantize_symmetric_int8_activation(sar_norm.astype(np.float64))
    print(f"    scale_x (full tile) = {scale_x:.8f}")

    r0, c0 = args.row_offset, args.col_offset
    if r0 < 0 or c0 < 0 or r0 + BLOCK > sar_norm.shape[1] or c0 + BLOCK > sar_norm.shape[2]:
        raise SystemExit(
            f"ERROR: requested block [{r0}:{r0+BLOCK}, {c0}:{c0+BLOCK}] falls outside "
            f"the tile's {sar_norm.shape[1]}x{sar_norm.shape[2]} extent."
        )
    x_int8_block = x_int8_full[:, r0:r0 + BLOCK, c0:c0 + BLOCK]  # (3, BLOCK, BLOCK)
    print(f"    Extracted {BLOCK}x{BLOCK} sub-block at row_offset={r0}, col_offset={c0}")
    print(f"    Sub-block INT8 range per channel: "
          f"{[f'[{int(x_int8_block[c].min())}, {int(x_int8_block[c].max())}]' for c in range(3)]}")

    # -- 6. Exact INT32 valid convolution, kernel K's folded INT8 weights --
    print(f"\n[6] Computing exact INT32 raw convolution for kernel {K} over the real sub-block ...")
    raw_conv_int32 = valid_conv_int32(x_int8_block, wk_folded_int8, BLOCK)  # (N_VALID, N_VALID)
    print(f"    raw_conv_int32 ({N_VALID}x{N_VALID}):\n{raw_conv_int32}")

    # -- 7. Fixed-point Q.F constants, using the REAL scale_x --
    print(f"\n[7] Computing Q.{FRAC_BITS} fixed-point constants (REAL scale_x, not 1.0) ...")
    combined_scale = scale_x * scale_w_folded
    SCALE_FX = int(round(combined_scale * (2 ** FRAC_BITS)))
    BIAS_FX = int(round(bk_folded * (2 ** FRAC_BITS)))
    print(f"    combined_scale = scale_x * scale_w_folded = {combined_scale:.10f}")
    print(f"    SCALE_FX (unsigned, Q.{FRAC_BITS}) = {SCALE_FX}")
    print(f"    BIAS_FX  (signed,   Q.{FRAC_BITS}) = {BIAS_FX}")

    product_fx = raw_conv_int32 * SCALE_FX
    biased_fx = product_fx + BIAS_FX
    relu_fx = np.maximum(biased_fx, 0)
    max_abs_biased_fx = int(np.abs(biased_fx).max())
    bits_needed = max_abs_biased_fx.bit_length() + 1
    relu_clamped = bool(np.any(biased_fx <= 0))
    print(f"    biased_fx ({N_VALID}x{N_VALID}):\n{biased_fx}")
    print(f"    relu_fx   ({N_VALID}x{N_VALID}):\n{relu_fx}")
    print(f"    max |biased_fx| = {max_abs_biased_fx} (needs >= {bits_needed} signed bits)")
    print(f"    ReLU clamping exercised on this real patch? {relu_clamped}")

    # -- 7b. Width checks against the existing kernel0/kernel2 VHDL's fixed widths --
    scale_fx_bits_needed = SCALE_FX.bit_length() + 1
    width_warnings = []
    if scale_fx_bits_needed > SCALE_FX_BITS:
        width_warnings.append(
            f"SCALE_FX={SCALE_FX} needs >= {scale_fx_bits_needed} signed bits, exceeds "
            f"the existing {SCALE_FX_BITS}-bit SCALE_FX_SIGNED constant width."
        )
    if bits_needed > OUTPUT_BITS:
        width_warnings.append(
            f"max |biased_fx|={max_abs_biased_fx} needs >= {bits_needed} signed bits, "
            f"exceeds the existing {OUTPUT_BITS}-bit datapath width."
        )
    if width_warnings:
        print("\n    WIDTH WARNING(S):")
        for w in width_warnings:
            print(f"      - {w}")
    else:
        print(f"\n    Width check OK: fits within the existing {SCALE_FX_BITS}-bit SCALE_FX "
              f"and {OUTPUT_BITS}-bit datapath widths, unchanged.")

    # -- 8. Write JSON + CSV vectors --
    positions = []
    idx = 0
    for r in range(N_VALID):
        for c in range(N_VALID):
            positions.append({
                "output_index": idx, "row": r, "col": c,
                "raw_conv_int32": int(raw_conv_int32[r, c]),
                "product_fx": int(product_fx[r, c]),
                "biased_fx": int(biased_fx[r, c]),
                "relu_fx": int(relu_fx[r, c]),
                "relu_float_equiv": float(relu_fx[r, c]) / (2 ** FRAC_BITS),
            })
            idx += 1

    data = {
        "checkpoint_path": str(args.checkpoint.relative_to(REPO_ROOT)),
        "kernel_id": K,
        "tile_name": args.tile_name,
        "uavsar_path": row["uavsar_path"],
        "block_size": BLOCK,
        "row_offset": r0, "col_offset": c0,
        "n_valid_positions": N_POSITIONS,
        "scale_bn": scale_bn, "b_folded": bk_folded, "scale_w_folded": scale_w_folded,
        "w_folded_int8": {f"channel_{c}": wk_folded_int8[c].flatten().tolist() for c in range(3)},
        "scale_x_full_tile": scale_x,
        "frac_bits": FRAC_BITS, "scale_fx": SCALE_FX, "bias_fx": BIAS_FX,
        "relu_clamping_exercised": relu_clamped,
        "width_warnings": width_warnings,
        "positions": positions,
        "notes": [
            "REAL UAVSAR tile input (NOT the synthetic 5x5x3 toy patch used by "
            "every other VHDL testbench in this repo).",
            "scale_x is computed from the FULL normalized 256x256 tile (matching "
            "analyze_first_layer_fixed_point_segmentation_impact.py's convention "
            "exactly), then the INT8 sub-block used here is a direct slice of "
            "that already-quantized full tile -- not independently re-quantized.",
            f"Kernel {K} only, not all 32 output channels -- a real-data "
            "verification SUBSET.",
        ],
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[8] Written: {json_path.relative_to(REPO_ROOT)}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output_index", "row", "col", "raw_conv_int32",
                          "product_fx", "biased_fx", "relu_fx", "relu_float_equiv"])
        for p in positions:
            writer.writerow([p["output_index"], p["row"], p["col"], p["raw_conv_int32"],
                              p["product_fx"], p["biased_fx"], p["relu_fx"],
                              f"{p['relu_float_equiv']:.8f}"])
    print(f"    Written: {csv_path.relative_to(REPO_ROOT)}")

    pos_rows_md = "\n".join(
        f"| {p['output_index']} | {p['row']} | {p['col']} | {p['raw_conv_int32']} | "
        f"{p['product_fx']} | {p['biased_fx']} | **{p['relu_fx']}** | {p['relu_float_equiv']:.6f} |"
        for p in positions
    )
    md_text = f"""# Kernel {K} Folded Conv-BN-ReLU -- REAL-TILE Fixed-Point Test Vectors

Generated by `scripts/generate_real_tile_bn_relu_fixed_point_vectors.py --kernel-id {K}`.

**Real-data verification SUBSET**: kernel {K} only, not all 32 output
channels. Reuses the existing, unmodified
`stream_conv3x3_3chan_kernel{K}_bn_relu_pipelined` architecture (new
package + new entity + new testbench only -- see
`hardware/vhdl_conv3x3/reports/`).

## Source tile

| Field | Value |
|---|---|
| Tile name | `{args.tile_name}` |
| uavsar_path | `{row["uavsar_path"]}` |
| Split CSV | `{args.split_csv.relative_to(REPO_ROOT)}` |
| Sub-block | {BLOCK}x{BLOCK}, offset (row={r0}, col={c0}) |
| Valid 3x3 windows | {N_POSITIONS} ({N_VALID}x{N_VALID}) |

## BatchNorm folding (kernel {K})

```
scale_bn[{K}] = {scale_bn:.6f}
b_folded[{K}] = {bk_folded:.6f}
```

## Quantization

```
scale_w_folded[{K}]   = {scale_w_folded:.8f}   (per-output-channel, folded weights)
scale_x (full tile)   = {scale_x:.8f}          (per-tile, matches
                                                 analyze_first_layer_fixed_point_segmentation_impact.py)
```

INT8 folded weights (row-major):
**Channel 0:** `{wk_folded_int8[0].flatten().tolist()}`
**Channel 1:** `{wk_folded_int8[1].flatten().tolist()}`
**Channel 2:** `{wk_folded_int8[2].flatten().tolist()}`

## Fixed-point format (Q.{FRAC_BITS})

| Constant | Value |
|---|---:|
| `SCALE_FX` | {SCALE_FX} |
| `BIAS_FX` | {BIAS_FX} |

**ReLU clamping exercised on this real patch?** {"Yes" if relu_clamped else "No"}.

**Width check** (existing {SCALE_FX_BITS}-bit SCALE_FX, {OUTPUT_BITS}-bit datapath):
{"".join(f"- {w}" for w in width_warnings) if width_warnings else "OK -- fits unchanged."}

## Expected outputs ({N_POSITIONS} valid positions, row-major)

| idx | row | col | raw_conv_int32 | product_fx | biased_fx | relu_fx (golden) | relu_fx / 2^{FRAC_BITS} |
|---|---|---|---:|---:|---:|---:|---:|
{pos_rows_md}

## Limitations

- Kernel {K} only, real-data verification SUBSET (not all 32 channels).
- Valid (no-padding) convolution only, matching every VHDL prototype in this repo.
- Golden-vector generation only; GHDL results are reported separately under
  `hardware/vhdl_conv3x3/reports/`.
"""
    md_path.write_text(md_text)
    print(f"    Written: {md_path.relative_to(REPO_ROOT)}")

    # -- 9. Write the shared real-tile stimulus package (idempotent: same
    # content every time for the same --tile-name/--row-offset/--col-offset/
    # --block-size, since both kernel0 and kernel2 see the identical patch). --
    print(f"\n[9] Writing shared real-tile stimulus package: {STIMULUS_PKG_PATH.relative_to(REPO_ROOT)} ...")
    ch_flat = [x_int8_block[c].flatten().tolist() for c in range(3)]
    stim_pkg_text = f"""\
-- real_tile_stimulus_pkg.vhd
-- Shared REAL UAVSAR-tile-derived pixel stimulus for the real-tile
-- Conv-BN-ReLU verification testbench(es).
--
-- Generated by: scripts/generate_real_tile_bn_relu_fixed_point_vectors.py
-- This package is regenerated identically regardless of --kernel-id (kernel
-- 0 and kernel 2 both observe the SAME real patch) -- content is
-- deterministic given the same --tile-name/--row-offset/--col-offset/
-- --block-size arguments.
--
-- ---- Source -----------------------------------------------------------
-- Tile        : {args.tile_name} ({row["uavsar_path"]})
-- Split CSV   : {args.split_csv.relative_to(REPO_ROOT)}
-- Sub-block   : {BLOCK}x{BLOCK}, top-left offset (row={r0}, col={c0}) within the
--               full 256x256 tile
-- Normalization: FloodTileDataset._normalize_per_tile (percentile-clip 1-99
--               + zero-mean/unit-variance), scripts/train_unet_baseline_tuned.py
-- Quantization : scale_x = max(|x_full_tile|) / 127, computed from the FULL
--               normalized tile (matches
--               analyze_first_layer_fixed_point_segmentation_impact.py's
--               convention exactly); scale_x = {scale_x:.8f} for this tile.
--               This {BLOCK}x{BLOCK} block is a direct slice of the
--               ALREADY-quantized full-tile INT8 array, not independently
--               re-quantized.
--
-- ---- What this is NOT --------------------------------------------------
-- This is real sensor-derived pixel data (via the model's own real
-- normalization + quantization pipeline), NOT a synthetic toy patch. It is
-- NOT the full 256x256 tile -- only a small, real, contiguous sub-block,
-- chosen so the EXISTING window3x3_stream/stream_conv3x3_3chan_cell
-- streaming interface (unmodified) can be exercised exactly as it already
-- is for the synthetic 5x5 toy patch elsewhere in this repo.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package real_tile_stimulus_pkg is

    constant BLOCK_SIZE : integer := {BLOCK};

    -- Flat, row-major INT8 pixel arrays, one per input channel, length
    -- BLOCK_SIZE*BLOCK_SIZE = {BLOCK * BLOCK}.
    type block_pixels_t is array (0 to {BLOCK * BLOCK - 1}) of integer range -128 to 127;

    constant REAL_TILE_CH0 : block_pixels_t := {vhdl_int_array(ch_flat[0])};
    constant REAL_TILE_CH1 : block_pixels_t := {vhdl_int_array(ch_flat[1])};
    constant REAL_TILE_CH2 : block_pixels_t := {vhdl_int_array(ch_flat[2])};

end package real_tile_stimulus_pkg;
"""
    STIMULUS_PKG_PATH.write_text(stim_pkg_text)
    print(f"    Written: {STIMULUS_PKG_PATH.relative_to(REPO_ROOT)}")

    # -- 10. Write the kernel-specific real-tile VHDL package --
    print(f"\n[10] Writing kernel-{K} real-tile VHDL package: {pkg_path.relative_to(REPO_ROOT)} ...")
    relu_fx_flat = relu_fx.flatten().tolist()
    pkg_name = f"first_conv_bn_relu_kernel{K}_real_tile{SUFFIX}_pkg"
    expected_rows = ",\n        ".join(
        ", ".join(str(v) for v in relu_fx_flat[i:i + 4])
        for i in range(0, len(relu_fx_flat), 4)
    )
    pkg_text = f"""\
-- {pkg_name}.vhd
-- VHDL-2008 package of INT8 folded weights, Q.{FRAC_BITS} fixed-point
-- SCALE_FX/BIAS_FX constants, and Q.{FRAC_BITS} fixed-point golden output
-- constants for the kernel-{K} folded Conv-BN-ReLU design, evaluated on a
-- REAL UAVSAR-tile-derived patch (see real_tile_stimulus_pkg.vhd for the
-- pixel data), NOT the synthetic 5x5x3 toy patch.
--
-- Generated by: scripts/generate_real_tile_bn_relu_fixed_point_vectors.py --kernel-id {K}
-- Source data : hardware/vhdl_conv3x3/real_tile_vectors/kernel{K}/kernel{K}_real_tile_bn_relu_fixed_point_vectors.json
--
-- ---- Source of weights --------------------------------------------------
-- Checkpoint : {args.checkpoint.relative_to(REPO_ROOT)}
-- Tensor key : {CONV_KEY_EXPECTED} (kernel {K} only), folded with
--              {BN_PREFIX_EXPECTED} (BatchNorm2d, eval-mode running stats)
--
-- ---- BatchNorm folding ----------------------------------------------------
-- scale_bn[{K}] = {scale_bn:.6f}
-- b_folded[{K}] = {bk_folded:.6f}
-- (eps = {BN_EPS}, torch.nn.BatchNorm2d default)
--
-- ---- Folded weight quantization ------------------------------------------
-- scale_w_folded[{K}] = {scale_w_folded:.8f}
-- symmetric per-output-channel INT8, SAME convention as
-- scripts/analyze_first_layer_fixed_point_segmentation_impact.py.
--
-- ---- REAL-TILE activation quantization (NOT scale_x=1.0) -------------------
-- scale_x = {scale_x:.8f}, computed from the FULL normalized 256x256 tile
-- '{args.tile_name}', matching
-- analyze_first_layer_fixed_point_segmentation_impact.py's per-tile
-- convention exactly. This is the KEY difference from every toy-patch
-- kernel package in this repo, where scale_x = 1.0 was exact by
-- construction.
--
-- ---- Fixed-point format (Q.{FRAC_BITS}, signed, {FRAC_BITS} fractional bits) --------------
-- real_value ~= fixed_value / 2^{FRAC_BITS}
-- SCALE_FX = round(scale_x * scale_w_folded[{K}] * 2^{FRAC_BITS})
-- BIAS_FX  = round(b_folded[{K}] * 2^{FRAC_BITS})
-- Datapath: product_fx = raw_conv_int32 * SCALE_FX (already Q.{FRAC_BITS}, no shift);
--           biased_fx = product_fx + BIAS_FX (same format, no shift);
--           relu_fx = biased_fx if biased_fx > 0 else 0.
--
-- ---- What this validates --------------------------------------------------
-- This package supplies the folded INT8 weights (IDENTICAL to
-- first_conv_bn_relu_kernel{K}_pkg.vhd's, since weights do not depend on
-- input), the REAL-TILE fixed-point scale/bias constants, and the exact
-- Q.{FRAC_BITS} fixed-point expected outputs for a REAL {BLOCK}x{BLOCK} UAVSAR
-- patch, so a testbench can confirm the EXISTING, UNMODIFIED
-- stream_conv3x3_3chan_kernel{K}_bn_relu_pipelined datapath produces the SAME
-- fixed-point integer values as this Python script on REAL sensor-derived
-- data. It does NOT validate the other 31 output channels, does NOT
-- implement padding, and does NOT claim board-tested or measured
-- performance.

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package {pkg_name} is

    -- 9 INT8 weights per input channel, row-major (w0=top-left .. w8=bottom-right)
    -- IDENTICAL to first_conv_bn_relu_kernel{K}_pkg's K{K}_FOLDED_CH*_W --
    -- weights do not depend on the input patch.
    type int8_kernel_t is array (0 to 8) of integer range -128 to 127;

    constant K{K}_RT_FOLDED_CH0_W : int8_kernel_t := {vhdl_int_array(wk_folded_int8[0].flatten())};
    constant K{K}_RT_FOLDED_CH1_W : int8_kernel_t := {vhdl_int_array(wk_folded_int8[1].flatten())};
    constant K{K}_RT_FOLDED_CH2_W : int8_kernel_t := {vhdl_int_array(wk_folded_int8[2].flatten())};

    -- Fixed-point Q.{FRAC_BITS} constants -- REAL scale_x baked in (differs
    -- from the toy-patch package's SCALE_FX, which assumed scale_x=1.0).
    constant FRAC_BITS : integer := {FRAC_BITS};
    constant SCALE_FX  : integer := {SCALE_FX};   -- unsigned, Q.{FRAC_BITS}
    constant BIAS_FX   : integer := {BIAS_FX};   -- signed,   Q.{FRAC_BITS}

    -- {N_POSITIONS} Q.{FRAC_BITS} fixed-point expected outputs (row-major,
    -- output_index 0..{N_POSITIONS - 1}), EXACT integers -- this is what the
    -- VHDL testbench compares against.
    type fx_outputs_t is array (0 to {N_POSITIONS - 1}) of integer;
    constant K{K}_RT_BN_RELU_EXPECTED_FX : fx_outputs_t := (
        {expected_rows}
    );

end package {pkg_name};
"""
    pkg_path.write_text(pkg_text)
    print(f"    Written: {pkg_path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Kernel {K} (REAL TILE '{args.tile_name}', block {BLOCK}x{BLOCK} @ "
          f"row={r0},col={c0}):")
    print(f"  scale_bn={scale_bn:.6f} b_folded={bk_folded:.6f} scale_w_folded={scale_w_folded:.8f}")
    print(f"  scale_x(full tile)={scale_x:.8f}")
    print(f"  SCALE_FX={SCALE_FX}  BIAS_FX={BIAS_FX}  (Q.{FRAC_BITS})")
    print(f"  ReLU clamping exercised: {relu_clamped}")
    print(f"  Width warnings: {width_warnings if width_warnings else 'none'}")
    print(f"  Expected relu_fx ({N_POSITIONS} values, row-major): {relu_fx_flat}")


if __name__ == "__main__":
    main()
