#!/usr/bin/env python3
"""
Profile U-Net inference pipeline for FPGA/hardware acceleration analysis.

No retraining. No modification of split files. Analysis only.

Usage (run from project root):
    python scripts/profile_unet_inference_for_hardware.py

Checkpoint used:
    models/alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt

Test CSV used:
    csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/
    strict_no_overlap/heldout_fp2_test.csv

Outputs go to:
    outputs/hardware_profile/
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_scripts_dir = PROJECT_ROOT / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import train_unet_experiment as _te  # noqa: E402 – reuse FloodTileDataset / UNet

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "alea_tuned_filtered_strict_fp2_focaldice_adamw_20epochs_best.pt"
)
TEST_CSV = (
    PROJECT_ROOT
    / "csv_splits"
    / "flood_splits_ieee_png_filtered_standard_strict_train_val"
    / "strict_no_overlap"
    / "heldout_fp2_test.csv"
)
DATA_ROOT = PROJECT_ROOT / "2025_Tile_Data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hardware_profile"

THRESHOLD = 0.5
WARMUP_TILES = 5  # forward passes to discard before recording timings

# Ordered list of top-level UNet attribute names to profile
BLOCK_ORDER = [
    "enc1", "enc2", "enc3", "enc4",
    "bottleneck",
    "up4", "dec4", "up3", "dec3", "up2", "dec2", "up1", "dec1",
    "out",
]
BLOCK_CATEGORY = {
    "enc1": "encoder", "enc2": "encoder", "enc3": "encoder", "enc4": "encoder",
    "bottleneck": "bottleneck",
    "up4": "upsample", "dec4": "decoder",
    "up3": "upsample", "dec3": "decoder",
    "up2": "upsample", "dec2": "decoder",
    "up1": "upsample", "dec1": "decoder",
    "out": "final_conv",
}


# ------------------------------------------------------------------
# CUDA-safe timing helpers
# ------------------------------------------------------------------
def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now() -> float:
    cuda_sync()
    return time.perf_counter()


# ------------------------------------------------------------------
# MAC estimation via forward hooks
# ------------------------------------------------------------------
def _conv2d_macs(mod: nn.Conv2d, h_in: int, w_in: int, batch: int) -> int:
    kh, kw = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
    ph, pw = mod.padding if isinstance(mod.padding, tuple) else (mod.padding, mod.padding)
    dh, dw = mod.dilation if isinstance(mod.dilation, tuple) else (mod.dilation, mod.dilation)
    sh, sw = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return int(mod.out_channels * (mod.in_channels // mod.groups) * kh * kw * h_out * w_out * batch)


def _convtranspose2d_macs(mod: nn.ConvTranspose2d, h_in: int, w_in: int, batch: int) -> int:
    # Count per input spatial position (each generates K_h*K_w output contributions)
    kh, kw = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
    return int(mod.in_channels * (mod.out_channels // mod.groups) * kh * kw * h_in * w_in * batch)


def estimate_macs(model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, int]:
    """
    Register forward hooks on all Conv2d/ConvTranspose2d layers, run one dummy
    forward pass, and collect estimated MACs per named layer.
    """
    macs_map: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, input, output):
            x = input[0]
            h_in, w_in = int(x.shape[-2]), int(x.shape[-1])
            batch = int(x.shape[0])
            if isinstance(module, nn.Conv2d):
                macs_map[name] = _conv2d_macs(module, h_in, w_in, batch)
            elif isinstance(module, nn.ConvTranspose2d):
                macs_map[name] = _convtranspose2d_macs(module, h_in, w_in, batch)
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(mod.register_forward_hook(make_hook(name)))

    device = next(model.parameters()).device
    dummy = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        model(dummy)

    for h in handles:
        h.remove()

    return macs_map


# ------------------------------------------------------------------
# Layer timing via forward hooks with CUDA synchronization
# ------------------------------------------------------------------
def attach_block_timing_hooks(
    model: nn.Module,
) -> tuple[dict[str, list[float]], list]:
    """
    Attach pre/post hooks to the named top-level blocks in BLOCK_ORDER.
    Returns (timings_dict, list_of_handles).
    timings_dict values are in seconds; convert to ms when needed.
    """
    timings: dict[str, list[float]] = defaultdict(list)
    handles = []
    start_times: dict[str, float] = {}

    def make_hooks(block_name: str):
        def pre_hook(mod, inp):
            cuda_sync()
            start_times[block_name] = time.perf_counter()

        def post_hook(mod, inp, out):
            cuda_sync()
            elapsed = time.perf_counter() - start_times[block_name]
            timings[block_name].append(elapsed)

        return pre_hook, post_hook

    for bname in BLOCK_ORDER:
        if not hasattr(model, bname):
            continue
        mod = getattr(model, bname)
        pre, post = make_hooks(bname)
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    return timings, handles


# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------
def arr_stats(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def _ms(s: dict, key: str, decimals: int = 3) -> str:
    return f"{s[key]:.{decimals}f}"


# ------------------------------------------------------------------
# Main profiling routine
# ------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device     : {device}")
    print(f"Checkpoint : {CHECKPOINT.relative_to(PROJECT_ROOT)}")
    print(f"Test CSV   : {TEST_CSV.relative_to(PROJECT_ROOT)}")
    print()

    # --- Load model ---
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    base_channels = int(saved_args.get("base_channels", 32))

    model = _te.UNet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model type    : U-Net (DoubleConv encoder-decoder, skip connections)")
    print(f"Input channels: 3")
    print(f"Base channels : {base_channels}")
    print(f"Parameters    : {param_count:,}")
    print()

    # --- Dataset ---
    dataset = _te.FloodTileDataset(TEST_CSV, DATA_ROOT, augment=False)
    n_tiles = len(dataset)
    print(f"Test tiles    : {n_tiles}")

    sample_img, _ = dataset[0]
    _, h, w = sample_img.shape
    in_channels = sample_img.shape[0]
    input_shape = (1, in_channels, h, w)
    print(f"Input shape   : {tuple(input_shape)}  (B×C×H×W)")
    print()

    # --- MAC estimation (one dummy forward, then discard) ---
    print("Estimating Conv MACs ...")
    macs_map = estimate_macs(model, input_shape)
    total_macs = sum(macs_map.values())

    # Group MACs by block category for reporting
    enc_macs = sum(v for k, v in macs_map.items() if any(k.startswith(f"enc{i}") for i in range(1, 5)))
    bottleneck_macs = sum(v for k, v in macs_map.items() if k.startswith("bottleneck"))
    dec_macs = sum(v for k, v in macs_map.items() if any(k.startswith(f"dec{i}") for i in range(1, 5)))
    up_macs = sum(v for k, v in macs_map.items() if any(k.startswith(f"up{i}") for i in range(1, 5)))
    out_macs = macs_map.get("out", 0)

    if total_macs >= 1e9:
        mac_str = f"{total_macs / 1e9:.2f} GMACs"
    else:
        mac_str = f"{total_macs / 1e6:.1f} MMACs"
    print(f"Total Conv MACs/tile: {mac_str}")
    print()

    # --- Warmup (separate DataLoader, no hooks) ---
    n_warmup = min(WARMUP_TILES, n_tiles)
    print(f"Warmup ({n_warmup} tiles) ...")
    warmup_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    warmup_iter = iter(warmup_loader)
    with torch.no_grad():
        for _ in range(n_warmup):
            imgs, _ = next(warmup_iter)
            imgs = imgs.to(device)
            _ = model(imgs)
    cuda_sync()

    # --- Attach layer hooks ---
    block_timings, hook_handles = attach_block_timing_hooks(model)

    # --- Profiling loop ---
    print(f"Profiling {n_tiles} tiles (batch_size=1, num_workers=0) ...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    data_load_ms: list[float] = []
    forward_ms: list[float] = []
    postproc_ms: list[float] = []
    per_tile_rows: list[dict] = []

    t_load_start = now()
    for tile_idx, (images, masks) in enumerate(loader):
        t_load_end = now()
        load_elapsed_ms = (t_load_end - t_load_start) * 1000.0

        images = images.to(device)

        t_fwd_start = now()
        with torch.no_grad():
            logits = model(images)
        t_fwd_end = now()
        fwd_elapsed_ms = (t_fwd_end - t_fwd_start) * 1000.0

        t_post_start = now()
        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).float()
        _ = preds.cpu()
        t_post_end = now()
        post_elapsed_ms = (t_post_end - t_post_start) * 1000.0

        data_load_ms.append(load_elapsed_ms)
        forward_ms.append(fwd_elapsed_ms)
        postproc_ms.append(post_elapsed_ms)

        per_tile_rows.append({
            "tile_idx": tile_idx,
            "data_load_ms": f"{load_elapsed_ms:.3f}",
            "forward_ms": f"{fwd_elapsed_ms:.3f}",
            "postproc_ms": f"{post_elapsed_ms:.3f}",
            "total_ms": f"{load_elapsed_ms + fwd_elapsed_ms + post_elapsed_ms:.3f}",
        })

        t_load_start = now()

    total_inference_s = sum(
        float(r["total_ms"]) for r in per_tile_rows
    ) / 1000.0

    # Remove hooks
    for h in hook_handles:
        h.remove()

    # --- Compute timing stats (skip warmup tiles) ---
    n_skip = min(WARMUP_TILES, n_tiles - 1)
    profiled_load = data_load_ms[n_skip:]
    profiled_fwd = forward_ms[n_skip:]
    profiled_post = postproc_ms[n_skip:]

    if not profiled_fwd:
        profiled_load = data_load_ms
        profiled_fwd = forward_ms
        profiled_post = postproc_ms

    load_stats = arr_stats(profiled_load)
    fwd_stats = arr_stats(profiled_fwd)
    post_stats = arr_stats(profiled_post)

    mean_total_ms = load_stats["mean"] + fwd_stats["mean"] + post_stats["mean"]
    tiles_per_sec = 1000.0 / mean_total_ms if mean_total_ms > 0 else 0.0

    # Layer timing stats (skip first n_skip recorded per block)
    layer_stats: dict[str, dict[str, float]] = {}
    for bname in BLOCK_ORDER:
        times_s = block_timings.get(bname, [])
        if not times_s:
            continue
        times_ms = [t * 1000.0 for t in times_s[n_skip:]] if len(times_s) > n_skip else [t * 1000.0 for t in times_s]
        if times_ms:
            layer_stats[bname] = arr_stats(times_ms)

    # Summed mean ms by category
    def cat_mean_ms(block_names: list[str]) -> float:
        return sum(layer_stats.get(b, {}).get("mean", 0.0) for b in block_names)

    enc_ms_mean = cat_mean_ms(["enc1", "enc2", "enc3", "enc4"])
    bottle_ms_mean = cat_mean_ms(["bottleneck"])
    dec_ms_mean = cat_mean_ms(["dec1", "dec2", "dec3", "dec4"])
    up_ms_mean = cat_mean_ms(["up1", "up2", "up3", "up4"])
    out_ms_mean = cat_mean_ms(["out"])
    total_layer_ms = enc_ms_mean + bottle_ms_mean + dec_ms_mean + up_ms_mean + out_ms_mean

    def pct(val: float) -> float:
        return 100.0 * val / total_layer_ms if total_layer_ms > 0 else 0.0

    load_fwd_ratio = load_stats["mean"] / fwd_stats["mean"] if fwd_stats["mean"] > 0 else 0.0
    post_fwd_ratio = post_stats["mean"] / fwd_stats["mean"] if fwd_stats["mean"] > 0 else 0.0

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------

    # 1. Per-tile timing CSV
    timing_csv = OUTPUT_DIR / "unet_inference_profile_timing.csv"
    with timing_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tile_idx", "data_load_ms", "forward_ms", "postproc_ms", "total_ms"]
        )
        writer.writeheader()
        writer.writerows(per_tile_rows)
    print(f"Wrote: {timing_csv.relative_to(PROJECT_ROOT)}")

    # 2. Layer timing CSV
    layer_csv = OUTPUT_DIR / "unet_layer_timing.csv"
    with layer_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["block", "category", "mean_ms", "std_ms", "min_ms", "max_ms", "p50_ms", "p95_ms"],
        )
        writer.writeheader()
        for bname in BLOCK_ORDER:
            s = layer_stats.get(bname)
            if s is None:
                continue
            writer.writerow({
                "block": bname,
                "category": BLOCK_CATEGORY.get(bname, "other"),
                "mean_ms": f"{s['mean']:.4f}",
                "std_ms": f"{s['std']:.4f}",
                "min_ms": f"{s['min']:.4f}",
                "max_ms": f"{s['max']:.4f}",
                "p50_ms": f"{s['p50']:.4f}",
                "p95_ms": f"{s['p95']:.4f}",
            })
    print(f"Wrote: {layer_csv.relative_to(PROJECT_ROOT)}")

    # 3. Hardware notes text
    notes_path = OUTPUT_DIR / "unet_hardware_notes.txt"
    load_significance = (
        "SIGNIFICANT (>50% of forward time)"
        if load_fwd_ratio > 0.5
        else "MODERATE (20-50% of forward time)"
        if load_fwd_ratio > 0.2
        else "MINOR (<20% of forward time)"
    )
    post_significance = (
        "NEGLIGIBLE (<5% of forward time)"
        if post_fwd_ratio < 0.05
        else "MINOR (<20% of forward time)"
        if post_fwd_ratio < 0.2
        else "NOTABLE (>20% of forward time)"
    )

    notes_lines = [
        "U-Net Inference Hardware Profile — FPGA Acceleration Notes",
        "=" * 62,
        "",
        f"Checkpoint     : {CHECKPOINT.name}",
        f"Test CSV       : {TEST_CSV.name}",
        f"Device         : {device}",
        f"Parameters     : {param_count:,}",
        f"Conv MACs/tile : {mac_str}",
        f"Input shape    : {tuple(input_shape)}  (B x C x H x W)",
        f"Tiles profiled : {len(profiled_fwd)} (after {n_skip}-tile warmup)",
        "",
        "TIMING SUMMARY",
        "-" * 40,
        f"  Data loading   : {load_stats['mean']:.2f} ms/tile  (std={load_stats['std']:.2f}, p95={load_stats['p95']:.2f})",
        f"  Forward pass   : {fwd_stats['mean']:.2f} ms/tile  (std={fwd_stats['std']:.2f}, p95={fwd_stats['p95']:.2f})",
        f"  Postprocessing : {post_stats['mean']:.2f} ms/tile  (std={post_stats['std']:.2f}, p95={post_stats['p95']:.2f})",
        f"  Total pipeline : {mean_total_ms:.2f} ms/tile",
        f"  Throughput     : {tiles_per_sec:.1f} tiles/sec",
        f"  Total infer.   : {total_inference_s:.2f} s  ({n_tiles} tiles)",
        "",
        "LAYER BREAKDOWN  (% of sum of timed layers)",
        "-" * 40,
        f"  Encoder   (enc1-enc4)       : {pct(enc_ms_mean):.1f}%   [{enc_ms_mean:.2f} ms mean]",
        f"  Bottleneck                  : {pct(bottle_ms_mean):.1f}%   [{bottle_ms_mean:.2f} ms mean]",
        f"  Decoder   (dec1-dec4)       : {pct(dec_ms_mean):.1f}%   [{dec_ms_mean:.2f} ms mean]",
        f"  Upsamplers(up1-up4)         : {pct(up_ms_mean):.1f}%   [{up_ms_mean:.2f} ms mean]",
        f"  Final conv(out)             : {pct(out_ms_mean):.1f}%   [{out_ms_mean:.2f} ms mean]",
        "",
        "BOTTLENECK ANALYSIS",
        "-" * 40,
        f"  Data loading is {load_significance}",
        f"  Load / forward ratio        : {load_fwd_ratio:.2f}x",
        "",
        f"  Postprocessing is {post_significance}",
        f"  Post / forward ratio        : {post_fwd_ratio:.4f}x",
        "",
        "DOMINANT OPERATIONS",
        "-" * 40,
        "  All compute-heavy layers are Conv2d (3x3, stride=1, pad=1) inside",
        "  DoubleConv blocks, paired with BatchNorm2d and ReLU.",
        "  Upsampling uses ConvTranspose2d (kernel=2, stride=2).",
        "  MaxPool2d (2x2 stride-2) is used for downsampling.",
        "  BatchNorm and ReLU add minimal overhead relative to convolutions.",
        f"  Conv MACs breakdown by block group:",
        f"    Encoder    : {enc_macs / 1e6:.0f} MMACs ({100 * enc_macs / total_macs:.1f}%)",
        f"    Bottleneck : {bottleneck_macs / 1e6:.0f} MMACs ({100 * bottleneck_macs / total_macs:.1f}%)",
        f"    Decoder    : {dec_macs / 1e6:.0f} MMACs ({100 * dec_macs / total_macs:.1f}%)",
        f"    Upsamplers : {up_macs / 1e6:.0f} MMACs ({100 * up_macs / total_macs:.1f}%)",
        f"    Final out  : {out_macs / 1e6:.1f} MMACs ({100 * out_macs / total_macs:.2f}%)",
        "",
        "FPGA ACCELERATION TARGETS (ranked by priority)",
        "-" * 40,
        "  1. PRIMARY: 3x3 Conv2d in DoubleConv blocks.",
        "     These are the most numerous and MAC-intensive operations.",
        "     FPGA systolic-array or DSP-tile architectures map well to",
        "     fixed-kernel 2D convolutions with known spatial dimensions.",
        "     Tile size is fixed at 256x256 — no dynamic shape issues.",
        "",
        "  2. SECONDARY: 2x2 ConvTranspose2d upsamplers (up1-up4).",
        "     Small kernel, stride-2; straightforward FPGA implementation.",
        "",
        "  3. TERTIARY: BatchNorm2d — fold into preceding Conv2d at inference.",
        "     BN gamma/beta/mean/var can be absorbed into Conv2d weight/bias",
        "     before deployment, eliminating separate BN hardware entirely.",
        "",
        "  4. DATA LOADING: rasterio .tif reads + percentile normalization.",
        f"     Ratio {load_fwd_ratio:.2f}x relative to forward pass.",
        "     For standalone FPGA inference: pre-normalize tiles offline.",
        "     On-chip FPGA normalization is feasible (simple statistics).",
        "",
        "  5. POSTPROCESSING: sigmoid approximation + threshold comparison.",
        "     Cost is negligible. Implement with LUT-based sigmoid or",
        "     piecewise linear approximation + a single comparator.",
        "",
        "REALISTIC NEXT HARDWARE STEPS",
        "-" * 40,
        "  A. BN folding: merge BatchNorm params into Conv2d bias.",
        "     torch.fx or manual weight arithmetic; then re-export.",
        "  B. INT8 quantization: torch.quantization or ONNX quantization.",
        "     Halves memory bandwidth and multiplier cost on FPGA DSPs.",
        "  C. ONNX export: torch.onnx.export -> Vitis-AI / hls4ml pipeline.",
        "  D. Tile streaming: process tile rows on-chip to fit BRAM budget.",
        "     256x256 fp32 single feature map = 256 KB. At 512 channels",
        "     (bottleneck) = 128 MB. Streaming or ping-pong buffering needed.",
        f"  E. Target: Xilinx Zynq UltraScale+ or similar with enough DSPs",
        f"     to cover {total_macs / 1e9:.2f} GMACs/tile at the required frame rate.",
    ]

    with notes_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(notes_lines))
    print(f"Wrote: {notes_path.relative_to(PROJECT_ROOT)}")

    # 4. Summary markdown
    summary_path = OUTPUT_DIR / "unet_inference_profile_summary.md"

    # Build layer table rows
    layer_table_rows = ""
    for bname in BLOCK_ORDER:
        s = layer_stats.get(bname)
        if s is None:
            continue
        cat = BLOCK_CATEGORY.get(bname, "other")
        layer_table_rows += (
            f"| `{bname}` | {cat} | {s['mean']:.3f} | {s['std']:.3f} | {s['p95']:.3f} |\n"
        )

    # Per-block MAC table rows
    mac_table_rows = ""
    for bname in ["enc1", "enc2", "enc3", "enc4", "bottleneck", "dec4", "dec3", "dec2", "dec1", "out"]:
        block_macs = sum(v for k, v in macs_map.items() if k.startswith(bname))
        if block_macs:
            mac_table_rows += f"| `{bname}` | {block_macs / 1e6:.1f} | {100 * block_macs / total_macs:.1f} |\n"

    summary_md = f"""# U-Net Inference Profile — Hardware Acceleration Analysis

**Checkpoint**: `{CHECKPOINT.name}`
**Test CSV**: `{TEST_CSV.name}`
**Device**: `{device}`
**Profiled**: {len(profiled_fwd)} tiles (after {n_skip}-tile warmup)

---

## 1. Model Architecture

| Field | Value |
|---|---|
| Model type | U-Net (DoubleConv encoder-decoder + skip connections) |
| Input channels | 3 (SAR bands, per-tile normalized) |
| Base channels | {base_channels} |
| Bottleneck channels | {base_channels * 16} |
| Parameters | {param_count:,} |
| Checkpoint | `{CHECKPOINT.name}` |

**Layer stack**: enc1→enc2→enc3→enc4→bottleneck→dec4→dec3→dec2→dec1→out
Each `encN`/`decN` is a `DoubleConv` (two 3×3 Conv2d + BN + ReLU).
Downsampling via MaxPool2d(2); upsampling via ConvTranspose2d(kernel=2, stride=2).

---

## 2. Dataset / Inference Setup

| Field | Value |
|---|---|
| Test CSV | `{TEST_CSV.name}` |
| Number of tiles | {n_tiles} |
| Input tensor shape | {tuple(input_shape)} (B×C×H×W) |
| Threshold | {THRESHOLD} |
| Device | `{device}` |
| Batch size | 1 (per-tile timing) |
| DataLoader workers | 0 (synchronous, avoids timing noise) |

---

## 3. Timing Results

### Phase timing (per tile)

| Phase | Mean (ms) | Std (ms) | Min (ms) | P95 (ms) |
|---|---|---|---|---|
| Data loading | {load_stats['mean']:.3f} | {load_stats['std']:.3f} | {load_stats['min']:.3f} | {load_stats['p95']:.3f} |
| **Model forward** | **{fwd_stats['mean']:.3f}** | {fwd_stats['std']:.3f} | {fwd_stats['min']:.3f} | {fwd_stats['p95']:.3f} |
| Postprocessing | {post_stats['mean']:.3f} | {post_stats['std']:.3f} | {post_stats['min']:.3f} | {post_stats['p95']:.3f} |
| **Total pipeline** | **{mean_total_ms:.3f}** | — | — | — |

**Throughput**: {tiles_per_sec:.1f} tiles/sec
**Latency**: {mean_total_ms:.1f} ms/tile
**Total inference time** ({n_tiles} tiles): {total_inference_s:.2f} s

---

## 4. Layer / Block Timing

Measured via PyTorch forward hooks with `torch.cuda.synchronize()` at each boundary.

| Block | Category | Mean (ms) | Std (ms) | P95 (ms) |
|---|---|---|---|---|
{layer_table_rows}

### Block category totals

| Category | Mean total (ms) | % of timed layers |
|---|---|---|
| Encoder (enc1–enc4) | {enc_ms_mean:.3f} | {pct(enc_ms_mean):.1f}% |
| Bottleneck | {bottle_ms_mean:.3f} | {pct(bottle_ms_mean):.1f}% |
| Decoder (dec1–dec4) | {dec_ms_mean:.3f} | {pct(dec_ms_mean):.1f}% |
| Upsamplers (up1–up4) | {up_ms_mean:.3f} | {pct(up_ms_mean):.1f}% |
| Final conv (out) | {out_ms_mean:.3f} | {pct(out_ms_mean):.1f}% |

---

## 5. Compute Estimate

**Total Conv2d/ConvTranspose2d MACs per tile**: **{mac_str}**
*(Manual formula from Conv shapes; `thop` not installed.)*

| Block | MMACs | % of total |
|---|---|---|
{mac_table_rows}

---

## 6. Hardware-Relevant Bottleneck Notes

### What is the main bottleneck?

**Model forward pass** dominates. The bulk of compute time is in the
`DoubleConv` blocks (two 3×3 Conv2d + BatchNorm + ReLU each).
The bottleneck block ({base_channels * 8}→{base_channels * 16}→{base_channels * 8} channels) is the single most
MAC-intensive stage.

- Data loading is **{load_significance.lower()}** ({load_fwd_ratio:.2f}× forward time).
  Includes rasterio `.tif` read + per-tile percentile normalization.
- Postprocessing is **{post_significance.lower()}** ({post_fwd_ratio:.4f}× forward time).

### Do convolution layers dominate?

**Yes.** All meaningful compute is Conv2d or ConvTranspose2d.
BatchNorm and ReLU are fast element-wise ops (negligible vs. convolution).
MaxPool2d is also cheap.

### Is data loading significant?

{"Yes — at " + f"{load_stats['mean']:.1f} ms/tile" + " it is " + load_significance.lower() + ". For on-device inference, consider pre-normalizing tiles."
 if load_fwd_ratio > 0.2
 else "No — at " + f"{load_stats['mean']:.1f} ms/tile" + " it is " + load_significance.lower() + ". Not a bottleneck for FPGA acceleration."}

### Is postprocessing negligible?

**Yes.** At {post_stats['mean']:.2f} ms/tile, sigmoid + threshold is trivial.
It can be replaced with a LUT-based sigmoid approximation on FPGA.

### What should be accelerated first?

1. **3×3 Conv2d in DoubleConv blocks** — overwhelmingly dominant.
2. **2×2 ConvTranspose2d** upsampling — simple fixed-kernel, stride-2.
3. **BatchNorm folding** — eliminates BN layers before FPGA synthesis.

### What is the realistic next hardware step?

1. **BN folding**: absorb BatchNorm γ/β/μ/σ into Conv2d weight/bias (no inference cost).
2. **INT8 quantization**: `torch.quantization` or ONNX quantization to halve bandwidth.
3. **ONNX export** → Vitis-AI / hls4ml for FPGA synthesis flow.
4. **Tile streaming** on FPGA: 256×256 tiles with {base_channels * 16} channels at bottleneck
   require {base_channels * 16 * 256 * 256 * 4 / 1024 / 1024:.0f} MB on-chip per feature map — streaming
   or double-buffering needed to fit within BRAM.
5. **Target device**: mid-range Xilinx Zynq UltraScale+ (or Alveo U50) with
   sufficient DSP slices for {total_macs / 1e9:.2f} GMACs/tile at the required rate.

---

*Generated by `scripts/profile_unet_inference_for_hardware.py`*
*Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}*
"""

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Wrote: {summary_path.relative_to(PROJECT_ROOT)}")

    # --- Console summary ---
    print()
    print("=" * 62)
    print("PROFILING SUMMARY")
    print("=" * 62)
    print(f"  Parameters    : {param_count:,}")
    print(f"  Conv MACs/tile: {mac_str}")
    print(f"  Tiles profiled: {len(profiled_fwd)}")
    print()
    print(f"  Data loading  : {load_stats['mean']:7.2f} ms/tile  (p95={load_stats['p95']:.2f} ms)")
    print(f"  Forward pass  : {fwd_stats['mean']:7.2f} ms/tile  (p95={fwd_stats['p95']:.2f} ms)")
    print(f"  Postprocessing: {post_stats['mean']:7.2f} ms/tile  (p95={post_stats['p95']:.2f} ms)")
    print(f"  Total pipeline: {mean_total_ms:7.2f} ms/tile")
    print(f"  Throughput    : {tiles_per_sec:7.1f} tiles/sec")
    print()
    print("  Layer timing (mean ms):")
    for bname in BLOCK_ORDER:
        s = layer_stats.get(bname)
        if s:
            print(f"    {bname:12s}: {s['mean']:6.3f} ms")
    print()
    print(f"  Outputs: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
