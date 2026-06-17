#!/usr/bin/env python3
"""Train an experimental SAR-only U-Net with selectable architecture.

This extends train_unet_experiment.py with an --arch choice between the
plain UNet and an Attention U-Net (attention gates on the skip connections).
Dataset format, CSV columns, default data root, loss options, checkpoint
selection, augmentation, and output naming are otherwise unchanged, so this
script remains compatible with both the standard strict splits and the
strict PNG-filtered splits.

Example:
    python scripts/train_unet_arch_experiment.py \
        --train-csv csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap/heldout_fp1_train.csv \
        --val-csv csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap/heldout_fp1_validation.csv \
        --test-csv csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap/heldout_fp1_test.csv \
        --arch attention_unet --loss bce_dice --save-best-by val_dice --augment
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np


def print_basic_help() -> None:
    """Show help even before optional training packages are installed."""
    print(
        """usage: train_unet_arch_experiment.py [options]

Train an experimental SAR-only binary U-Net with selectable architecture,
loss, checkpoint-selection metric, and optional augmentation.

options:
  --data-root PATH        Root folder for tile paths (default: 2025_Tile_Data)
  --train-csv PATH        Training split CSV
  --val-csv PATH          Validation split CSV
  --test-csv PATH         Test split CSV
  --epochs N              Number of epochs (default: 20)
  --batch-size N          Batch size (default: 8)
  --lr LR                 Adam learning rate (default: 1e-3)
  --num-workers N         DataLoader workers (default: 2)
  --base-channels N       U-Net width (default: 32)
  --seed N                Random seed (default: 42)
  --models-dir PATH       Checkpoint output folder (default: models)
  --results-dir PATH      Metrics CSV output folder (default: results)
  --run-name NAME         Exact prefix for output files
  --device DEVICE         cuda or cpu
  --arch {unet,attention_unet,unet_plus_plus}
                          Model architecture (default: unet)
  --loss {bce,dice,bce_dice}
                          Training loss (default: bce)
  --save-best-by {val_loss,val_dice}
                          Checkpoint-selection metric (default: val_dice)
  --augment               Random flips + 90-degree rotations on train split only

Outputs:
  models/{run-name}_best.pt
  results/{run-name}_metrics.csv

Default split:
  strict_no_overlap/heldout_fp1_train.csv
  strict_no_overlap/heldout_fp1_validation.csv
  strict_no_overlap/heldout_fp1_test.csv
"""
    )


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print_basic_help()
    raise SystemExit(0)


def require_package(package_name: str) -> None:
    if importlib.util.find_spec(package_name) is None:
        raise SystemExit(
            f"Missing required package: {package_name}\n"
            f"Install the project environment packages, then rerun this script. "
            f"For example, check with: python3 -c \"import {package_name}\""
        )


require_package("torch")
require_package("rasterio")

import rasterio  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


class FloodTileDataset(Dataset):
    """Reads SAR image tiles and binary flood/change masks from split CSV rows.

    Same CSV contract as train_unet_baseline.FloodTileDataset (requires
    "uavsar_path" and "flood_mask_path" columns; extra columns such as the
    PNG-filtered split's "in_ieee_*" flags are ignored). Optionally applies
    paired train-time augmentation.
    """

    def __init__(self, csv_path: Path, data_root: Path, augment: bool = False) -> None:
        self.csv_path = csv_path
        self.data_root = data_root
        self.augment = augment
        self.rows = self._read_rows(csv_path)

    @staticmethod
    def _read_rows(csv_path: Path) -> list[dict[str, str]]:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"uavsar_path", "flood_mask_path"}
            missing = required_columns - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
            return list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        sar_path = self.data_root / row["uavsar_path"]
        mask_path = self.data_root / row["flood_mask_path"]

        with rasterio.open(sar_path) as src:
            sar = src.read(out_dtype="float32")
        if sar.shape[0] < 3:
            raise ValueError(f"Expected at least 3 SAR bands, got {sar.shape[0]} in {sar_path}")
        sar = sar[:3]

        with rasterio.open(mask_path) as src:
            mask = src.read(1, out_dtype="float32")

        sar = self._normalize_per_tile(sar)
        mask = (mask > 0).astype(np.float32)[None, :, :]

        image_t = torch.from_numpy(sar)
        mask_t = torch.from_numpy(mask)
        if self.augment:
            image_t, mask_t = self._apply_augmentation(image_t, mask_t)
        return image_t, mask_t

    @staticmethod
    def _normalize_per_tile(sar: np.ndarray) -> np.ndarray:
        """Robustly normalize each tile to roughly zero mean and unit variance."""
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

    @staticmethod
    def _apply_augmentation(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Random horizontal flip, vertical flip, and 90-degree rotation.

        The same randomly-drawn transform is applied to both the image and the
        mask so spatial correspondence is preserved.
        """
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small, plain U-Net for binary segmentation (same architecture as the baseline)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        x = self.bottleneck(self.pool(enc4))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, enc4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, enc3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, enc2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, enc1], dim=1))
        return self.out(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018), applied to a skip connection.

    x is the encoder skip feature, g is the decoder gating signal. Both are
    expected at the same spatial resolution and channel count (true here
    because g is taken right after the matching up-conv), which keeps the
    gate to a single 1x1-conv pair plus a scalar attention map.
    """

    def __init__(self, channels: int, inter_channels: int | None = None) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = max(channels // 2, 1)
        self.theta_x = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.phi_g = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        attn = self.relu(self.theta_x(x) + self.phi_g(g))
        attn = self.sigmoid(self.psi(attn))
        return x * attn


class AttentionUNet(nn.Module):
    """U-Net with attention gates on each skip connection before concatenation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(base_channels * 8)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(base_channels * 4)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(base_channels * 2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate(base_channels)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        x = self.bottleneck(self.pool(enc4))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, self.att4(enc4, x)], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, self.att3(enc3, x)], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, self.att2(enc2, x)], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, self.att1(enc1, x)], dim=1))
        return self.out(x)


class UNetPlusPlus(nn.Module):
    """U-Net++ with nested skip connections (Zhou et al., 2018), no deep supervision.

    Uses the same 5-level width schedule as UNet/AttentionUNet (base_channels,
    *2, *4, *8, *16). Simplified relative to the original paper: each nested
    node X^{i,j} consumes only its immediate same-level predecessor X^{i,j-1}
    plus the upsampled next-deeper node X^{i+1,j-1}, rather than densely
    concatenating every prior node at that level -- this keeps the block
    count small and the forward pass easy to follow. Only the final, most-
    nested node (X^{0,4}) feeds the 1x1 output conv.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = (base_channels * m for m in (1, 2, 4, 8, 16))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder backbone: X^{i,0}
        self.conv0_0 = DoubleConv(in_channels, c0)
        self.conv1_0 = DoubleConv(c0, c1)
        self.conv2_0 = DoubleConv(c1, c2)
        self.conv3_0 = DoubleConv(c2, c3)
        self.conv4_0 = DoubleConv(c3, c4)

        # Up-sampling between adjacent levels, reused across nesting depths j
        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(c1, c0, kernel_size=2, stride=2)

        # Nested skip-pathway nodes: X^{i,j}, j > 0
        self.conv3_1 = DoubleConv(c3 * 2, c3)
        self.conv2_1 = DoubleConv(c2 * 2, c2)
        self.conv2_2 = DoubleConv(c2 * 2, c2)
        self.conv1_1 = DoubleConv(c1 * 2, c1)
        self.conv1_2 = DoubleConv(c1 * 2, c1)
        self.conv1_3 = DoubleConv(c1 * 2, c1)
        self.conv0_1 = DoubleConv(c0 * 2, c0)
        self.conv0_2 = DoubleConv(c0 * 2, c0)
        self.conv0_3 = DoubleConv(c0 * 2, c0)
        self.conv0_4 = DoubleConv(c0 * 2, c0)

        self.out = nn.Conv2d(c0, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4(x4_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3(x3_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2(x2_0)], dim=1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1(x1_0)], dim=1))

        x2_2 = self.conv2_2(torch.cat([x2_1, self.up3(x3_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_1, self.up2(x2_1)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_1, self.up1(x1_1)], dim=1))

        x1_3 = self.conv1_3(torch.cat([x1_2, self.up2(x2_2)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_2, self.up1(x1_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_3, self.up1(x1_3)], dim=1))

        return self.out(x0_4)


def build_model(arch: str, base_channels: int) -> nn.Module:
    if arch == "unet":
        return UNet(in_channels=3, out_channels=1, base_channels=base_channels)
    if arch == "attention_unet":
        return AttentionUNet(in_channels=3, out_channels=1, base_channels=base_channels)
    if arch == "unet_plus_plus":
        return UNetPlusPlus(in_channels=3, out_channels=1, base_channels=base_channels)
    raise ValueError(f"Unknown arch: {arch}")


class DiceLoss(nn.Module):
    """Soft Dice loss computed from sigmoid(logits), averaged over the batch."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCEWithLogitsLoss + DiceLoss."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


def build_loss_fn(name: str) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss(smooth=1e-6)
    if name == "bce_dice":
        return BCEDiceLoss(smooth=1e-6)
    raise ValueError(f"Unknown loss: {name}")


def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets_bool = targets > 0.5

    intersection = (preds & targets_bool).sum().float()
    pred_sum = preds.sum().float()
    target_sum = targets_bool.sum().float()
    union = (preds | targets_bool).sum().float()

    eps = torch.tensor(1e-7, device=logits.device)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice.item()), float(iou.item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        dice, iou = dice_iou_from_logits(logits.detach(), masks)
        total_loss += float(loss.item())
        total_dice += dice
        total_iou += iou
        total_batches += 1

    if total_batches == 0:
        raise ValueError("DataLoader produced no batches.")

    return {
        "loss": total_loss / total_batches,
        "dice": total_dice / total_batches,
        "iou": total_iou / total_batches,
    }


METRICS_FIELDNAMES = [
    "epoch",
    "train_loss", "train_dice", "train_iou",
    "val_loss", "val_dice", "val_iou",
    "test_loss", "test_dice", "test_iou",
]


def write_metrics_csv(metrics_path: Path, rows: list[dict[str, float | int | str]]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDNAMES, restval="")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    default_split_dir = Path("csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap")
    parser = argparse.ArgumentParser(description="Train an experimental SAR-only binary U-Net.")
    parser.add_argument("--data-root", type=Path, default=Path("2025_Tile_Data"))
    parser.add_argument("--train-csv", type=Path, default=default_split_dir / "heldout_fp1_train.csv")
    parser.add_argument("--val-csv", type=Path, default=default_split_dir / "heldout_fp1_validation.csv")
    parser.add_argument("--test-csv", type=Path, default=default_split_dir / "heldout_fp1_test.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="unet_arch_experiment_strict_fp1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arch", choices=["unet", "attention_unet", "unet_plus_plus"], default="unet")
    parser.add_argument("--loss", choices=["bce", "dice", "bce_dice"], default="bce")
    parser.add_argument("--save-best-by", choices=["val_loss", "val_dice"], default="val_dice")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Random flips + 90-degree rotations on the training split only.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_improvement(save_best_by: str, candidate: float, best: float) -> bool:
    if save_best_by == "val_loss":
        return candidate < best
    return candidate > best  # val_dice


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Validation CSV: {args.val_csv}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Arch: {args.arch}")
    print(f"Loss: {args.loss}")
    print(f"Save best by: {args.save_best_by}")
    print(f"Augment: {args.augment}")

    train_dataset = FloodTileDataset(args.train_csv, args.data_root, augment=args.augment)
    val_dataset = FloodTileDataset(args.val_csv, args.data_root, augment=False)
    test_dataset = FloodTileDataset(args.test_csv, args.data_root, augment=False)
    print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(args.arch, args.base_channels).to(device)
    loss_fn = build_loss_fn(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / f"{args.run_name}_best.pt"
    metrics_path = args.results_dir / f"{args.run_name}_metrics.csv"

    best_metric_value = float("inf") if args.save_best_by == "val_loss" else float("-inf")
    history: list[dict[str, float | int | str]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device)

        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)
        write_metrics_csv(metrics_path, history)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        candidate_value = val_metrics["loss"] if args.save_best_by == "val_loss" else val_metrics["dice"]
        if is_improvement(args.save_best_by, candidate_value, best_metric_value):
            best_metric_value = candidate_value
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "arch": args.arch,
                    "loss": args.loss,
                    "save_best_by": args.save_best_by,
                    "base_channels": args.base_channels,
                    "best_metric_value": best_metric_value,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint ({args.save_best_by}={best_metric_value:.4f}): {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, loss_fn, device)
    print(
        "Best checkpoint test metrics: "
        f"loss={test_metrics['loss']:.4f} "
        f"dice={test_metrics['dice']:.4f} "
        f"iou={test_metrics['iou']:.4f}"
    )

    history.append(
        {
            "epoch": "test",
            "test_loss": test_metrics["loss"],
            "test_dice": test_metrics["dice"],
            "test_iou": test_metrics["iou"],
        }
    )
    write_metrics_csv(metrics_path, history)
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
