#!/usr/bin/env python3
"""Train a simple SAR-only U-Net baseline for flood/change segmentation.

Example:
    python scripts/train_unet_baseline.py \
        --train-csv csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap/heldout_fp1_train.csv \
        --val-csv csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap/heldout_fp1_validation.csv \
        --test-csv csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap/heldout_fp1_test.csv
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
        """usage: train_unet_baseline.py [options]

Train a simple SAR-only binary U-Net baseline.

options:
  --data-root PATH        Root folder for tile paths (default: 2025_Tile_Data)
  --train-csv PATH        Training split CSV
  --val-csv PATH          Validation split CSV
  --test-csv PATH         Test split CSV
  --epochs N              Number of epochs (default: 20)
  --batch-size N          Batch size (default: 8)
  --learning-rate LR      Adam learning rate (default: 1e-3)
  --num-workers N         DataLoader workers (default: 2)
  --base-channels N       U-Net width (default: 32)
  --seed N                Random seed (default: 42)
  --models-dir PATH       Checkpoint output folder (default: models)
  --results-dir PATH      Metrics CSV output folder (default: results)
  --run-name NAME         Prefix for output files
  --device DEVICE         cuda or cpu

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
    """Reads SAR image tiles and binary flood/change masks from split CSV rows."""

    def __init__(self, csv_path: Path, data_root: Path) -> None:
        self.csv_path = csv_path
        self.data_root = data_root
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

        return torch.from_numpy(sar), torch.from_numpy(mask)

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
    """Small, plain U-Net for binary segmentation."""

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


def write_metrics_csv(metrics_path: Path, rows: list[dict[str, float | int]]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    default_split_dir = Path("csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap")
    parser = argparse.ArgumentParser(description="Train a simple SAR-only binary U-Net baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("2025_Tile_Data"))
    parser.add_argument("--train-csv", type=Path, default=default_split_dir / "heldout_fp1_train.csv")
    parser.add_argument("--val-csv", type=Path, default=default_split_dir / "heldout_fp1_validation.csv")
    parser.add_argument("--test-csv", type=Path, default=default_split_dir / "heldout_fp1_test.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="unet_baseline_strict_fp1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Validation CSV: {args.val_csv}")
    print(f"Test CSV: {args.test_csv}")

    train_dataset = FloodTileDataset(args.train_csv, args.data_root)
    val_dataset = FloodTileDataset(args.val_csv, args.data_root)
    test_dataset = FloodTileDataset(args.test_csv, args.data_root)
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

    model = UNet(in_channels=3, out_channels=1, base_channels=args.base_channels).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / f"{args.run_name}_best.pt"
    metrics_path = args.results_dir / f"{args.run_name}_metrics.csv"

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device)

        row = {
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

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, loss_fn, device)
    print(
        "Best checkpoint test metrics: "
        f"loss={test_metrics['loss']:.4f} "
        f"dice={test_metrics['dice']:.4f} "
        f"iou={test_metrics['iou']:.4f}"
    )
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
