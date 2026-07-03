#!/usr/bin/env python3
"""Train a simple SAR-only U-Net baseline for flood/change segmentation.

Example: 
    python3 milton/train_test_unet.py \
        --train-csv {your training csv here} \
        --val-csv {your validation csv here} \
        --test-csv {your test csv here}
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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
  --train-csv PATH        Training split CSV
  --val-csv PATH          Validation split CSV
  --test-csv PATH         Test split CSV
  --epochs N              Number of epochs (default: 20)
  --batch-size N          Batch size (default: 8)
  --learning-rate LR      Adam learning rate (default: 1e-3)
  --num-workers N         DataLoader workers (default: 2)
  --base-channels N       U-Net width (default: 32)
  --seed N                Random seed (default: 42)
  --output-dir PATH       Root output folder for this run; creates checkpoints/results/visuals subfolders
  --models-dir PATH       Checkpoint output folder (default: models, ignored if --output-dir is given)
  --results-dir PATH      Metrics CSV output folder (default: results, ignored if --output-dir is given)
  --visuals-dir PATH      Visualization output folder (default: visuals inside --output-dir, or outputs/baseline_visuals)
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
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


class FloodTileDataset(Dataset):
    """Reads SAR image tiles and binary flood/change masks from split CSV rows."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
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

    def __getitem__(self, index: int):

        row = self.rows[index]

        sar_path = row["uavsar_path"]
        mask_path = row["flood_mask_path"]

        # -----------------------------
        # Read SAR
        # -----------------------------
        with rasterio.open(sar_path) as src:
            sar = src.read().astype(np.float32)

        if sar.shape[0] < 3:
            raise ValueError(
                f"Expected at least 3 SAR bands, got {sar.shape[0]} in {sar_path}"
            )

        sar = sar[:3]

        # Valid SAR pixels:
        # valid unless ALL THREE bands are zero
        sar_valid = ~(sar == 0).all(axis=0)

        # Convert only those invalid pixels to NaN
        sar[:, ~sar_valid] = np.nan

        # -----------------------------
        # Read mask
        # -----------------------------
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

            # start by assuming everything is valid
            valid_mask = np.ones(mask.shape, dtype=bool)

            # some padding for no data value

            valid_mask &= (mask < 200)

        # Binary flood mask
        binary_mask = np.zeros(mask.shape, dtype=np.float32)
        binary_mask[(mask == 1) & valid_mask] = 1.0

        # Normalize SAR using SAR validity mask
        sar = self._normalize_per_tile(sar, sar_valid)

        # Final validity mask
        valid = sar_valid & valid_mask

        return (
            torch.from_numpy(sar),
            torch.from_numpy(binary_mask[None]),
            torch.from_numpy(valid.astype(np.float32)[None]),
        )
    
    @staticmethod
    def _normalize_per_tile(
        sar: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:

        sar = sar.copy()

        for c in range(sar.shape[0]):

            band = sar[c]

            values = band[valid]

            if values.size == 0:
                band[:] = 0.0
                sar[c] = band
                continue

            low, high = np.percentile(values, [1.0, 99.0])

            values = np.clip(values, low, high)

            mean = values.mean()
            std = values.std()

            if std < 1e-6:
                band[:] = 0.0
            else:
                band[valid] = (values - mean) / std

            # Keep invalid pixels at zero
            band[~valid] = 0.0

            sar[c] = band

        return sar.astype(np.float32)


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

#modified to handle no data valuese
def dice_iou_from_logits(logits, targets, mask, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets_bool = targets > 0.5

    # apply mask (IMPORTANT)
    preds = preds & (mask > 0.5) #keep validity with some padding on either side
    targets_bool = targets_bool & (mask > 0.5)

    intersection = (preds & targets_bool).sum().float()
    pred_sum = preds.sum().float()
    target_sum = targets_bool.sum().float()
    union = (preds | targets_bool).sum().float()

    eps = 1e-7

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)

    return float(dice.item()), float(iou.item())


#CUSTOM LOSS FUNCTIONS 

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, mask=None):

        probs = torch.sigmoid(logits)

        if mask is not None:
            probs = probs * mask
            targets = targets * mask

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        denom = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets, mask=None):

        bce_loss = self.bce(logits, targets)
        if mask is not None:
            bce_loss = (bce_loss * mask).sum() / (mask.sum() + 1e-6)
        dice_loss = self.dice(logits, targets, mask)

        return bce_loss + dice_loss
    
#handles class imbalance by downweighting easy negatives and focusing more on hard examples

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, mask=None):

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs
        )

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)

        return loss.mean()
    
#focal and dice combined to handle class imbalance and optimize for segmentation metrics

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()

        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets, mask):

        focal_loss = self.focal(logits, targets, mask)
        dice_loss = self.dice(logits, targets, mask)

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
    
class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Compute a loss for every pixel
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, mask):

        # logits: (B, 1, H, W)
        # targets: (B, 1, H, W)
        # mask: (B, 1, H, W)

        loss = self.bce(logits, targets)

        # Ignore no-data pixels
        loss = loss * mask

        # Average only over valid pixels
        return loss.sum() / (mask.sum() + 1e-6)

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

    for images, masks, valid in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = loss_fn(logits, masks, valid)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        dice, iou = dice_iou_from_logits(logits.detach(), masks, valid)
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
    
    parser = argparse.ArgumentParser(description="Train a simple SAR-only binary U-Net baseline.")
    parser.add_argument("--train-csv", type=Path, default="milton/csv_splits/png_train.csv")
    parser.add_argument("--val-csv", type=Path, default= "milton/csv_splits/png_validation.csv")
    parser.add_argument("--test-csv", type=Path, default="milton/flood_ratio_csv_splits/test_10.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss-fn", choices=["bce", "dice", "bce_dice", "focal", "focal_dice"], default="bce")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Root output folder for this run. If set, checkpoints/results/visuals are created inside it.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--visuals-dir", type=Path, default=Path("outputs/baseline_visuals"))
    parser.add_argument("--run-name", default=None,
                        help="Prefix for output files. If omitted, inferred from output-dir or test-csv parent.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a non-empty output-dir.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    """Convert argparse Namespace to JSON-safe dict."""
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def infer_run_name(args: argparse.Namespace) -> str:
    """Infer a useful run name when --run-name is not provided."""
    if args.run_name:
        return args.run_name

    if args.output_dir is not None:
        return args.output_dir.name

    # Example: .../target_05_10_percent/test.csv -> target_05_10_percent
    test_parent = Path(args.test_csv).parent.name
    if test_parent:
        return f"baseline_{test_parent}_seed_{args.seed}"

    return f"unet_baseline_seed_{args.seed}"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.run_name = infer_run_name(args)

    if args.output_dir is not None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise RuntimeError(
                f"Output directory already exists and is not empty: {args.output_dir}\n"
                "Use a different --output-dir or pass --overwrite intentionally."
            )
        args.models_dir = args.output_dir / "checkpoints"
        args.results_dir = args.output_dir / "results"
        args.visuals_dir = args.output_dir / "visuals"

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Validation CSV: {args.val_csv}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Run name: {args.run_name}")
    print(f"Models dir: {args.models_dir}")
    print(f"Results dir: {args.results_dir}")
    print(f"Visuals dir: {args.visuals_dir}")
    #make datasets from the csvs
    train_dataset = FloodTileDataset(args.train_csv)
    val_dataset = FloodTileDataset(args.val_csv)
    test_dataset = FloodTileDataset(args.test_csv)
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
    loss_name = args.loss_fn  

    if loss_name == "dice":
         loss_fn = DiceLoss()
    elif loss_name == "bce_dice":
         loss_fn = BCEDiceLoss()
    elif loss_name == "focal":
        loss_fn = FocalLoss()
    elif loss_name == "focal_dice":
        loss_fn = FocalDiceLoss()
    else:
        loss_fn = MaskedBCEWithLogitsLoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw": 
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.visuals_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.models_dir / f"{args.run_name}_best.pt"
    metrics_path = args.results_dir / f"{args.run_name}_metrics.csv"
    test_metrics_path = args.results_dir / f"{args.run_name}_test_metrics.csv"
    config_path = args.results_dir / f"{args.run_name}_config.json"

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe_args(args), handle, indent=2)

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
            f"train_dice={train_metrics['dice']:.4f} "
            f"train_iou={train_metrics['iou']:.4f} "
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

    checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, loss_fn, device)
    print(
        "Best checkpoint test metrics: "
        f"loss={test_metrics['loss']:.4f} "
        f"dice={test_metrics['dice']:.4f} "
        f"iou={test_metrics['iou']:.4f}"
    )
    with test_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["loss", "dice", "iou"])
        writer.writeheader()
        writer.writerow(test_metrics)

    print(f"Metrics CSV: {metrics_path}")
    print(f"Test metrics CSV: {test_metrics_path}")
    print(f"Config JSON: {config_path}")


    # -------------------------------------------------------
    # Visualize one training prediction
    # -------------------------------------------------------
    model.eval()

    train_dataset = FloodTileDataset(Path(args.train_csv))

    image, mask, valid = test_dataset[0]

    with torch.no_grad():

        logits = model(image.unsqueeze(0).to(device))

        probs = torch.sigmoid(logits)

        prediction = (probs > 0.5).float()

    prediction = prediction.squeeze().cpu().numpy()
    mask = mask.squeeze().numpy()
    valid = valid.squeeze().numpy()

    sar = image[0].numpy()
    print("valid unique:", np.unique(valid))
    print("valid mean:", valid.mean())
    print("valid sum:", valid.sum())
    print("shape:", valid.shape)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(20,5))

    ax[0].imshow(sar, cmap="gray")
    ax[0].set_title("SAR")

    ax[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Ground Truth")

    ax[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title("Prediction")

    ax[3].imshow(valid, cmap="gray", vmin=0, vmax=1)
    ax[3].set_title("Valid Pixels")

    for a in ax:
        a.axis("off")

    visual_path = args.visuals_dir / f"{args.run_name}_test_sample.png"
    fig.savefig(visual_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {visual_path}")


if __name__ == "__main__":
    main()
