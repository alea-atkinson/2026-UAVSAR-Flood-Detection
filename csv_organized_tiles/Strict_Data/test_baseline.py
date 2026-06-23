#!/usr/bin/env python3

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from train_unet_baseline_tuned import (
    UNet,
    FloodTileDataset,
    DiceLoss,
    BCEDiceLoss,
    FocalLoss,
    FocalDiceLoss,
    run_epoch,
)

def parse_args() -> argparse.Namespace:
    default_split_dir = Path("csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap")
    parser = argparse.ArgumentParser(description="Test a simple SAR-only binary U-Net baseline.")
    parser.add_argument("--fp", type=int, default=1)
    return parser.parse_args()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

def main():

    args = parse_args()

    #CHECKPOINT = Path(
        #f"models/final_tuning_basic_fp{args.fp}_best.pt"
    #)

    CHECKPOINT = Path(
        f"models/tune_basic_fp{args.fp}_png_adam_bce_dice_best.pt"
    )
    print(CHECKPOINT)

    CSV_FILE = Path(
        f"csv_splits/flood_splits_standard_strict_train_val/strict_no_overlap/heldout_fp{args.fp}_test.csv"
    )

    DATA_ROOT = Path("2025_Tile_Data")

    BATCH_SIZE = 16
    NUM_WORKERS = 2

    # --------------------------------------------------
    # MAIN
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # load checkpoint

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )



    base_channels = 32

    # rebuild model

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=base_channels
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # dataset

    dataset = FloodTileDataset(
        CSV_FILE,
        DATA_ROOT
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # loss function

    loss_fn = BCEDiceLoss()

    # evaluation

    metrics = run_epoch(
        model,
        loader,
        loss_fn,
        device
    )

    print(f"Dataset size: test={len(dataset)}")



    #evaulate precision and recall


    model.eval()
    #calc false positives and false negatives for precision and recall

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.inference_mode():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            probs = torch.sigmoid(logits)

            preds = probs > 0.5 #threshold
            targets = masks > 0.5

            total_tp += (preds & targets).sum().item()
            total_fp += (preds & ~targets).sum().item()
            total_fn += (~preds & targets).sum().item()
            total_tn += (~preds & ~targets).sum().item()
        
        #calcuate precision and recal using fp and fn

        precision = total_tp / (total_tp + total_fp + 1e-7)

        recall = total_tp / (total_tp + total_fn + 1e-7)


    print("\n=== Metrics ===")
    print(f"Dice:      {metrics['dice']:.4f}")
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
 

    print("Checkpoint epoch:", checkpoint['epoch'])
    print("Best val loss:", checkpoint['best_val_loss'])



if __name__ == "__main__":
    main()