#!/usr/bin/env python3

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from your training script
from train_test_unet import (
    FloodTileDataset,
    UNet,
    FocalDiceLoss,
    run_epoch,
)

def evaluate_precision_recall(model, loader, device, threshold=0.5):
    
    #Computes average precision and recall over all valid pixels in the dataset.
    

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():

        for images, masks, valid in loader:

            images = images.to(device)
            masks = masks.to(device)
            valid = valid.to(device)

            logits = model(images)

            probs = torch.sigmoid(logits)
            preds = probs > threshold

            targets = masks > 0.5
            valid_mask = valid > 0.5

            preds = preds & valid_mask
            targets = targets & valid_mask

            total_tp += (preds & targets).sum().item()
            total_fp += (preds & ~targets).sum().item()
            total_fn += (~preds & targets).sum().item()

    eps = 1e-7

    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)

    return precision, recall

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV file to evaluate",
        default ="milton/flood_bin_csv_splits/florence_test_10-25.csv"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Saved model (.pt)",
        default = "models/unet_baseline_strict_tuned_fp1_best.pt"
    )

    parser.add_argument(
        "--index",
        type=int,
        default=2 ,
        help="Tile index to visualize",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ####################################
    # Load dataset
    ####################################

    dataset = FloodTileDataset(args.csv)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    ####################################
    # Load model
    ####################################

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=checkpoint["args"]["base_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ####################################
    # Evaluate
    ####################################

    loss_fn = FocalDiceLoss()

    metrics = run_epoch(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        device=device,
    )

    precision, recall = evaluate_precision_recall(
    model,
    loader,
    device,
)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Tiles : {len(dataset)}")
    print(f"Loss  : {metrics['loss']:.4f}")
    print(f"Dice  : {metrics['dice']:.4f}")
    print(f"IoU   : {metrics['iou']:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    ####################################
    # Visualize one tile
    ####################################

    image, mask, valid = dataset[args.index]

    with torch.no_grad():

        logits = model(
            image.unsqueeze(0).to(device)
        )

        probs = torch.sigmoid(logits)

        prediction = (probs > 0.5).float()

    sar = image[0].numpy()

    mask = mask.squeeze().numpy()

    prediction = prediction.squeeze().cpu().numpy()

    

    valid = valid.squeeze().numpy()

    print("\nVisualization Tile")
    print("------------------")
    print("Flood fraction:", mask.mean())
    print("Predicted flood fraction:", prediction.mean())

    fig, ax = plt.subplots(1, 4, figsize=(24, 5))

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

    plt.tight_layout()
    save_name = Path(args.csv).name
    save_path= f"milton/visualizations/{save_name}_tuned_visual.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()