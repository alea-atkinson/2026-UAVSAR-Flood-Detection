#!/usr/bin/env python3
"""Train a SPIE paper-style flood segmentation model on the available dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

IMAGE_SIZE = 128
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
PAPER_METRICS = {
    "unet": {"accuracy": 0.87, "precision": 0.72, "dice": 0.76, "recall": 0.80},
    "attention_unet": {"accuracy": 0.88, "precision": 0.74, "dice": 0.71, "recall": 0.68},
    "unetpp": {"accuracy": 0.80, "precision": 0.62, "dice": 0.57, "recall": 0.53},
}
MODEL_NAMES = {
    "unet": "U-Net",
    "attention_unet": "Attention U-Net",
    "unetpp": "U-Net++",
}


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def paired_files(dataset_root: Path, split: str) -> list[tuple[str, str]]:
    sar_dir = dataset_root / split / "sar"
    flood_dir = dataset_root / split / "flood"
    pairs: list[tuple[str, str]] = []
    for sar_path in sorted(sar_dir.glob("*.tif")):
        flood_path = flood_dir / sar_path.name.replace("_prep.tif", "_flood_prep.tif")
        if flood_path.exists():
            pairs.append((str(sar_path), str(flood_path)))
    return pairs


def load_pair_numpy(sar_path: bytes, flood_path: bytes) -> tuple[np.ndarray, np.ndarray]:
    sar_file = Path(sar_path.decode("utf-8"))
    flood_file = Path(flood_path.decode("utf-8"))
    with rasterio.open(sar_file) as src:
        sar = src.read(out_dtype="float32")
    if sar.ndim == 3:
        sar = np.transpose(sar, (1, 2, 0))
    if sar.shape[-1] > 3:
        sar = sar[..., :3]
    sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    with rasterio.open(flood_file) as src:
        mask = src.read(1, out_dtype="float32")
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (mask > 0.5).astype("float32")[..., None]
    return sar, mask


def tf_load_pair(sar_path: tf.Tensor, flood_path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image, mask = tf.numpy_function(load_pair_numpy, [sar_path, flood_path], [tf.float32, tf.float32])
    image.set_shape(INPUT_SHAPE)
    mask.set_shape((IMAGE_SIZE, IMAGE_SIZE, 1))
    return image, mask


def augment_pair(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    joined = tf.concat([image, mask], axis=-1)
    joined = tf.image.random_flip_left_right(joined)
    joined = tf.image.random_flip_up_down(joined)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    joined = tf.image.rot90(joined, k=k)
    image = joined[..., :3]
    mask = joined[..., 3:]

    brightness_delta = tf.random.uniform((), minval=-0.05, maxval=0.05, dtype=tf.float32)
    image = image + brightness_delta
    return image, tf.cast(mask > 0.5, tf.float32)


def make_dataset(
    pairs: list[tuple[str, str]],
    batch_size: int,
    training: bool,
    augment: bool,
    seed: int,
) -> tf.data.Dataset:
    sar_paths = [pair[0] for pair in pairs]
    flood_paths = [pair[1] for pair in pairs]
    ds = tf.data.Dataset.from_tensor_slices((sar_paths, flood_paths))
    if training:
        ds = ds.shuffle(buffer_size=len(pairs), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if training and augment:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def soft_dice_metric(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + (1.0 - soft_dice_metric(y_true, y_pred))


def conv_block(x: tf.Tensor, filters: int, dropout: float, name: str) -> tf.Tensor:
    for idx in range(2):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
            name=f"{name}_conv{idx + 1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn{idx + 1}")(x)
        x = tf.keras.layers.Activation("relu", name=f"{name}_relu{idx + 1}")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return x


def build_unet(input_shape: tuple[int, int, int], dropout: float) -> tf.keras.Model:
    inputs = tf.keras.Input(input_shape)
    c1 = conv_block(inputs, 64, dropout, "enc1")
    p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 128, dropout, "enc2")
    p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 256, dropout, "enc3")
    p3 = tf.keras.layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 512, dropout, "enc4")
    p4 = tf.keras.layers.MaxPooling2D()(c4)
    bridge = conv_block(p4, 1024, dropout, "bridge")

    u4 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding="same")(bridge)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    d4 = conv_block(u4, 512, dropout, "dec4")
    u3 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same")(d4)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    d3 = conv_block(u3, 256, dropout, "dec3")
    u2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d3)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 128, dropout, "dec2")
    u1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d2)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    d1 = conv_block(u1, 64, dropout, "dec1")
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="flood_probability")(d1)
    return tf.keras.Model(inputs, outputs, name="paper_style_unet")


def attention_gate(skip: tf.Tensor, gating: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    theta = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_theta")(skip)
    phi = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_phi")(gating)
    add = tf.keras.layers.Activation("relu", name=f"{name}_relu")(tf.keras.layers.Add()([theta, phi]))
    psi = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name=f"{name}_psi")(add)
    return tf.keras.layers.Multiply(name=f"{name}_scale")([skip, psi])


def build_attention_unet(input_shape: tuple[int, int, int], dropout: float) -> tf.keras.Model:
    inputs = tf.keras.Input(input_shape)
    c1 = conv_block(inputs, 64, dropout, "enc1")
    p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 128, dropout, "enc2")
    p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 256, dropout, "enc3")
    p3 = tf.keras.layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 512, dropout, "enc4")
    p4 = tf.keras.layers.MaxPooling2D()(c4)
    bridge = conv_block(p4, 1024, dropout, "bridge")

    u4 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding="same")(bridge)
    g4 = attention_gate(c4, u4, 256, "att4")
    d4 = conv_block(tf.keras.layers.Concatenate()([u4, g4]), 512, dropout, "dec4")
    u3 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same")(d4)
    g3 = attention_gate(c3, u3, 128, "att3")
    d3 = conv_block(tf.keras.layers.Concatenate()([u3, g3]), 256, dropout, "dec3")
    u2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d3)
    g2 = attention_gate(c2, u2, 64, "att2")
    d2 = conv_block(tf.keras.layers.Concatenate()([u2, g2]), 128, dropout, "dec2")
    u1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d2)
    g1 = attention_gate(c1, u1, 32, "att1")
    d1 = conv_block(tf.keras.layers.Concatenate()([u1, g1]), 64, dropout, "dec1")
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="flood_probability")(d1)
    return tf.keras.Model(inputs, outputs, name="paper_style_attention_unet")


def upsample_concat(x: tf.Tensor, skips: list[tf.Tensor], filters: int, name: str) -> tf.Tensor:
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"{name}_up")(x)
    return tf.keras.layers.Concatenate(name=f"{name}_concat")([x, *skips])


def build_unetpp(input_shape: tuple[int, int, int], dropout: float) -> tf.keras.Model:
    inputs = tf.keras.Input(input_shape)
    x00 = conv_block(inputs, 64, dropout, "x00")
    x10 = conv_block(tf.keras.layers.MaxPooling2D()(x00), 128, dropout, "x10")
    x20 = conv_block(tf.keras.layers.MaxPooling2D()(x10), 256, dropout, "x20")
    x30 = conv_block(tf.keras.layers.MaxPooling2D()(x20), 512, dropout, "x30")
    x40 = conv_block(tf.keras.layers.MaxPooling2D()(x30), 1024, dropout, "x40")

    x01 = conv_block(upsample_concat(x10, [x00], 64, "x01"), 64, dropout, "x01_conv")
    x11 = conv_block(upsample_concat(x20, [x10], 128, "x11"), 128, dropout, "x11_conv")
    x21 = conv_block(upsample_concat(x30, [x20], 256, "x21"), 256, dropout, "x21_conv")
    x31 = conv_block(upsample_concat(x40, [x30], 512, "x31"), 512, dropout, "x31_conv")

    x02 = conv_block(upsample_concat(x11, [x00, x01], 64, "x02"), 64, dropout, "x02_conv")
    x12 = conv_block(upsample_concat(x21, [x10, x11], 128, "x12"), 128, dropout, "x12_conv")
    x22 = conv_block(upsample_concat(x31, [x20, x21], 256, "x22"), 256, dropout, "x22_conv")

    x03 = conv_block(upsample_concat(x12, [x00, x01, x02], 64, "x03"), 64, dropout, "x03_conv")
    x13 = conv_block(upsample_concat(x22, [x10, x11, x12], 128, "x13"), 128, dropout, "x13_conv")

    x04 = conv_block(upsample_concat(x13, [x00, x01, x02, x03], 64, "x04"), 64, dropout, "x04_conv")
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="flood_probability")(x04)
    return tf.keras.Model(inputs, outputs, name="paper_style_unetpp")


def build_model(model_name: str, dropout: float) -> tf.keras.Model:
    builders: dict[str, Callable[[tuple[int, int, int], float], tf.keras.Model]] = {
        "unet": build_unet,
        "attention_unet": build_attention_unet,
        "unetpp": build_unetpp,
    }
    return builders[model_name](INPUT_SHAPE, dropout)


def global_threshold_metrics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    threshold: float,
) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for images, masks in dataset:
        probs = model.predict(images, verbose=0)
        pred = probs >= threshold
        true = masks.numpy() >= 0.5
        tp += int(np.logical_and(pred, true).sum())
        fp += int(np.logical_and(pred, ~true).sum())
        fn += int(np.logical_and(~pred, true).sum())
        tn += int(np.logical_and(~pred, ~true).sum())

    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = (2 * tp) / max((2 * tp + fp + fn), 1)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
    }


def json_ready_history(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("spie/Preprocessed-128-paper-filtered"))
    parser.add_argument("--output-root", type=Path, default=Path("spie/paper_style_rerun/artifacts"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)

    output_dir = args.output_root / args.model
    train_pairs = paired_files(args.dataset_root, "train")
    val_pairs = paired_files(args.dataset_root, "val")
    test_pairs = paired_files(args.dataset_root, "test")
    config = {
        "model": args.model,
        "model_label": MODEL_NAMES[args.model],
        "rerun_label": "paper-style rerun on available Preprocessed-128 dataset after water filtering",
        "dataset_note": (
            "The exact curated 3,873-tile SPIE dataset is unavailable in this repo. "
            "This run uses Preprocessed-128 filtered to 0.05 <= water_ratio <= 0.85."
        ),
        "dataset_root": str(args.dataset_root.resolve()),
        "input_shape": list(INPUT_SHAPE),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "loss": "binary_crossentropy + dice_loss",
        "training_monitor_metric": "soft_dice_metric",
        "dropout": args.dropout,
        "early_stopping": False,
        "reduce_lr_on_plateau": False,
        "final_evaluation": "global thresholded TP/FP/FN/TN",
        "threshold": args.threshold,
        "augmentation": {
            "enabled": not args.no_augment,
            "train_only": True,
            "operations": ["horizontal_flip", "vertical_flip", "rot90", "sar_brightness_delta"],
        },
        "paper_metrics": PAPER_METRICS[args.model],
    }
    write_json(output_dir / "config.json", config)
    if args.config_only:
        print(json.dumps({"config": str(output_dir / "config.json")}, indent=2))
        return

    train_ds = make_dataset(train_pairs, args.batch_size, training=True, augment=not args.no_augment, seed=args.seed)
    val_ds = make_dataset(val_pairs, args.batch_size, training=False, augment=False, seed=args.seed)
    test_ds = make_dataset(test_pairs, args.batch_size, training=False, augment=False, seed=args.seed)

    model = build_model(args.model, args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=bce_dice_loss,
        metrics=[soft_dice_metric],
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    write_json(output_dir / "training_history.json", json_ready_history(history))

    metrics = global_threshold_metrics(model, test_ds, threshold=args.threshold)
    metrics["paper_metrics"] = PAPER_METRICS[args.model]
    metrics["gap"] = {
        key: float(metrics[key] - PAPER_METRICS[args.model][key])
        for key in ("accuracy", "precision", "recall", "dice")
    }
    write_json(output_dir / "final_test_metrics.json", metrics)
    model.save(output_dir / "model.keras")
    print(json.dumps({"output_dir": str(output_dir), "final_test_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
