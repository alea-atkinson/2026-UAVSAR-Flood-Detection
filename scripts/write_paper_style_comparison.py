#!/usr/bin/env python3
"""Write a comparison table for the available-dataset SPIE paper-style rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER = {
    "unet": {"label": "U-Net", "accuracy": 0.87, "precision": 0.72, "recall": 0.80, "dice": 0.76},
    "attention_unet": {"label": "Attention U-Net", "accuracy": 0.88, "precision": 0.74, "recall": 0.68, "dice": 0.71},
    "unetpp": {"label": "U-Net++", "accuracy": 0.80, "precision": 0.62, "recall": 0.53, "dice": 0.57},
}


def load_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(value: object) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_rows(artifact_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_key, paper in PAPER.items():
        metrics = load_metrics(artifact_root / model_key / "final_test_metrics.json")
        gap = None
        if metrics is not None:
            gap = {
                "accuracy": metrics["accuracy"] - paper["accuracy"],
                "precision": metrics["precision"] - paper["precision"],
                "recall": metrics["recall"] - paper["recall"],
                "dice": metrics["dice"] - paper["dice"],
            }
        rows.append({
            "Model": paper["label"],
            "Accuracy": None if metrics is None else metrics["accuracy"],
            "Precision": None if metrics is None else metrics["precision"],
            "Recall": None if metrics is None else metrics["recall"],
            "Dice": None if metrics is None else metrics["dice"],
            "Paper Accuracy": paper["accuracy"],
            "Paper Precision": paper["precision"],
            "Paper Recall": paper["recall"],
            "Paper Dice": paper["dice"],
            "Gap": None if gap is None else gap,
        })
    return rows


def write_markdown(rows: list[dict[str, object]], output: Path) -> None:
    lines = [
        "# Paper-Style Rerun Comparison",
        "",
        "This is a paper-style rerun on the available 25,168-pair `Preprocessed-128` repo dataset after applying `0.05 <= water_ratio <= 0.85`. The exact curated 3,873-tile SPIE dataset is unavailable here.",
        "",
        "| Model | Accuracy | Precision | Recall | Dice | Paper Accuracy | Paper Precision | Paper Recall | Paper Dice | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        gap = row["Gap"]
        if isinstance(gap, dict):
            gap_text = ", ".join(f"{key} {value:+.4f}" for key, value in gap.items())
        else:
            gap_text = "pending"
        lines.append(
            "| {Model} | {Accuracy} | {Precision} | {Recall} | {Dice} | {Paper Accuracy} | {Paper Precision} | {Paper Recall} | {Paper Dice} | {Gap} |".format(
                **{key: fmt(value) for key, value in row.items() if key != "Gap"},
                Gap=gap_text,
            )
        )
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("spie/paper_style_rerun/artifacts"))
    parser.add_argument("--json-output", type=Path, default=Path("spie/paper_style_rerun/comparison_table.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("spie/paper_style_rerun/comparison_table.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.artifact_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(rows, indent=2) + "\n")
    write_markdown(rows, args.markdown_output)
    print(json.dumps({
        "json_output": str(args.json_output),
        "markdown_output": str(args.markdown_output),
    }, indent=2))


if __name__ == "__main__":
    main()
