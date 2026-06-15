#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path


SOURCE_NOTEBOOK_ENV = "UNET_128_NOTEBOOK_SOURCE"
OUTPUT_NOTEBOOK_ENV = "UNET_128_NOTEBOOK_OUTPUT"
DEFAULT_SOURCE_FILENAME = "unet_128_2025reu.ipynb"
DEFAULT_OUTPUT_NOTEBOOK = Path("notebooks/unet_128_2025reu_vscode.ipynb")


def to_source_block(text: str) -> list[str]:
    return [f"{line}\n" for line in text.strip("\n").splitlines()]


def replace_import_cell(cell: dict) -> None:
    source_lines = [line.rstrip("\n") for line in cell.get("source", [])]
    if "import subprocess" not in source_lines:
        insert_at = next(
            (index + 1 for index, line in enumerate(source_lines) if line.strip() == "import json"),
            None,
        )
        if insert_at is None:
            import_lines = [
                index for index, line in enumerate(source_lines) if line.strip().startswith(("import ", "from "))
            ]
            insert_at = import_lines[-1] + 1 if import_lines else 0
        source_lines.insert(insert_at, "import subprocess")
    cell["source"] = [f"{line}\n" for line in source_lines]


def replace_dependency_cell(cell: dict) -> None:
    cell["source"] = to_source_block(
        """
# Install notebook dependencies in your active environment before running, for example:
# python -m pip install rasterio tensorflow matplotlib scikit-learn
        """
    )


def replace_gpu_cell(cell: dict) -> None:
    cell["source"] = to_source_block(
        """
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("GPU Information:")
            print(result.stdout.strip().splitlines()[0])
        else:
            print("GPU detected, but nvidia-smi is unavailable.")

        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        print(f"\\nMixed precision policy: {policy.name}")
        print("Compute dtype:", policy.compute_dtype)
        print("Variable dtype:", policy.variable_dtype)

    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPUs found")
        """
    )


def replace_data_cell(cell: dict) -> None:
    cell["source"] = to_source_block(
        """
PROJECT_ROOT = Path.cwd()
DATA_DIR_ENV = "UNET_128_DATA_DIR"

# Override any of these with environment variables before launching Jupyter/VS Code,
# or edit the values directly in this cell.


def configured_path(env_var, default):
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else Path(default)


BASE_PATH = configured_path(DATA_DIR_ENV, PROJECT_ROOT / "Preprocessed-128")
TRAIN_SAR = configured_path("UNET_128_TRAIN_SAR_DIR", BASE_PATH / "train" / "sar")
TRAIN_FLOOD = configured_path("UNET_128_TRAIN_FLOOD_DIR", BASE_PATH / "train" / "flood")
VAL_SAR = configured_path("UNET_128_VAL_SAR_DIR", BASE_PATH / "val" / "sar")
VAL_FLOOD = configured_path("UNET_128_VAL_FLOOD_DIR", BASE_PATH / "val" / "flood")
TEST_SAR = configured_path("UNET_128_TEST_SAR_DIR", BASE_PATH / "test" / "sar")
TEST_FLOOD = configured_path("UNET_128_TEST_FLOOD_DIR", BASE_PATH / "test" / "flood")
METADATA_PATH = configured_path("UNET_128_METADATA_DIR", BASE_PATH / "metadata")

print(f"Using base dataset directory: {BASE_PATH}")

print("Checking data paths...")
for path_name, path in [
    ("Base", BASE_PATH),
    ("Train SAR", TRAIN_SAR),
    ("Train Flood", TRAIN_FLOOD),
    ("Val SAR", VAL_SAR),
    ("Val Flood", VAL_FLOOD),
    ("Test SAR", TEST_SAR),
    ("Test Flood", TEST_FLOOD),
    ("Metadata", METADATA_PATH),
]:
    if path.exists():
        if path.is_dir() and path_name not in {"Base", "Metadata"}:
            file_count = len(list(path.glob("*.tif")))
            print(f"{path_name}: {file_count} files")
        else:
            print(f"{path_name}: exists")
    else:
        print(f"{path_name}: not found at {path}")

norm_stats_path = METADATA_PATH / "normalization_stats.json"
if norm_stats_path.exists():
    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)
        """
    )


def replace_checkpoint_cell(cell: dict, checkpoint_dir_name: str) -> None:
    cell["source"] = to_source_block(
        """
OUTPUT_DIR_ENV = "UNET_128_OUTPUT_DIR"

# Override with UNET_128_OUTPUT_DIR if you want checkpoints somewhere else.
checkpoint_dir = Path(
    os.environ.get(
        OUTPUT_DIR_ENV,
        str(Path.cwd() / "artifacts" / "__CHECKPOINT_DIR_NAME__"),
    )
).expanduser()
checkpoint_dir.mkdir(parents=True, exist_ok=True)

callbacks_list = [
    callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / "best_model.h5"),
        monitor="val_dice_coefficient",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_dice_coefficient",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    ),
    callbacks.CSVLogger(
        str(checkpoint_dir / "training_log.csv"),
        append=False,
    ),
]

EPOCHS = 80
STEPS_PER_EPOCH = train_size // BATCH_SIZE
VALIDATION_STEPS = val_size // BATCH_SIZE

print(f"\\nTraining configuration:")
print(f"Epochs: {EPOCHS}")
print(f"Steps per epoch: {STEPS_PER_EPOCH}")
print(f"Validation steps: {VALIDATION_STEPS}")
print(f"Total training samples per epoch: {STEPS_PER_EPOCH * BATCH_SIZE}")
        """
        .replace("__CHECKPOINT_DIR_NAME__", checkpoint_dir_name)
    )


def clear_cell_runtime_state(cell: dict) -> None:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def parse_args() -> tuple[Path, Path]:
    source = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else discover_source_notebook()
    output = (
        Path(sys.argv[2]).expanduser()
        if len(sys.argv) > 2
        else Path(os.environ.get(OUTPUT_NOTEBOOK_ENV, DEFAULT_OUTPUT_NOTEBOOK)).expanduser()
    )
    return source, output


def discover_source_notebook() -> Path:
    configured = os.environ.get(SOURCE_NOTEBOOK_ENV)
    if configured:
        return Path(configured).expanduser()

    search_roots = [Path.cwd(), Path.cwd().parent]
    direct_candidates = [
        root / DEFAULT_SOURCE_FILENAME
        for root in search_roots
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Source notebook not found. Pass it as the first argument or set "
        f"{SOURCE_NOTEBOOK_ENV}."
    )


def extract_checkpoint_dir_name(source: str) -> str:
    marker = "checkpoint_dir = Path('/content/drive/MyDrive/"
    if marker not in source:
        return "unet_128_checkpoints"
    return source.split(marker, 1)[1].split("')", 1)[0]


def main() -> None:
    source_notebook, output_notebook = parse_args()
    notebook = json.loads(source_notebook.read_text())

    for cell in notebook.get("cells", []):
        clear_cell_runtime_state(cell)
        source = "".join(cell.get("source", []))

        if source.strip() == "!pip install rasterio":
            replace_dependency_cell(cell)
        elif "import tensorflow as tf" in source and "Python version" in source:
            replace_import_cell(cell)
        elif "gpu_info = !nvidia-smi" in source:
            replace_gpu_cell(cell)
        elif "from google.colab import drive" in source:
            replace_data_cell(cell)
        elif "checkpoint_dir = Path('/content/drive/MyDrive/" in source:
            replace_checkpoint_cell(cell, extract_checkpoint_dir_name(source))

    output_notebook.parent.mkdir(parents=True, exist_ok=True)
    output_notebook.write_text(json.dumps(notebook, indent=1))


if __name__ == "__main__":
    main()
