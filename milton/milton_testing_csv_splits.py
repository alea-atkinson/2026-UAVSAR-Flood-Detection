from pathlib import Path
import csv
import random

# -----------------------------
# Directories
# -----------------------------

data_root = Path("2025_Tile_Data")

train_uavsar = data_root / "UAVSAR"
train_masks = data_root / "flood_change_mask_tiles"

test_uavsar = "milton/tiles"
test_masks = data_root / "test/masks"

output_dir = Path("csv_splits")

validation_fraction = 0.20
random_seed = 42

# -----------------------------
# Function to match images/masks
# -----------------------------

def make_pairs(image_dir, mask_dir):

    pairs = []

    for image in sorted(image_dir.glob("*.tif")):

        mask = mask_dir / image.name

        if not mask.exists():
            print(f"No mask found for {image.name}")
            continue

        pairs.append({
            "uavsar_path": str(image),
            "flood_mask_path": str(mask)
        })
    return pairs

# -----------------------------
# Build datasets
# -----------------------------


#combined flight path directory
train_val_pairs = make_pairs(
    Path("2025_Tile_Data/UAVSAR"), 
    Path("2025_Tile_Data/flood_change_mask_tiles"))

test_pairs=[]

#run once for every flight path
for i in range(1, 8):

    fp_pairs = make_pairs(
        Path(f"milton/tiles/fp{i}/uavsar"),
        Path(f"milton/tiles/fp{i}/masks")
    )

    test_pairs.extend(fp_pairs)

print(f"Training/Validation tiles: {len(train_val_pairs)}")
print(f"Testing tiles: {len(test_pairs)}")

# -----------------------------
# Random train/validation split
# -----------------------------

random.seed(random_seed)
random.shuffle(train_val_pairs)

n_train = int(len(train_val_pairs) * (1 - validation_fraction))

train_pairs = train_val_pairs[:n_train]
validation_pairs = train_val_pairs[n_train:]

print(f"Training: {len(train_pairs)}")
print(f"Validation: {len(validation_pairs)}")
print(f"Testing: {len(test_pairs)}")

# -----------------------------
# Write CSVs
# -----------------------------

output_dir.mkdir(parents=True, exist_ok=True)

def write_csv(filename, rows):

    with open(output_dir / filename, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["uavsar_path", "flood_mask_path"]
        )

        writer.writeheader()
        writer.writerows(rows)

write_csv("train.csv", train_pairs)
write_csv("validation.csv", validation_pairs)
write_csv("test.csv", test_pairs)

print("Finished writing CSV files.")