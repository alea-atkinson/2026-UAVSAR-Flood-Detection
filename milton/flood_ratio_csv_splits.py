from pathlib import Path
import csv
import numpy as np
import rasterio

# -----------------------------
# Input / Output
# -----------------------------

input_csv = Path("milton/train_milton_csv_splits/test.csv") #testing on all milton tiles
output_csv = Path("milton/flood_bin_csv_splits/florence_test_50.csv") 

# Keep only tiles with >theshold flood
lower = 0.50
upper = 2

high_flood_tiles = []

# -----------------------------
# Filter tiles
# -----------------------------

with open(input_csv, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:

        # Read SAR
        with rasterio.open(row["uavsar_path"]) as src:
            sar = src.read()[:3]

        # Valid SAR pixels
        sar_valid = ~(sar == 0).all(axis=0)

        # Read mask
        with rasterio.open(row["flood_mask_path"]) as src:
            mask = src.read(1)

            if src.nodata is None:
                mask_valid = np.ones(mask.shape, dtype=bool)
            else:
                mask_valid = mask != src.nodata

        valid = sar_valid & mask_valid

        valid_pixels = valid.sum()

        if valid_pixels == 0:
            continue

        flood_pixels = ((mask == 1) & valid).sum()

        flood_fraction = flood_pixels / valid_pixels

        if flood_fraction >= lower and flood_fraction < upper:
            high_flood_tiles.append(row)

# -----------------------------
# Save filtered CSV
# -----------------------------

output_csv.parent.mkdir(parents=True, exist_ok=True)

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["uavsar_path", "flood_mask_path"]
    )

    writer.writeheader()
    writer.writerows(high_flood_tiles)

print(f"Saved {len(high_flood_tiles)} high-flood tiles.")
print(f"Output CSV: {output_csv}")