from pathlib import Path
import csv
import numpy as np
import rasterio

csv_file = Path("milton/csv_splits/png_test.csv")

valid_fracs = []
flood_fracs = []

no_flood = 0
high_flood = 0
with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:

        # Read SAR
        with rasterio.open(row["uavsar_path"]) as src:
            sar = src.read()[:3]

        # Valid SAR pixels (all three bands are NOT zero)
        sar_valid = ~(sar == 0).all(axis=0)

        # Read mask
        with rasterio.open(row["flood_mask_path"]) as src:
            mask = src.read(1)

            if src.nodata is None:
                mask_valid = np.ones(mask.shape, dtype=bool)
            else:
                mask_valid = mask != src.nodata

        # Final valid pixels
        valid = sar_valid & mask_valid

        valid_pixels = valid.sum()
        total_pixels = valid.size

        if valid_pixels == 0:
            continue

        flood_pixels = ((mask == 1) & valid).sum()

        if flood_pixels == 0 :
            no_flood += 1

        valid_fraction = valid_pixels / total_pixels

        

        flood_fraction = flood_pixels / valid_pixels

        if flood_fraction > 0.05:
            high_flood += 1

        valid_fracs.append(valid_fraction)
        flood_fracs.append(flood_fraction)

        print(
            f"{Path(row['uavsar_path']).name:35s} "
            f"Valid={100*valid_fraction:6.2f}% "
            f"Flood={100*flood_fraction:6.2f}%"
        )

print("\nDataset Summary")
print("-------------------------")
print(f"Average valid area : {100*np.mean(valid_fracs):.2f}%")
print(f"Average flood area : {100*np.mean(flood_fracs):.2f}%")
print(f"Minimum flood area : {100*np.min(flood_fracs):.2f}%")
print(f"Maximum flood area : {100*np.max(flood_fracs):.2f}%")
print(f"No flood count: {no_flood}")
print(f"High flood count: {high_flood}")