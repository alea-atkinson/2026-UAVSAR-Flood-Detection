import numpy as np
import rasterio
from rasterio.windows import Window


uav_file = "milton/raw_data/peacer_19512_24075_004_241011_L090_CX_01_pauli.tif"
mask_file = "milton/reprojections/converted_aligned_mask_peacer_19512_24075_004_241011_L090_CX_01_pauli.tif"

with rasterio.open(uav_file) as src:
    band1 = src.read(1)

rows, cols = np.where(band1 > 0)

print("Valid rows:", rows.min(), rows.max())
print("Valid cols:", cols.min(), cols.max())