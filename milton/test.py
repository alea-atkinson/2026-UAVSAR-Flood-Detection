import rasterio
import numpy as np

with rasterio.open("milton/tiles/fp1/masks/tile_00007.tif") as src:
    print("nodata value:", src.nodata)
    data = src.read(1)

    print("unique values:", np.unique(data))