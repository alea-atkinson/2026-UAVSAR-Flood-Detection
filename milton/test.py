import rasterio
import numpy as np

with rasterio.open("milton/tiles/fp1/uavsar/tile_00001.tif") as src:
    print(src.meta)