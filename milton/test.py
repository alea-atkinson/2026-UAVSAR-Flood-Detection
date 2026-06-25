import rasterio
import numpy as np

with rasterio.open("milton/lee_uavsar/filtered_flcorr_06206_24075_007_241011_L090_CX_01_pauli.tif") as src:
    print(src.meta)