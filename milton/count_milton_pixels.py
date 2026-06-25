

import rasterio
import numpy as np
from rasterio.transform import xy

files = [
    "milton/raw_data/OPERA_DSWx-S1_BWTR_ChngMap_20241011-20241003.tif"
    "milton/raw_data/OPERA_DSWx-S1_BWTR_ChngMap_20241011-20241008.tif"




]


with rasterio.open(f"milton/filtered_masks_by_fp/coverage_filtered_mask_fp1.tif") as src:
    print("File: mask")
    print("CRS:", src.crs)
    print("Shape:", src.shape)
    print("Transform:", src.transform)
    print("Res:", src.res)
    print("Data:", src.dtypes)



with rasterio.open(f"milton/lee_uavsar/filtered_flcorr_06206_24075_007_241011_L090_CX_01_pauli.tif") as src:
        print("File: UAVSAR")
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Res:", src.res)
        print("Data:", src.dtypes)