import rasterio
import numpy as np
from rasterio.transform import xy
from rasterio.warp import reproject, Resampling

uavsar_files = [

    "flcorr_06206_24075_007_241011_L090_CX_01_pauli.tif"
    "peacer_19512_24075_004_241011_L090_CX_01_pauli.tif",
    "stjohn_17815_24075_002_241011_L090_CX_01_pauli.tif",
    "tampab_15102_24075_006_241011_L090_CX_01_pauli.tif",
    "tampab_33105_24075_005_241011_L090_CX_01_pauli.tif",
    "tampaf_06207_24075_010_241011_L090_CX_01_pauli.tif",
    "tampaf_24209_24075_011_241011_L090_CX_01_pauli.tif"
]

fp = 1

for uav_file in uavsar_files:

    mask_file = f"milton/reprojections/converted_aligned_mask_{uav_file}"
    with rasterio.open(mask_file) as src:
        mask = src.read(1)

    with rasterio.open(f"milton/lee_uavsar/filtered_{uav_file}") as src:
        uav = src.read()

    coverage = np.any(uav > 0, axis=0) #any valid data

    new_mask = mask.copy() #copy the og mask into the new one
 
    new_mask[~coverage] = 255 #mark any pixels with no data as invalid



    with rasterio.open(mask_file) as src:

        profile = src.profile #store og metadata
    #save new mask
    with rasterio.open(
        f"milton/filtered_masks_by_fp/coverage_filtered_mask_fp{fp}.tif",
        "w",
        **profile
    ) as dst:

        dst.write(new_mask, 1)

    fp += 1