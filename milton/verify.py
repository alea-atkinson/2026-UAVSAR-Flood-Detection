import rasterio
import numpy as np
from rasterio.transform import xy
from rasterio.warp import reproject, Resampling

uavsar_files = [

    "flcorr_06206_24075_007_241011_L090_CX_01_pauli.tif",
    "peacer_19512_24075_004_241011_L090_CX_01_pauli.tif",
    "stjohn_17815_24075_002_241011_L090_CX_01_pauli.tif",
    "tampab_15102_24075_006_241011_L090_CX_01_pauli.tif",
    "tampab_33105_24075_005_241011_L090_CX_01_pauli.tif",
    "tampaf_06207_24075_010_241011_L090_CX_01_pauli.tif",
    "tampaf_24209_24075_011_241011_L090_CX_01_pauli.tif"
]


for uav_file in uavsar_files:
    print("File: ", uav_file)
    with rasterio.open(f"milton/reprojections/aligned_mask_{uav_file}") as src:
       mask = src.read(1)

    vals, counts = np.unique(mask, return_counts=True)

    for v, c in zip(vals, counts):
        print(v, c)