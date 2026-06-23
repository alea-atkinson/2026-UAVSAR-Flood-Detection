"""import rasterio
import numpy as np

with rasterio.open("milton/OPERA_DSWx-S1_BWTR_ChngMap_20241011-20241003.tif") as src:
    arr = src.read(1)

vals, counts = np.unique(arr, return_counts=True)

for v, c in zip(vals, counts):
    print(v, c)"""


import rasterio
import numpy as np
from rasterio.transform import xy

files = [
    "milton/OPERA_DSWx-S1_BWTR_ChngMap_20241011-20241003.tif",
    "milton/OPERA_DSWx-S1_BWTR_ChngMap_20241011-20241008.tif",
    "milton/flcorr_06206_24075_007_241011_L090_CX_01_pauli.tif",
    "milton/peacer_19512_24075_004_241011_L090_CX_01_pauli.tif",
    "milton/stjohn_17815_24075_002_241011_L090_CX_01_pauli.tif",
    "milton/tampab_15102_24075_006_241011_L090_CX_01_pauli.tif",
    "milton/tampab_33105_24075_005_241011_L090_CX_01_pauli.tif",
    "milton/tampaf_06207_24075_010_241011_L090_CX_01_pauli.tif",
    "milton/tampaf_24209_24075_011_241011_L090_CX_01_pauli.tif"



]
"""
for f in files:
    with rasterio.open(f) as src:
        print("File: ", f)
        print("CRS:", src.crs)
        print("Bounds:", src.bounds)
        print("Shape:", src.shape)
        print("Transform:", src.transform)
        print(""repr(src.transform))"""

with rasterio.open("milton/tampab_15102_24075_006_241011_L090_CX_01_pauli.tif") as src:
        print(repr(src.transform))