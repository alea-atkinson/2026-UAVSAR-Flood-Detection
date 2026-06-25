import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


def enhanced_lee(img, win_size=7, cu=0.523, cmax=1.73):
    img = img.astype(np.float32)

    local_mean = uniform_filter(img, win_size)

    local_mean_sq = uniform_filter(img**2, win_size)

    local_var = local_mean_sq - local_mean**2
    local_var = np.maximum(local_var, 0)

    local_std = np.sqrt(local_var)

    ci = local_std / (local_mean + 1e-8)

    result = np.zeros_like(img)

    # homogeneous
    mask1 = ci <= cu
    result[mask1] = local_mean[mask1]

    # heterogeneous
    mask2 = ci >= cmax
    result[mask2] = img[mask2]

    # intermediate
    mask3 = ~(mask1 | mask2)

    w = np.exp(
        -(ci[mask3] - cu) /
        (cmax - ci[mask3] + 1e-8)
    )

    result[mask3] = (
        local_mean[mask3] * w +
        img[mask3] * (1 - w)
    )

    return result




uavsar_files = [

    "tampab_15102_24075_006_241011_L090_CX_01_pauli.tif",
    "tampab_33105_24075_005_241011_L090_CX_01_pauli.tif",
    "tampaf_06207_24075_010_241011_L090_CX_01_pauli.tif",
    "tampaf_24209_24075_011_241011_L090_CX_01_pauli.tif"
]




for uav_file in uavsar_files:

  

    with rasterio.open(f"milton/raw_data/{uav_file}") as src:
        data = src.read()
        profile = src.profile

    filtered = np.zeros_like(data, dtype=np.float32) #needs to be converted to floating point for the lee filter

    for band in range(data.shape[0]):
        print(f"Filtering band {band+1}")
        filtered[band] = enhanced_lee(
            data[band],
            win_size=7
        )

    profile.update(dtype="float32") #matches florence datatype

    with rasterio.open(
        f"milton/lee_uavsar/filtered_{uav_file}",
        "w",
        **profile
    ) as dst:
        dst.write(filtered)