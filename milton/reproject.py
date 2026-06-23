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


opera_files = [
    "milton/OPERA_DSWx-S1_BWTR_ChngMap_20241011-20241003.tif",
    "milton/OPERA_DSWx-S1_BWTR_ChngMap_20241011-20241008.tif"
]

#operate on one ground truth mask at a time
#reproject the opera file onto the uavsar file
with rasterio.open(opera_files[0]) as op:
    #each uavsar file
    for uav_file in uavsar_files:
        file_name= "milton/" + uav_file
        with rasterio.open(file_name) as uav:
            #make empty array. -9999 means no data
            aligned_mask = np.full(
                (uav.height, uav.width),
                -9999,
                dtype=np.float32
            )
            #use rasterio reprojection to align
            reproject(
                source=rasterio.band(op, 1),
                destination=aligned_mask, #write into empty array

                src_transform=op.transform, #inform rasterio about the source (opera) information
                src_crs=op.crs,

                dst_transform=uav.transform, #inform ratserio about the destination (uavsar) information
                dst_crs=uav.crs,

                resampling=Resampling.nearest
            )
            name = f"aligned_mask_{uav_file}"
            with rasterio.open(
                name,
                "w",
                driver="GTiff",
                height=uav.height,
                width=uav.width,
                count=1,
                dtype=aligned_mask.dtype,
                crs=uav.crs,
                transform=uav.transform
            ) as dst:
                dst.write(aligned_mask, 1) 


for uav_file in uavsar_files:
    with rasterio.open(f"milton/reprojections/aligned_mask_{uav_file}") as src:
        mask = src.read(1)
        profile = src.profile

    # Convert flood loss label
    mask[mask == -1] = 0 #water loss will be treated as non flooded
    #no data values will be ignored/thrown out
    nodata = -3.4028235e+38
    mask[np.isclose(mask, nodata)] = 255 #with specific tolerance

    # Save new mask
    with rasterio.open(f"milton/reprojections/converted_aligned_mask_{uav_file}", "w", **profile) as dst:
        dst.write(mask, 1)               