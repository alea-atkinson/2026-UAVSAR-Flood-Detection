import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import numpy as np

uavsar_files = [
    "flcorr_06206_24075_007_241011_L090_CX_01_pauli.tif",
    "peacer_19512_24075_004_241011_L090_CX_01_pauli.tif",
    "stjohn_17815_24075_002_241011_L090_CX_01_pauli.tif",
    "tampab_15102_24075_006_241011_L090_CX_01_pauli.tif",
    "tampab_33105_24075_005_241011_L090_CX_01_pauli.tif",
    "tampaf_06207_24075_010_241011_L090_CX_01_pauli.tif",
    "tampaf_24209_24075_011_241011_L090_CX_01_pauli.tif"
]

target_crs = "EPSG:32617" #need to use this crs because it is meter based insteead of degree based3
target_resolution = 20 #resolution used in IEEE AKA the NC data

count = 1

for uav_file in uavsar_files:

    print(f"Processing flight path {count}")

    #################################################
    # Reproject UAVSAR to 20 m UTM grid
    #################################################

    input_uav = f"milton/lee_uavsar/filtered_{uav_file}"
    output_uav = f"milton/uavsar_20_res/filtered_20m_fp{count}.tif"

    with rasterio.open(input_uav) as src:

        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution
        )

        profile = src.profile.copy()

        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height
        )

        with rasterio.open(output_uav, "w", **profile) as dst:

            for band in range(1, src.count + 1):

                destination = np.empty(
                    (height, width),
                    dtype=np.float32
                )

                reproject(
                    source=rasterio.band(src, band),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )

                dst.write(destination, band)

    #################################################
    # Use UAVSAR output as reference grid
    #################################################

    with rasterio.open(output_uav) as ref:

        target_transform = ref.transform
        target_width = ref.width
        target_height = ref.height
        target_crs_ref = ref.crs

    #################################################
    # Reproject flood mask to UAVSAR grid
    #################################################

    input_mask = (
        f"milton/filtered_masks_by_fp/"
        f"coverage_filtered_mask_fp{count}.tif"
    )

    output_mask = (
        f"milton/masks_20_res/"
        f"mask_20m_fp{count}.tif"
    )

    with rasterio.open(input_mask) as src:

        destination = np.full(
            (target_height, target_width),
            255,
            dtype=np.uint8
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs_ref,
            resampling=Resampling.nearest,
            src_nodata=255,
            dst_nodata=255
        )

        mask_profile = src.profile.copy()

        mask_profile.update(
            driver="GTiff",
            height=target_height,
            width=target_width,
            transform=target_transform,
            crs=target_crs_ref,
            count=1,
            dtype="uint8",
            nodata=255
        )

        with rasterio.open(output_mask, "w", **mask_profile) as dst:
            dst.write(destination, 1)

    count += 1

print("Finished.")