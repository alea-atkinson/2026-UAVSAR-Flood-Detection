import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import os

uavsar_files = [

    "flcorr_06206_24075_007_241011_L090_CX_01_pauli.tif",
    "peacer_19512_24075_004_241011_L090_CX_01_pauli.tif",
    "stjohn_17815_24075_002_241011_L090_CX_01_pauli.tif",
    "tampab_15102_24075_006_241011_L090_CX_01_pauli.tif",
    "tampab_33105_24075_005_241011_L090_CX_01_pauli.tif",
    "tampaf_06207_24075_010_241011_L090_CX_01_pauli.tif",
    "tampaf_24209_24075_011_241011_L090_CX_01_pauli.tif"
]


tile_size = 256

fp = 1 #starting flight path

#for each flight path, open UAVSAR data and corresponding mask

for uav_file in uavsar_files:
    
    os.makedirs(f"milton/tiles/fp{fp}/uavsar", exist_ok=True)
    os.makedirs(f"milton/tiles/fp{fp}/masks", exist_ok=True)

    with rasterio.open(f"milton/raw_data/{uav_file}") as uav, rasterio.open(f"milton/reprojections/converted_aligned_mask_{uav_file}") as mask:

        tile_id = 0
        
        #tile by height, width
        for row in range(0, uav.height, tile_size):
            for col in range(0, uav.width, tile_size):
                #window size = 256
                window = Window(col, row, tile_size, tile_size)
                #read in data
                uavsar_tile = uav.read(window=window)
                mask_tile = mask.read(1, window=window)

                # skip incomplete edge tiles (not the correct size)
                if uavsar_tile.shape != (3, tile_size, tile_size):
                    continue

                if mask_tile.shape != (tile_size, tile_size):
                    continue
                #correct size, now count flood pixels
                flood_pixels = np.sum(mask_tile == 1)
                #if contains no flood pixels, skip it
                if flood_pixels == 0:
                    continue
                #preserve georeferencing data
                tile_transform = window_transform(
                window,
                uav.transform
            )
                #save valid tiles

                #uavsar
                uavsar_path = f"tiles/uavsar/tile_{tile_id:05d}.tif" #format tile name with id as string

                with rasterio.open(
                    uavsar_path,
                    "w",
                    driver="GTiff",
                    height=tile_size,
                    width=tile_size,
                    count=3,
                    dtype=uavsar_tile.dtype,
                    crs=uav.crs,
                    transform=tile_transform
                ) as dst:
                    dst.write(uavsar_tile)
                #mask
                mask_path = f"tiles/masks/tile_{tile_id:05d}.tif"

                with rasterio.open(
                    mask_path,
                    "w",
                    driver="GTiff",
                    height=tile_size,
                    width=tile_size,
                    count=1,
                    dtype=mask_tile.dtype,
                    crs=uav.crs,
                    transform=tile_transform
                ) as dst:

                    dst.write(mask_tile, 1) #write band 1
                tile_id+=1
    fp+=1 #for file naming