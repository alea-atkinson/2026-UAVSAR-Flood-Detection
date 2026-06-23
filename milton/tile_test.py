import numpy as np
import rasterio
from rasterio.windows import Window


uav_file = "milton/raw_data/peacer_19512_24075_004_241011_L090_CX_01_pauli.tif"
mask_file = "milton/reprojections/converted_aligned_mask_peacer_19512_24075_004_241011_L090_CX_01_pauli.tif"

with rasterio.open(mask_file) as src:
    mask = src.read(1)

rows, cols = np.where(mask == 1)

idx = len(rows) // 2

row = rows[idx]
col = cols[idx]

print("Middle flood pixel:", row, col)

tile_size = 256

with rasterio.open(uav_file) as uav, rasterio.open(mask_file) as mask:

    # First tile (top-left corner)
    window = Window(
    col - 128,
    row - 128,
    256,
    256
)

    image_tile = uav.read(window=window)
    mask_tile = mask.read(1, window=window)

    tile_transform = rasterio.windows.transform(
        window,
        uav.transform
    )

    # Save image tile
    with rasterio.open(
        "test_uav_tile.tif",
        "w",
        driver="GTiff",
        height=tile_size,
        width=tile_size,
        count=3,
        dtype=image_tile.dtype,
        crs=uav.crs,
        transform=tile_transform
    ) as dst:
        dst.write(image_tile)
    print(image_tile.shape)
    print(np.min(image_tile))
    print(np.max(image_tile))
    # Save mask tile
    with rasterio.open(
        "test_mask_tile.tif",
        "w",
        driver="GTiff",
        height=tile_size,
        width=tile_size,
        count=1,
        dtype=mask_tile.dtype,
        crs=mask.crs,
        transform=tile_transform
    ) as dst:
        dst.write(mask_tile, 1)

        print(uav.shape)
        print(mask.shape)

        print(uav.transform)
        print(mask.transform)

        print("Flood pixel:", row, col)

        print("Band1 value at flood pixel:", band1[row, col])

