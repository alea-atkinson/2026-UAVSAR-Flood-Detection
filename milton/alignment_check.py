from pathlib import Path
import rasterio

mismatches = 0
for i in range(1, 8):

    uav_dir = Path(f"milton/tiles/fp{i}/uavsar")
    mask_dir = Path(f"milton/tiles/fp{i}/masks")

    for uav_file in uav_dir.glob("*.tif"):

        mask_file = mask_dir / uav_file.name

        if not mask_file.exists():
            print(f"Missing mask for {uav_file.name}")
            continue

        with rasterio.open(uav_file) as uavsar, \
             rasterio.open(mask_file) as mask:

            if(uavsar.crs != mask.crs):
                print(f"FP {i}: {uav_file.name} crs mismatch")
                print(f"UAV: {uavsar.crs}, Mask: {mask.crs}")
                mismatches += 1
            if(uavsar.transform != mask.transform):
                print(f"FP {i}: {uav_file.name} transform mismatch")
                print(f"UAV: {uavsar.transform}, Mask: {mask.transform}")
                mismatches += 1
            if(uavsar.width != mask.width):
                print(f"FP {i}: {uav_file.name} width mismatch")
                print(f"UAV: {uavsar.width}, Mask: {mask.width}")
                mismatches += 1
            if(uavsar.height != mask.height):
                print(f"FP {i}: {uav_file.name} height mismatch")
                print(f"UAV: {uavsar.height}, Mask: {mask.height}")
                mismatches += 1
    print(f"Total mismatches: {mismatches}")
