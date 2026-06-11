#!/usr/bin/env python3
import os
import csv
import shutil
from pathlib import Path

def main():
    #names = ["_test", "_train_pool", "_validation"]
    #for each flight path, read the csv file and move the flood mask and land cover mask to the new folder
    
    #make the new directories split by flight path
    for i in range (1, 8):
       Path ("Only_PNG_Data/fp" + str(i) +"/flood_mask").mkdir(parents=True, exist_ok=True)
       Path("Only_PNG_Data/fp" + str(i) +"/land_cover").mkdir(parents=True, exist_ok=True)
       Path("Only_PNG_Data/fp" + str(i) + "/UAVSAR").mkdir(parents=True, exist_ok=True)

    with open("flood_dataset_index_ieee_png_filtered.csv", mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # Maps fields to a dictionary
        for row in reader:
            #if tile is valid
            
            file_name = row['flood_mask_path']
            fp = row['flight_path']
            #start name after first / (this is the tile name)
            idx = file_name.find("/")
            short_name = file_name[idx + 1:]
            #old paths
            src_path1 = "Data/flood_change_mask_tiles/" + short_name
            src_path2 = "Data/land_cover_mask_tiles/" + short_name
            src_path3 = "Data/UAVSAR/" + short_name
            

            dst_path1 = "Only_PNG_Data/" + fp + "/flood_mask/" + short_name
            dst_path2 = "Only_PNG_Data/" + fp + "/land_cover/" + short_name
            dst_path3 = "Only_PNG_Data/" + fp + "/UAVSAR/" + short_name
            #copy to new destination
            shutil.copy(src_path1, dst_path1)
            shutil.copy(src_path2, dst_path2)
            shutil.copy(src_path3, dst_path3)


if __name__ == "__main__":
    print("Starting script")
    main()



