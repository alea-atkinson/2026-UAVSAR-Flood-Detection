#!/usr/bin/env python3
import os
import csv
import shutil
from pathlib import Path

def main():
    names = ["test", "train", "_validation"]
    rule = "standard"
    #for each flight path, read the csv file and move the flood mask and land cover mask to the new folder
    
    #make the new directories split by flight path
    for i in range (1, 8):
        for j in range(0, 3):
            #holdout directories
            Path ("PNG_Data/" +rule+"/" + names[j] + "/fp" + str(i) +"/flood_mask").mkdir(parents=True, exist_ok=True)
            Path("PNG_Data/" +rule+"/" + names[j] + "/fp" + str(i) +"/land_cover").mkdir(parents=True, exist_ok=True)
            Path("PNG_Data/" +rule+"/" + names[j] + "/fp" + str(i) + "/UAVSAR").mkdir(parents=True, exist_ok=True)

            #no holdout directories
            Path("PNG_Data/" +rule+"/" + names[j] + "/no_holdout" + "/UAVSAR").mkdir(parents=True, exist_ok=True)
            Path("PNG_Data/" +rule+"/" + names[j] + "/no_holdout" + "/flood_mask").mkdir(parents=True, exist_ok=True)
            Path("PNG_Data/" +rule+"/" + names[j] + "/no_holdout" + "/land_cover").mkdir(parents=True, exist_ok=True) 
            
            with open("csv_splits/flood_splits_ieee_png_filtered_standard_strict_train_val/standard/heldout_fp" + str(i) + "_" +names[j] + ".csv", mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)  # Maps fields to a dictionary
                for row in reader:
                    #if tile is valid
                    
                    file_name = row['flood_mask_path']
                    #start name after first / (this is the tile name)
                    idx = file_name.find("/")
                    short_name = file_name[idx + 1:]
                    #old paths
                    src_path1 = "2025_Tile_Data/flood_change_mask_tiles/" + short_name
                    src_path2 = "2025_Tile_Data/land_cover_mask_tiles/" + short_name
                    src_path3 = "2025_Tile_Data/UAVSAR/" + short_name
                    
                    #split by holdout

                    dst_path1 = "PNG_Data/" +  rule + "/" + names[j] + "/fp" + str(i) + "/flood_mask/" + short_name
                    dst_path2 = "PNG_Data/" + rule + "/" + names[j] + "/fp" + str(i) + "/land_cover/" + short_name
                    dst_path3 = "PNG_Data/" + rule + "/" + names[j] + "/fp" + str(i) + "/UAVSAR/" + short_name
                    
                    #no holdout paths
                    
                    dst_path4 = "PNG_Data/" + rule + "/" + names[j] + "/no_holdout" + "/flood_mask/" + short_name
                    dst_path5 = "PNG_Data/" + rule + "/" + names[j] + "/no_holdout" + "/land_cover/" + short_name
                    dst_path6 = "PNG_Data/" + rule + "/" + names[j] + "/no_holdout" + "/UAVSAR/" + short_name

                    #copy to new destination
                    os.link(src_path1, dst_path1)
                    os.link(src_path2, dst_path2)
                    os.link(src_path3, dst_path3)
                    #copy to no holdout if it doesn't already exist there
                    
                    if not os.path.exists(dst_path4): #only need to check one of the three since if one exists, they all exist
                        os.link(src_path1, dst_path4)
                        os.link(src_path2, dst_path5)
                        os.link(src_path3, dst_path6)


if __name__ == "__main__":
    print("Starting script")
    main()



