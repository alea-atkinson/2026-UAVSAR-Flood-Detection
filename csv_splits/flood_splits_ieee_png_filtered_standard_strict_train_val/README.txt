IEEE PNG filtered standard/strict leave-one-flight-path-out split files

This version filters the previously indexed fp1-fp7 TIF records to only tile_names that appear in both:
- new_data_cd PNG dataset: sar/ and ground_truth/
- new_data_lcc PNG dataset: sar/ and ground_truth/

Rationale: these are the tile names used in the IEEE-style PNG training datasets, while still keeping the fp1-fp7 TIF records so flight-path information and land-cover masks are available for the 2026 leave-one-flight-path-out experiment.

Counts:
- CD PNG tile names with both sar and ground_truth: 839
- LCC PNG tile names with both sar and ground_truth: 859
- Tile names appearing in both CD and LCC PNG datasets: 817
- Filtered fp1-fp7 TIF records with flood and land-cover masks: 1067

standard/: hold out one full flight path as test. The train_pool contains usable filtered records from the other flight paths, even if some tile names overlap with the held-out flight path.

strict_no_overlap/: hold out one full flight path as test. The train_pool contains usable filtered records from the other flight paths, but removes any records whose tile_name appears in the held-out test flight path.

For each held-out flight path, files include:
- heldout_fp*_test.csv
- heldout_fp*_train_pool.csv
- heldout_fp*_train.csv
- heldout_fp*_validation.csv

The train/validation split is deterministic with seed 42 and grouped by tile_name, so the same tile_name does not appear in both training and validation.
