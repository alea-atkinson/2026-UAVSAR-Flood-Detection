Standard vs strict leave-one-flight-path-out split files

Each split uses only records with both a flood/change mask and a land-cover mask for supervised training/evaluation. Records missing flood/change masks are retained in the dataset index but excluded from these split files.

standard/: hold out one full flight path as test. The train_pool contains usable records from the other flight paths, even if some tile names overlap with the held-out flight path.

strict_no_overlap/: hold out one full flight path as test. The train_pool contains usable records from the other flight paths, but removes any records whose tile_name appears in the held-out test flight path. This is intended to reduce possible spatial/location leakage.

For each held-out flight path, files include:
- heldout_fp*_test.csv: usable records from the held-out flight path.
- heldout_fp*_train_pool.csv: all usable non-test records after applying the standard or strict rule.
- heldout_fp*_train.csv: training subset created from train_pool after the test path is set aside.
- heldout_fp*_validation.csv: validation subset created from train_pool after the test path is set aside.

The train/validation split is deterministic with seed 42 and is grouped by tile_name so the same tile_name does not appear in both training and validation.
