# Random Split Architecture Comparison

All rows use the same threshold-sweep process: sweep thresholds from 0.05 to 0.95 on the validation split, select the threshold with the best validation Dice, then evaluate the test split once at that selected threshold.

| Model | Split | Selected threshold | Val Dice | Val IoU | Val precision | Val recall | Test Dice | Test IoU | Test precision | Test recall | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet baseline | Random tile-name-group split | 0.25 | 0.6707 | 0.5046 | 0.5833 | 0.7890 | 0.5918 | 0.4203 | 0.4994 | 0.7263 | `outputs/random_split/threshold_sweep_random_splits/jz_unet_baseline_random_tile_name_group.csv` |
| UNet++ | Random tile-name-group split | 0.50 | 0.6706 | 0.5044 | 0.6977 | 0.6455 | 0.6489 | 0.4803 | 0.6398 | 0.6584 | `outputs/random_split/threshold_sweep_random_splits/jz_unetpp_random_tile_name_group.csv` |
| Attention UNet | Random tile-name-group split | 0.50 | 0.6718 | 0.5058 | 0.7233 | 0.6272 | 0.6406 | 0.4713 | 0.6668 | 0.6165 | `outputs/random_split/threshold_sweep_random_splits/jz_attention_unet_random_tile_name_group.csv` |
| Tuned UNet | Random tile-name-group split | 0.50 | 0.6919 | 0.5290 | 0.7358 | 0.6530 | 0.6858 | 0.5218 | 0.6752 | 0.6967 | `outputs/random_split/threshold_sweep_random_splits/jz_tuned_unet_random_tile_name_group.csv` |
| UNet baseline | Random record split | 0.35 | 0.6708 | 0.5047 | 0.6577 | 0.6844 | 0.6555 | 0.4875 | 0.6007 | 0.7212 | `outputs/random_split/threshold_sweep_random_splits/jz_unet_baseline_random_record.csv` |
| UNet++ | Random record split | 0.35 | 0.6561 | 0.4882 | 0.6806 | 0.6334 | 0.6585 | 0.4909 | 0.6590 | 0.6580 | `outputs/random_split/threshold_sweep_random_splits/jz_unetpp_random_record.csv` |
| Attention UNet | Random record split | 0.50 | 0.6454 | 0.4765 | 0.6843 | 0.6107 | 0.6523 | 0.4841 | 0.6440 | 0.6609 | `outputs/random_split/threshold_sweep_random_splits/jz_attention_unet_random_record.csv` |
| Tuned UNet | Random record split | 0.50 | 0.6882 | 0.5246 | 0.7662 | 0.6246 | 0.6974 | 0.5354 | 0.7467 | 0.6542 | `outputs/random_split/threshold_sweep_random_splits/jz_tuned_unet_random_record.csv` |
