# U-Net Baseline and Tuning Experiment Inventory

This inventory covers **only** the plain U-Net baseline and the follow-up hyperparameter / ablation experiments run by JJ. It does not include land-cover error analysis, probability/uncertainty maps, Alea's tuned scripts, or whole-team project summaries.

All experiments use: **IEEE PNG-filtered strict no-overlap leave-one-flight-path-out splits**.

## B. Motivation and Design Rationale

**Original baseline** (`train_unet_baseline.py`): establishes a reference point — plain U-Net, BCE loss, Adam, no augmentation, checkpoint selected by val_loss. All other variants change exactly one or two things at a time relative to this baseline.

**Threshold sweep** (`threshold_sweep_baseline.py`, `threshold_sweep_single_checkpoint.py`): checks whether the default 0.5 cutoff is too conservative. Sweeps thresholds 0.05–0.95 in steps of 0.05 on the validation split only, selects the threshold that maximizes global pixel-level Dice, then applies it once to the held-out test split. These are global pixel-level metrics — not batch-averaged — and are NOT comparable with the default-0.5 training-script metrics.

**BCE + Dice loss** (`train_unet_experiment.py` with `--loss bce_dice`): tests whether adding a soft Dice term to the objective directly optimizes segmentation overlap rather than relying on the threshold sweep to recover it.

**Augmentation** (`--augment` flag): tests whether simple spatial transforms (random horizontal flip, vertical flip, 90° rotation) improve generalization to the unseen flight path.

**Validation-Dice checkpointing** (`--save-best-by val_dice`): tests whether saving the checkpoint with the highest validation Dice (instead of lowest validation loss) is a better surrogate for test-time segmentation quality.

**Weighted BCE** (`train_unet_weighted_experiment.py` with `--loss weighted_bce`): tests whether up-weighting the rare positive (flood) class via `pos_weight = negative_pixels / positive_pixels` (estimated from training masks before training) helps the model learn flood boundaries better.

**Architecture extensions** (`train_unet_arch_experiment.py` with `--arch attention_unet` or `--arch unet_plus_plus`): reported in a separate section. These change the model architecture, not just hyperparameters, so they are not direct ablations of the baseline and should not be compared as if they are.

## C. Master Table of Experiments (Plain U-Net Only)

| Run name | Script | FP | Loss | Ckpt sel | Aug | Epochs | LR | BS | Base-ch | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| filtered_strict_fp1_unet_20epochs | train_unet_baseline.py | fp1 | bce | val_loss | False | 20 | 1e-3 | 8 | 32 | fp1-fp7 |
| filtered_strict_fp2_unet_20epochs | train_unet_baseline.py | fp2 | bce | val_loss | False | 20 | 1e-3 | 8 | 32 | fp1-fp7 |
| filtered_strict_fp3_unet_20epochs | train_unet_baseline.py | fp3 | bce | val_loss | False | 20 | 1e-3 | 8 | 32 | fp1-fp7 |
| filtered_strict_fp4_unet_20epochs | train_unet_baseline.py | fp4 | bce | val_loss | False | 20 | 1e-3 | 8 | 32 | fp1-fp7 |
| filtered_strict_fp5_unet_20epochs | train_unet_baseline.py | fp5 | bce | val_loss | False | 20 | 1e-3 | 8 | 32 | fp1-fp7 |
| filtered_strict_fp6_unet_20epochs | train_unet_baseline.py | fp6 | bce | val_loss | False | 20 | 1e-3 | 8 | 32 | fp1-fp7 |
| filtered_strict_fp7_unet_20epochs | train_unet_baseline.py | fp7 | bce | val_loss | False | 20 | 1e-3 | 8 | 32 | fp1-fp7 |
| experiment_fp1_bce_dice_aug_20epochs | train_unet_experiment.py | fp1 | bce_dice | val_loss | True | 20 | 1e-3 | 8 | 32 | fp1 only |
| experiment_fp1_bce_dice_noaug_20epochs | train_unet_experiment.py | fp1 | bce_dice | val_loss | False | 20 | 1e-3 | 8 | 32 | fp1 only |
| experiment_fp1_bce_valdice_noaug_20epochs | train_unet_experiment.py | fp1 | bce | val_dice | False | 20 | 1e-3 | 8 | 32 | fp1 only |
| experiment_fp1_weighted_bce_valdice_20epochs | train_unet_weighted_experiment.py | fp1 | weighted_bce | val_dice | not_in_name | 20 | 1e-3 | 8 | 32 | fp1 only |

## D. Results Table

> **Important — two metric types are used:**
> - `default_0.5`: computed during training, **batch-averaged** Dice/IoU at fixed threshold 0.5. Not pixel-global.
> - `threshold_sweep_global_pixel`: computed by threshold sweep scripts over **all pixels** of the split at once. Threshold was selected on the validation set only, then applied once to the test set.

### D.1 Baseline (fp1–fp7) — Default 0.5 (batch-averaged) vs. Threshold-Sweep (global pixel)

| FP | Default 0.5 Dice | Default 0.5 IoU | Selected Thresh | Val Dice (sweep) | Test Dice (sweep) | Test IoU (sweep) | Test Prec (sweep) | Test Recall (sweep) |
|---|---|---|---|---|---|---|---|---|
| fp1 | 0.3618 | 0.2415 | 0.25 | 0.6752 | 0.5205 | 0.3518 | 0.4597 | 0.5998 |
| fp2 | 0.6129 | 0.4771 | 0.35 | 0.6427 | 0.7445 | 0.5930 | 0.7447 | 0.7444 |
| fp3 | 0.5155 | 0.3741 | 0.35 | 0.6553 | 0.6264 | 0.4560 | 0.7483 | 0.5387 |
| fp4 | 0.3742 | 0.2618 | 0.50 | 0.6774 | 0.4657 | 0.3035 | 0.5288 | 0.4160 |
| fp5 | 0.5506 | 0.4012 | 0.25 | 0.6343 | 0.6725 | 0.5066 | 0.6109 | 0.7480 |
| fp6 | 0.4165 | 0.2928 | 0.30 | 0.6881 | 0.6008 | 0.4294 | 0.6940 | 0.5297 |
| fp7 | 0.2709 | 0.1764 | 0.30 | 0.6779 | 0.5570 | 0.3860 | 0.6799 | 0.4718 |

### D.2 Experiment Variants — fp1 Only

| Run | Loss | Ckpt sel | Aug | Default 0.5 Dice | Default 0.5 IoU | Selected Thresh | Val Dice (sweep) | Test Dice (sweep) | Test IoU (sweep) | Test Prec | Test Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| experiment_fp1_bce_dice_aug_20epochs | bce_dice | val_loss | True | 0.4682 | 0.3340 | 0.40 | 0.6853 | 0.5094 | 0.3418 | 0.4274 | 0.6305 |
| experiment_fp1_bce_dice_noaug_20epochs | bce_dice | val_loss | False | 0.4426 | 0.3085 | 0.55 | 0.6898 | 0.4879 | 0.3226 | 0.3815 | 0.6767 |
| experiment_fp1_bce_valdice_noaug_20epochs | bce | val_dice | False | 0.4206 | 0.3003 | 0.35 | 0.6907 | 0.4950 | 0.3289 | 0.4141 | 0.6152 |
| experiment_fp1_weighted_bce_valdice_20epochs | weighted_bce | val_dice | not_in_name | 0.3997 | 0.2701 | 0.65 | 0.6979 | 0.4914 | 0.3257 | 0.3873 | 0.6719 |

## E. Plain-English Interpretation

**Best plain U-Net variant (sweep metric):** The baseline with threshold sweep achieves the highest test Dice on fp2 (0.7445) and lowest on fp7 (0.4657). Average across fp1–fp7 at the sweep threshold: 0.5982.

**Threshold sweep effect:** Comparing the baseline default-0.5 batch-averaged Dice with the threshold-sweep global-pixel Dice reveals the 0.5 threshold is often suboptimal. For example, fp1 improves from 0.3618 (default 0.5) to 0.5205 (sweep at 0.25). fp2 improves from 0.6129 to 0.7445 (sweep at 0.35). These are not directly comparable numbers (different metric computation methods), but they confirm that 0.5 is too conservative for this dataset — the model's raw output is systematically underconfident about flood pixels.

**Augmentation (fp1 only):** Comparing `bce_dice` with augmentation vs. without augmentation, both at the sweep threshold: aug=True gives test Dice 0.5094 vs. aug=False gives 0.4879. The difference is small and both runs are on fp1 only — insufficient to conclude augmentation clearly helps. More flight paths need to be run to confirm.

**BCE + Dice loss (fp1 only):** Switching from BCE to BCE+Dice improved the sweep threshold selection (0.40 or 0.55 vs. 0.35 for BCE) but the test Dice at the sweep threshold is similar to or slightly below the BCE val_dice checkpoint variant (0.488–0.509 vs. 0.495). No clear winner on fp1 alone — needs more flight paths.

**Weighted BCE (fp1 only):** The weighted_bce run selects a notably higher threshold (0.65) and achieves test Dice 0.4914 (sweep). This is comparable to other variants but not clearly better. The higher selected threshold suggests the model was pushed toward predicting higher probabilities for flood pixels, as expected. The default-0.5 test Dice (0.3997) is lower than the baseline default-0.5 (0.3618 on fp1). Not yet a clear win for fp1 alone.

**What still needs to be compared more carefully:**
- All experiment variants (BCE+Dice, weighted_bce, augmentation) have only been run on fp1. Results from a single heldout flight path are insufficient to rank variants reliably.
- The baseline has full fp1–fp7 coverage; experiment variants are fp1 only. Cross-fp comparison is therefore not yet possible.
- The metric type distinction (batch-average vs. global-pixel) must be respected — the default-0.5 numbers and the sweep numbers are not interchangeable.

## F. Architecture Extensions (Not Plain U-Net Tuning)

The runs below use non-standard architectures (Attention U-Net, U-Net++) from `train_unet_arch_experiment.py`. They are reported separately because they change the model architecture, making them incomparable to baseline hyperparameter ablations.

| Run | Arch | FP | Loss | Ckpt sel | Default 0.5 Dice | Default 0.5 IoU | Sweep Thresh | Test Dice (sweep) | Test IoU (sweep) |
|---|---|---|---|---|---|---|---|---|---|
| attention_unet_fp1_bce_valdice_20epochs | attention_unet | fp1 | bce | val_dice | 0.4360 | 0.3051 | 0.45 | 0.4696 | 0.3068 |
| attention_unet_fp2_bce_valdice_20epochs | attention_unet | fp2 | bce | val_dice | 0.6509 | 0.5192 | 0.45 | 0.7397 | 0.5869 |
| attention_unet_fp3_bce_valdice_20epochs | attention_unet | fp3 | bce | val_dice | 0.5277 | 0.3917 | 0.35 | 0.6241 | 0.4536 |
| attention_unet_fp4_bce_valdice_20epochs | attention_unet | fp4 | bce | val_dice | 0.4229 | 0.3030 | 0.50 | 0.4953 | 0.3292 |
| attention_unet_fp5_bce_valdice_20epochs | attention_unet | fp5 | bce | val_dice | 0.5703 | 0.4309 | 0.50 | 0.6244 | 0.4539 |
| attention_unet_fp6_bce_valdice_20epochs | attention_unet | fp6 | bce | val_dice | 0.5279 | 0.3910 | 0.50 | 0.6245 | 0.4540 |
| attention_unet_fp7_bce_valdice_20epochs | attention_unet | fp7 | bce | val_dice | 0.3673 | 0.2613 | 0.40 | 0.5490 | 0.3784 |
| unetpp_fp1_bce_valdice_20epochs | unet_plus_plus | fp1 | bce | val_dice | 0.4289 | 0.2972 | 0.35 | 0.4761 | 0.3124 |
| unetpp_fp2_bce_valdice_20epochs | unet_plus_plus | fp2 | bce | val_dice | 0.6405 | 0.5088 | 0.40 | 0.7492 | 0.5990 |
| unetpp_fp3_bce_valdice_20epochs | unet_plus_plus | fp3 | bce | val_dice | 0.5495 | 0.4064 | 0.45 | 0.6082 | 0.4370 |
| unetpp_fp4_bce_valdice_20epochs | unet_plus_plus | fp4 | bce | val_dice | 0.3839 | 0.2594 | 0.55 | 0.4615 | 0.3000 |
| unetpp_fp5_bce_valdice_20epochs | unet_plus_plus | fp5 | bce | val_dice | 0.5679 | 0.4295 | 0.35 | 0.6174 | 0.4465 |
| unetpp_fp6_bce_valdice_20epochs | unet_plus_plus | fp6 | bce | val_dice | 0.4973 | 0.3648 | 0.40 | 0.6333 | 0.4634 |
| unetpp_fp7_bce_valdice_20epochs | unet_plus_plus | fp7 | bce | val_dice | 0.3787 | 0.2712 | 0.40 | 0.5604 | 0.3893 |

## G. Missing or Unclear Data

- No critical missing data detected.

**Additional known gaps:**
- Augmentation status for `experiment_fp1_weighted_bce_valdice_20epochs` not encoded in run name (default=False assumed).
- Augmentation status for arch extension runs not encoded in run names.
- Precision and recall are not available for default-0.5 training-script evaluations (only batch-averaged Dice and IoU).
- Baseline train/val history CSVs (`filtered_strict_fp{N}_unet_20epochs_metrics.csv`) do not contain a test row — test metrics at 0.5 came from terminal output, collected into `filtered_strict_20epoch_test_summary.csv`.
- Experiment variants (BCE+Dice, weighted_bce, augmentation) have been run on fp1 only. fp2–fp7 results do not exist yet.

## H. Script Default Hyperparameters (Extracted from Source)

| Script | Epochs | LR | Batch | Base-ch | Seed | Optimizer | Loss options | Ckpt-sel options | Arch options | Augment flag |
|---|---|---|---|---|---|---|---|---|---|---|
| train_unet_baseline.py | 20 | 1e-3 | 8 | 32 | 42 | Adam | bce (fixed) | val_loss, val_dice | unknown | False |
| train_unet_experiment.py | 20 | 1e-3 | 8 | 32 | 42 | Adam | bce, dice, bce_dice | val_loss, val_dice | unknown | True |
| train_unet_weighted_experiment.py | 20 | 1e-3 | 8 | 32 | 42 | Adam | bce, dice, bce_dice, weighted_bce | val_loss, val_dice | unknown | True |
| train_unet_arch_experiment.py | 20 | 1e-3 | 8 | 32 | 42 | Adam | bce, dice, bce_dice | val_loss, val_dice | unet, attention_unet, unet_plus_plus | True |
| threshold_sweep_baseline.py | unknown | unknown | 8 | unknown | unknown | unknown | unknown | unknown | unknown | False |
| threshold_sweep_single_checkpoint.py | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | False |
