# Alea-Tuned U-Net vs. Original U-Net Baseline

**Setup:** Alea's tuned plain U-Net (FocalDiceLoss, AdamW, lr=9.327e-05, wd=6.088e-06, batch_size=16, base_channels=32, epochs=20) trained on the IEEE PNG-filtered strict no-overlap leave-one-flight-path-out splits.

**Comparison baseline:** original plain U-Net (BCEWithLogitsLoss, Adam, lr=1e-3, wd=0, batch_size=8, base_channels=32, epochs=20, val_loss checkpoint) with validation-selected threshold sweep.

> All metrics are global pixel-level, computed by threshold sweep (threshold selected on validation Dice only, applied once to held-out test split).

## Alea-Tuned Results per Flight Path

| FP | Sel thresh | Val Dice | Test Dice | Test IoU | Test Prec | Test Recall |
|---|---|---|---|---|---|---|
| fp1 | 0.50 | 0.7084 | 0.5152 | 0.3470 | 0.4221 | 0.6611 |
| fp2 | 0.75 | 0.7045 | 0.7824 | 0.6426 | 0.7849 | 0.7799 |
| fp3 | 0.60 | 0.7281 | 0.6283 | 0.4580 | 0.8197 | 0.5093 |
| fp4 | 0.55 | 0.7215 | 0.5445 | 0.3741 | 0.5834 | 0.5104 |
| fp5 | 0.45 | 0.6763 | 0.6182 | 0.4474 | 0.4737 | 0.8896 |
| fp6 | 0.65 | 0.7441 | 0.6419 | 0.4727 | 0.7830 | 0.5439 |
| fp7 | 0.50 | 0.7169 | 0.5172 | 0.3488 | 0.5901 | 0.4603 |
| **mean (n=7)** | — | 0.7143 | **0.6068** | 0.4415 | 0.6367 | 0.6221 |
| std | — | 0.0212 | 0.0940 | 0.1031 | 0.1605 | 0.1610 |

## Head-to-Head Comparison: Alea Tuned vs. Original U-Net

| FP | Alea Dice | Orig Dice | Delta | Verdict | Alea Prec | Orig Prec | Alea Recall | Orig Recall |
|---|---|---|---|---|---|---|---|---|
| fp1 | 0.5152 | 0.5205 | -0.0053 | worse | 0.4221 | 0.4597 | 0.6611 | 0.5998 |
| fp2 | 0.7824 | 0.7445 | +0.0379 | **better** | 0.7849 | 0.7447 | 0.7799 | 0.7444 |
| fp3 | 0.6283 | 0.6264 | +0.0019 | **better** | 0.8197 | 0.7483 | 0.5093 | 0.5387 |
| fp4 | 0.5445 | 0.4657 | +0.0788 | **better** | 0.5834 | 0.5288 | 0.5104 | 0.4160 |
| fp5 | 0.6182 | 0.6725 | -0.0543 | worse | 0.4737 | 0.6109 | 0.8896 | 0.7480 |
| fp6 | 0.6419 | 0.6008 | +0.0411 | **better** | 0.7830 | 0.6940 | 0.5439 | 0.5297 |
| fp7 | 0.5172 | 0.5570 | -0.0398 | worse | 0.5901 | 0.6799 | 0.4603 | 0.4718 |
| **mean** | **0.6068** | **0.5982** | **+0.0086** | **better** | — | — | — | — |

## Interpretation

Over 7 completed flight path(s): Alea-tuned is better on 4, worse on 3, tied on 0.

Average test Dice: Alea-tuned **0.6068** vs. original **0.5982** (+0.0086 improvement).

---
*Metrics: global pixel-level Dice/IoU/precision/recall. Threshold selected on validation set only and applied once to the held-out test set. These are directly comparable to the original baseline threshold-sweep results.*
