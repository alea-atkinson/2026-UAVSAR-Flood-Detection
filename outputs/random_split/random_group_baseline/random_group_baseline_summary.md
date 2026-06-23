# Random Group Split Baseline Summary

**Split:** IEEE PNG-filtered random tile-name group split (no tile_name overlap between train/val/test).

**Models:**
- **Original U-Net:** BCEWithLogitsLoss, Adam, lr=1e-3, batch=8, base_channels=32, epochs=20
- **Alea-tuned U-Net:** FocalDiceLoss, AdamW, lr=9.327e-5, wd=6.088e-6, batch=16, base_channels=32, epochs=20

> All metrics are global pixel-level. Threshold selected on validation Dice (sweep 0.05–0.95, step 0.05), applied once to the held-out test split.

## Random Split Results

| Model | Sel Thresh | Val Dice | Test Dice | Test IoU | Test Prec | Test Recall |
|---|---|---|---|---|---|---|
| Original U-Net | 0.25 | 0.6734 | 0.5917 | 0.4202 | 0.5069 | 0.7106 |
| Alea-tuned U-Net | 0.55 | 0.7053 | 0.6844 | 0.5203 | 0.7201 | 0.6522 |

## Comparison: Random Split vs. Strict Leave-One-Flight-Path-Out

Strict results are averaged over fp1–fp7 (global threshold applied to all seven held-out flight paths).

| Model | Random Test Dice | Strict Test Dice | Delta | Random Prec | Strict Prec | Random Recall | Strict Recall |
|---|---|---|---|---|---|---|---|
| Original U-Net | 0.5917 | 0.6012 | **-0.0095** | 0.5069 | 0.6562 | 0.7106 | 0.5666 |
| Alea-tuned U-Net | 0.6844 | 0.6267 | **+0.0577** | 0.7201 | 0.6694 | 0.6522 | 0.6063 |

### Strict Baseline Reference

| Model | Threshold | Test Dice | Test Precision | Test Recall |
|---|---|---|---|---|
| Original U-Net (strict) | 0.35 | 0.6012 | 0.6562 | 0.5666 |
| Alea-tuned U-Net (strict) | 0.55 | 0.6267 | 0.6694 | 0.6063 |

## Analysis

### Q1: How high is performance on the random mixed-flight-path split?

On the random tile-name group split, the original U-Net achieves a test Dice of **0.5917** (threshold=0.25) and the Alea-tuned U-Net achieves **0.6844** (threshold=0.55). The Alea-tuned model clearly outperforms the original on this split by +0.0927 Dice and has higher precision (0.7201 vs 0.5069). The original U-Net's random-split Dice (0.5917) is marginally *below* its strict baseline (0.6012, −0.0095), while the Alea-tuned model's random-split Dice (0.6844) is above its strict baseline (0.6267, +0.0577). The split effect on performance is therefore model- and training-recipe dependent rather than a uniform inflation.

### Q2: Does Alea-tuned still beat original on the random split?

Yes — Alea-tuned U-Net outperforms the original on the random split (Dice 0.6844 vs 0.5917, delta=+0.0927).

### Q3: How much higher is random-split performance than strict leave-one-flight-path-out performance?

The split effect is not uniform. The Alea-tuned U-Net scores +0.0577 higher on the random split than on the strict split, consistent with test tiles coming from flight paths already seen during training. The original U-Net, however, scores −0.0095 lower on the random split than on the strict split — a near-zero difference suggesting no meaningful advantage from the easier evaluation setting for this model. Rather than a consistent inflation, the direction and magnitude of the random-vs-strict gap depends on the model's training recipe.

### Q4: What does this suggest about random splits vs unseen-flight-path generalization?

The results here show that the random group split does not systematically inflate performance for both models: the original U-Net's Dice is essentially unchanged (−0.0095) while the Alea-tuned model gains +0.0577. This suggests the split effect is training-recipe dependent — the Alea-tuned model benefits more from seeing tiles of the same flight paths during training, while the original model does not.

That said, the random group split remains a useful mixed-flight-path baseline and a sanity check on whether a model trains and converges correctly across the full dataset. It is not, however, a substitute for the strict leave-one-flight-path-out protocol, which forces generalisation to entirely unseen acquisitions. **The strict split should be treated as the primary benchmark** for reporting and model selection whenever unseen-flight-path generalisation is the deployment goal. Random-split numbers provide a complementary reference point but should not be used to claim generalisation performance.

---
*Metrics: global pixel-level Dice/IoU/precision/recall. Threshold selected on validation set only and applied once to the held-out test set. Strict baseline numbers are averaged over fp1–fp7 held-out flight paths.*
