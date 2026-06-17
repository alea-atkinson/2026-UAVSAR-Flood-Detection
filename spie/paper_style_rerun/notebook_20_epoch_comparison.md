# 20-Epoch Notebook Run Comparison

These results came from the notebook artifacts in `spie/notebooks/artifacts/paper_reproduction`.

Important dataset note: the saved `training_results.json` files show these runs used `spie/Preprocessed-128` directly, with 17,465 train, 3,850 validation, and 3,853 test samples. They were not run on the 15,714-pair `Preprocessed-128-paper-filtered` dataset. Treat this as a 20-epoch available-dataset run, not an exact SPIE reproduction.

| Model | Accuracy | Precision | Recall | Dice | Paper Accuracy | Paper Precision | Paper Recall | Paper Dice | Dice Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 0.8333 | 0.6260 | 0.6018 | 0.6137 | 0.8700 | 0.7200 | 0.8000 | 0.7600 | -0.1463 |
| Attention U-Net | 0.8268 | 0.5947 | 0.6690 | 0.6297 | 0.8800 | 0.7400 | 0.6800 | 0.7100 | -0.0803 |
| U-Net++ | 0.8352 | 0.6307 | 0.6061 | 0.6182 | 0.8000 | 0.6200 | 0.5300 | 0.5700 | +0.0482 |

| Model | Epochs | Total Time | Mean Epoch Time | Best Val Soft Dice | Best Val Loss |
|---|---:|---:|---:|---:|---:|
| U-Net | 20 | 46.3 min | 138.8 sec | 0.5153 | 0.9536 |
| Attention U-Net | 20 | 47.3 min | 142.0 sec | 0.5214 | 0.9728 |
| U-Net++ | 20 | 56.2 min | 168.7 sec | 0.5141 | 0.9498 |

Ranked by test Dice:

1. Attention U-Net: 0.6297
2. U-Net++: 0.6182
3. U-Net: 0.6137

The three models are close on Dice, within about 0.016. Attention U-Net has the best recall and Dice in this run, while U-Net++ has the best accuracy and precision. Compared with the SPIE paper, U-Net and Attention U-Net underperform on Dice, while U-Net++ exceeds the paper's reported U-Net++ Dice, likely because this run used a different dataset and only 20 epochs.
