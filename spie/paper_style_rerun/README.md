# SPIE Paper-Style Rerun on Available Data

This rerun is labeled as a paper-style rerun on the available repo dataset. The exact curated 3,873-tile SPIE dataset is not available in this repository.

Available-data approximation:

- Source dataset: `spie/Preprocessed-128`
- Filtered dataset: `spie/Preprocessed-128-paper-filtered`
- Water-ratio rule: `0.05 <= water_ratio <= 0.85`
- Split: 70/15/15 from the filtered available pairs
- Input size: 128x128x3
- Batch size: 8
- Epochs: 80
- Optimizer: Adam, learning rate 0.001
- Loss: binary cross entropy plus Dice loss
- Training monitor: soft Dice metric
- Final evaluation: global thresholded TP/FP/FN/TN at threshold 0.5
- No early stopping
- No ReduceLROnPlateau

Run the dataset report:

```bash
.venv-tf312/bin/python scripts/report_paper_style_dataset.py
```

Write configs without training:

```bash
.venv-tf312/bin/python scripts/train_paper_style_model.py --model unet --config-only
.venv-tf312/bin/python scripts/train_paper_style_model.py --model attention_unet --config-only
.venv-tf312/bin/python scripts/train_paper_style_model.py --model unetpp --config-only
```

Train each model:

```bash
.venv-tf312/bin/python scripts/train_paper_style_model.py --model unet
.venv-tf312/bin/python scripts/train_paper_style_model.py --model attention_unet
.venv-tf312/bin/python scripts/train_paper_style_model.py --model unetpp
```

Update the comparison table after any model finishes:

```bash
.venv-tf312/bin/python scripts/write_paper_style_comparison.py
```
