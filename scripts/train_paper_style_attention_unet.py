#!/usr/bin/env python3
"""Run the paper-style Attention U-Net training job."""

from train_paper_style_model import main

if __name__ == "__main__":
    import sys

    sys.argv.extend(["--model", "attention_unet"])
    main()
