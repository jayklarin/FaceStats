# src/training/train_attractiveness.py
"""
train_attractiveness.py
------------------------

Training script for the attractiveness regression model.

Responsibilities:
- Load embeddings + metadata
- Prepare training/validation splits
- Train PyTorch MLP with logging + checkpoints
- Save:
    - trained model weights
    - training curves (loss)
    - evaluation metrics

Tools:
- PyTorch
- pandas
- numpy
- matplotlib (learning curves)

TODO:
- Add early stopping
- Add Weights & Biases optional integration
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.training.attractiveness_model import AttractivenessMLP


def train_model(X, y, model_path):
    # TODO: dataloader, optimizer, loss, training loop
    # TODO: save model + loss curve plot
    pass
