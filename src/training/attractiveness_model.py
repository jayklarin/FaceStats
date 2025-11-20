# src/training/attractiveness_model.py
"""
attractiveness_model.py
------------------------

Defines the PyTorch attractiveness regression model (MLP).

Responsibilities:
- Define model architecture (MLP layers)
- Provide logic for forward pass
- Provide utilities for loading/saving model weights

Tools:
- torch.nn
- torch.optim

TODO:
- Tune architecture size
- Add dropout/batchnorm support
"""

import torch.nn as nn


class AttractivenessMLP(nn.Module):
    def __init__(self, input_dim=512, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)
