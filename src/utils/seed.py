# src/utils/seed.py
"""
seed.py
-------

Ensures reproducible experiments.

Responsibilities:
- Set seeds for NumPy
- Set seeds for PyTorch
- Set seeds for random module

Tools:
- numpy
- torch
- random

TODO:
- Add deterministic PyTorch backend mode
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
