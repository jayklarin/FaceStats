# src/inference/attractiveness_infer.py
"""
attractiveness_infer.py
------------------------

Runs the attractiveness regression model (PyTorch MLP).

Responsibilities:
- Load trained attractiveness model
- Normalize embeddings
- Predict 1–10 attractiveness score
- Save results to checkpoint

Tools:
- PyTorch
- numpy
- pandas

TODO:
- Implement GPU/MPS support
- Add confidence intervals (optional)
"""

import torch
import numpy as np


class AttractivenessPredictor:
    def __init__(self, model_path: str):
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

    def predict(self, emb: np.ndarray) -> float:
        """Return attractiveness score (1–10)."""
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(x).item())
