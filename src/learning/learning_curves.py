"""
learning_curves.py
--------------------
Plots MLP training curves for instructor review.

Responsibilities:
- Load saved curve data (list of losses)
- Plot loss vs. epoch
"""

import matplotlib.pyplot as plt
from pathlib import Path
import json


def plot_curves():
    with open("models/learning_curves.json") as f:
        curves = json.load(f)

    plt.figure(figsize=(8, 5))
    plt.plot(curves["loss"], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Attractiveness Model Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/learning_curve.png")
    print("Saved training curve → models/learning_curve.png")
