"""
learning_curves.py
-------------------

Generates training curves and diagnostic plots.

Responsibilities:
- Plot loss curves
- Plot accuracy curves for classification models
- Save plots for instructor review

Tools:
- matplotlib
- seaborn
"""

from pathlib import Path
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


class LearningCurvePlotter:
    """
    Utility class for plotting training diagnostics such as loss and accuracy curves.

    Supports:
    - Regression training curves (loss only)
    - Classification training curves (loss + accuracy)
    - Saving high-quality PNGs for instructor review
    """

    def __init__(self, output_dir: str = "data/processed/learning_curves"):
        """
        Initialize the plotter with an output directory.

        Parameters
        ----------
        output_dir : str
            Directory where generated PNG curves will be stored.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set(style="whitegrid", context="talk")

    # ---------------------------------------------------------------------- #
    #  Loss Curve
    # ---------------------------------------------------------------------- #
    def plot_loss_curve(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: str = "Training Loss Curve",
        filename: str = "loss_curve.png",
    ):
        """
        Plot a regression/classification loss curve.

        Parameters
        ----------
        train_losses : List[float]
            Loss values for each training epoch.
        val_losses : List[float], optional
            Validation loss per epoch.
        title : str
            Title for the plot.
        filename : str
            Final exported filename.
        """

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", linewidth=2)

        if val_losses is not None:
            plt.plot(val_losses, label="Validation Loss", linewidth=2)

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[Saved] Loss curve → {save_path}")

    # ---------------------------------------------------------------------- #
    #  Accuracy Curve
    # ---------------------------------------------------------------------- #
    def plot_accuracy_curve(
        self,
        train_acc: List[float],
        val_acc: Optional[List[float]] = None,
        title: str = "Training Accuracy Curve",
        filename: str = "accuracy_curve.png",
    ):
        """
        Plot classification accuracy curves.

        Parameters
        ----------
        train_acc : List[float]
            Training accuracy per epoch.
        val_acc : List[float], optional
            Validation accuracy per epoch.
        title : str
            Plot title.
        filename : str
            File to save the output image.
        """

        plt.figure(figsize=(10, 6))
        plt.plot(train_acc, label="Train Accuracy", linewidth=2)

        if val_acc is not None:
            plt.plot(val_acc, label="Validation Accuracy", linewidth=2)

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[Saved] Accuracy curve → {save_path}")

    # ---------------------------------------------------------------------- #
    #  Combined (Loss + Accuracy)
    # ---------------------------------------------------------------------- #
    def plot_combined(
        self,
        history: Dict[str, List[float]],
        filename: str = "combined_curves.png",
    ):
        """
        Plots both loss + accuracy curves for classification in a single image.

        Expects keys:
            history["train_loss"], history["val_loss"],
            history["train_acc"], history["val_acc"]

        Parameters
        ----------
        history : dict
            Dictionary of training metrics.
        filename : str
            Output filename.
        """

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Loss subplot
        axs[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axs[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        # Accuracy subplot
        axs[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
        axs[1].plot(history["val_acc"], label="Val Acc", linewidth=2)
        axs[1].set_title("Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[Saved] Combined loss + accuracy curves → {save_path}")

