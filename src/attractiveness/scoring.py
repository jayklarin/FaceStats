"""
Shared helper to turn raw model outputs into 1–10 attractiveness deciles.

Usage pattern:
- Instantiate AttractivenessScorer with a model checkpoint path.
- Call score_embeddings(df) where df has ["filename", "embedding"].
- Persist outputs via save_scores().
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import polars as pl
import torch

class AttractivenessRegressorV1(torch.nn.Module):
    """
    Matches the stored checkpoint: 512 → 256 → 64 → 1 with ReLU activations.
    """

    def __init__(self, input_dim=512, hidden1=256, hidden2=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_deciles(preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given raw predictions, return percentile ranks (0–1) and deciles (1–10).
    We use dense ranking to guarantee even bucket counts.
    """
    preds = np.atleast_1d(preds)
    pred_series = pl.Series(preds)
    rank_dense = pred_series.rank(method="dense")
    pct = (rank_dense / float(len(pred_series))).to_numpy()
    deciles = np.clip(np.ceil(pct * 10), 1, 10).astype(np.int64)
    return pct, deciles


class AttractivenessScorer:
    """
    Loads the attractiveness regressor and produces both raw and deciled scores.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: Path) -> torch.nn.Module:
        model = AttractivenessRegressorV1()
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _predict(self, embeddings: Iterable[np.ndarray]) -> np.ndarray:
        X = torch.tensor(np.stack(list(embeddings)), dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X).squeeze().cpu().numpy().astype("float32")
        return preds

    def score_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Input: Polars DataFrame with columns ["filename", "embedding"].
        Output: DataFrame with ["filename", "attractiveness_raw", "attractiveness_pct", "attractiveness"].
        """
        if not {"filename", "embedding"}.issubset(df.columns):
            raise ValueError("DataFrame must include 'filename' and 'embedding' columns.")

        preds = self._predict(df["embedding"])
        preds = np.atleast_1d(preds)
        pct, deciles = compute_deciles(preds)

        return pl.DataFrame(
            {
                "filename": df["filename"],
                "attractiveness_raw": preds,
                "attractiveness_pct": pct,
                "attractiveness": deciles,
            }
        )

    @staticmethod
    def save_scores(
        df_scores: pl.DataFrame,
        parquet_path: Path,
        numpy_path: Optional[Path] = None,
    ) -> None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_scores.write_parquet(parquet_path)
        if numpy_path:
            numpy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(numpy_path, df_scores["attractiveness"].to_numpy())
