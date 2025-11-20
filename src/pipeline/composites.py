# src/pipeline/composites.py
"""
composites.py
-------------

Generates composite faces (mean faces, PCA faces, attractiveness deciles).

Responsibilities:
- Load aligned images + embeddings
- Group images by score range or demographic filters
- Compute pixel-wise mean image
- Optionally compute PCA-based "eigenfaces"
- Save composites to `data/processed/composites/`

Inputs:
- Aligned images + metadata

Outputs:
- Composite images (JPEG/PNG)
- Optional PCA plots

Tools:
- numpy
- PIL / cv2
- sklearn.decomposition.PCA

TODO:
- Implement mean-face builder
- Implement weighted composites
- Add PCA visualization tools
"""

import numpy as np
from pathlib import Path
import cv2
from sklearn.decomposition import PCA


def mean_face(img_list: list[np.ndarray]) -> np.ndarray:
    """Return pixel-wise mean face."""
    stack = np.stack(img_list)
    return stack.mean(axis=0).astype(np.uint8)


def pca_face(img_list: list[np.ndarray], n_components=50):
    """Return PCA projection + reconstructed composite."""
    flat = np.array([img.flatten() for img in img_list])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(flat)
    recon = pca.inverse_transform(reduced.mean(axis=0))
    side = int(np.sqrt(recon.size / 3))
    return recon.reshape(side, side, 3).astype(np.uint8)
