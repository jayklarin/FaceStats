# src/pipeline/embeddings.py
"""
embeddings.py
-------------

Generates 512-D face embeddings using InsightFace ArcFace.

Responsibilities:
- Load ArcFace model (ONNX or PyTorch)
- Convert aligned images into normalized tensors
- Run forward pass to obtain embedding vectors
- L2 normalize embeddings
- Chunk embeddings into parquet/arrow batches
- Save them inside `data/interim/checkpoints/`

Inputs:
- Preprocessed aligned faces in `data/interim/`

Outputs:
- Parquet/Arrow batches containing:
    - filename
    - embedding (512-D vector)

Tools:
- insightface.ArcFace
- PyTorch backend
- numpy
- pandas / pyarrow

TODO:
- Implement batch encoder
- Add GPU/MPS device support
- Add progress tracking + timing
"""

from pathlib import Path
import numpy as np
import onnxruntime as ort
import cv2

from src.utils.file_io import list_images


class EmbeddingExtractor:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)

    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        """Return L2-normalized 512-D embedding."""
        # TODO: implement preprocessing + ONNX inference
        return np.zeros(512, dtype=np.float32)

    def process_folder(self, folder: str) -> dict:
        """Return dict: filename → embedding vector."""
        result = {}
        for img_path in list_images(folder):
            img = cv2.imread(str(img_path))
            emb = self.get_embedding(img)
            result[img_path.name] = emb
        return result
