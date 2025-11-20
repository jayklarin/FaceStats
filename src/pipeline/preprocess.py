# src/pipeline/preprocess.py
"""
preprocess.py
-------------

Handles face detection, alignment, and standardized preprocessing.

Responsibilities:
- Load ONNXRuntime session for InsightFace RetinaFace detector
- Detect faces and return bounding boxes + landmarks
- Perform 5-point or 68-point alignment
- Crop, resize (512×512), and normalize image data
- Save aligned images into `data/interim/` for embedding extraction

Inputs:
- Raw image files in `data/raw/`

Outputs:
- Aligned images saved to `data/interim/`
- Metadata (optional) saved as Parquet/CSV

Tools:
- onnxruntime
- numpy
- Pillow
- insightface (RetinaFace)
- cv2 (OpenCV)

TODO:
- Implement detection + alignment
- Handle multiple faces per frame (keep largest)
- Add exception handling + logging
"""

import os
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from src.utils.file_io import list_images
from src.utils.image_ops import save_image


class FacePreprocessor:
    def __init__(self, model_path: str, output_dir: str, size: int = 512):
        self.size = size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = ort.InferenceSession(model_path)

    def detect_and_align(self, img: np.ndarray) -> np.ndarray | None:
        """Run ONNX RetinaFace → return aligned image or None."""
        # TODO: implement ONNX RetinaFace detection + alignment
        return None

    def process_folder(self, folder: str):
        """Iterate through raw images and produce aligned crops."""
        for img_path in list_images(folder):
            img = cv2.imread(str(img_path))
            aligned = self.detect_and_align(img)

            if aligned is not None:
                out_path = self.output_dir / img_path.name
                save_image(out_path, aligned)


def run_preprocessing():
    """Convenience function for orchestrator."""
    # TODO: load config + run pipeline
    pass
