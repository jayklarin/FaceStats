# src/inference/age_gender_infer.py
"""
age_gender_infer.py
-------------------

Runs the InsightFace Attribute Model to estimate age and gender.

Responsibilities:
- Load ONNX model for age/gender prediction
- Preprocess aligned images
- Predict:
    - age (float)
    - gender (0=Female, 1=Male)
- Save predictions to parquet checkpoint

Tools:
- InsightFace Attribute Model
- ONNXRuntime
- numpy
- pandas

TODO:
- Implement inference loop
- Optimize batching
"""

import numpy as np
import onnxruntime as ort


class AgeGenderPredictor:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)

    def predict(self, img: np.ndarray) -> dict:
        """Return {'age': int, 'gender': 'male/female'}."""
        # TODO: implement ONNX call + postprocessing
        return {"age": None, "gender": None}
