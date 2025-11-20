# src/inference/ethnicity_infer.py
"""
ethnicity_infer.py
-------------------

Runs the LightGBM 7-class ethnicity classifier.

Responsibilities:
- Load trained LightGBM model + LabelEncoder
- Predict ethnicity probabilities
- Store:
    - ethnicity_id
    - ethnicity_label
    - prob_*
- Write parquet batches for checkpointing

Tools:
- lightgbm
- scikit-learn LabelEncoder
- pandas
- numpy

TODO:
- Implement batch predictions
- Add top-k outputs
"""

import numpy as np
import joblib
import lightgbm as lgb


class EthnicityPredictor:
    def __init__(self, model_path: str, encoder_path: str):
        self.model = lgb.Booster(model_file=model_path)
        self.encoder = joblib.load(encoder_path)

    def predict(self, emb: np.ndarray):
        """Return label + probability vector."""
        probs = self.model.predict(emb.reshape(1, -1))[0]
        idx = np.argmax(probs)
        label = self.encoder.inverse_transform([idx])[0]
        return label, probs
