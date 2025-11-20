# src/inference/user_upload_api.py
"""
user_upload_api.py
-------------------

Backend for "Upload Face → Get Score" feature.

Responsibilities:
- Accept uploaded image
- Run:
    1. preprocessing
    2. embedding extraction
    3. attractiveness score
    4. ethnicity classification
    5. age + gender model
- Return JSON response

Tools:
- FastAPI (recommended)
- Pillow
- numpy

TODO:
- Implement full API
- Add CORS + rate limiting
"""

import cv2
from pathlib import Path

from src.pipeline.embeddings import EmbeddingExtractor
from src.inference.attractiveness_infer import AttractivenessPredictor
from src.inference.ethnicity_infer import EthnicityPredictor
from src.inference.age_gender_infer import AgeGenderPredictor


class UserUploadScorer:
    def __init__(self, config):
        self.embedder = EmbeddingExtractor(config["arcface"])
        self.attr = AttractivenessPredictor(config["mlp"])
        self.eth = EthnicityPredictor(config["lgbm"], config["encoder"])
        self.age = AgeGenderPredictor(config["attribute"])

    def score(self, img_path: str) -> dict:
        img = cv2.imread(img_path)
        emb = self.embedder.get_embedding(img)

        attr = self.attr.predict(emb)
        eth_label, eth_probs = self.eth.predict(emb)
        age_gender = self.age.predict(img)

        return {
            "attractiveness": attr,
            "ethnicity": eth_label,
            "ethnicity_probs": eth_probs.tolist(),
            "age_gender": age_gender,
        }
