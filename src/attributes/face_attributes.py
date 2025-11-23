
import os
import joblib
import numpy as np
from PIL import Image

# ------------------------------------------------------------
# Load trained models
# ------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)

GENDER_MODEL = joblib.load(os.path.join(MODEL_DIR, "gender_clf.pkl"))
ETHNICITY_MODEL = joblib.load(os.path.join(MODEL_DIR, "ethnicity_clf.pkl"))

# Mapping (must match training order)
GENDER_CLASSES = ["female", "male"]
ETHNICITY_CLASSES = [
    "white",
    "black",
    "latino/hispanic",
    "east_or_southeast_asian",
    "indian",
    "middle_eastern"
]

# ------------------------------------------------------------
# Extract CLIP embedding for a single image
# (Used at inference time on new images)
# ------------------------------------------------------------
def get_embedding(image_path):
    from src.embeddings.clip_embedder import get_clip_embedding
    return get_clip_embedding(image_path)

# ------------------------------------------------------------
# Main inference function used by pipelines
# ------------------------------------------------------------
def infer_attributes(image_path):
    emb = get_embedding(image_path)          # shape (N,)
    emb = np.array(emb).reshape(1, -1)

    gender_pred = GENDER_MODEL.predict(emb)[0]
    ethnicity_pred = ETHNICITY_MODEL.predict(emb)[0]

    return {
        "gender": GENDER_CLASSES[gender_pred],
        "ethnicity": ETHNICITY_CLASSES[ethnicity_pred],
    }
