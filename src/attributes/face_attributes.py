"""
Unified face attribute inference using InsightFace (age, gender, ethnicity).
No HuggingFace dependencies.
"""

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

# ------------------------
# Initialize InsightFace
# ------------------------

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)   # Works on CPU/MPS without modification


# ------------------------
# Single-image inference
# ------------------------

def infer_attributes(image_path: str):
    """
    Returns a dict with: age, gender, ethnicity
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    faces = app.get(img)

    if len(faces) == 0:
        return {
            "age": None,
            "gender": "unknown",
            "ethnicity": "unknown",
        }

    face = faces[0]

    # Gender
    gender = "male" if face.sex == 1 else "female"

    # Ethnicity probabilities from InsightFace
    race_probs = face.race  # array of 7 values
    race_labels = ["Asian", "White", "Black", "Indian", "Middle Eastern", "Latino", "Other"]

    race_idx = int(np.argmax(race_probs))
    race_conf = float(race_probs[race_idx])

    # Confidence threshold
    ETHNICITY_THRESHOLD = 0.55

    if race_conf >= ETHNICITY_THRESHOLD:
        ethnicity = race_labels[race_idx]
    else:
        ethnicity = "unknown"


    return {
        "age": float(face.age),
        "gender": gender,
        "ethnicity": ethnicity,
    }
