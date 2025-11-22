"""
age_infer.py
----------------
Predicts age groups using a HuggingFace image-classification model.

Responsibilities:
- Load HF age model
- Predict age category for each image
- Save parquet output
"""

from transformers import pipeline
from pathlib import Path
import polars as pl
from PIL import Image

IMG_DIR = Path("data/processed/preproc")
OUT_PATH = Path("data/attributes/age.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier")


def run_age_inference():
    rows = []

    for p in IMG_DIR.glob("*.jpg"):
        pred = age_pipe(Image.open(p.convert("RGB")))[0]
        rows.append({"filename": p.name, "age_label": pred["label"]})

    pl.DataFrame(rows).write_parquet(OUT_PATH)
    print("Saved age predictions.")
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
model = AutoModelForImageClassification.from_pretrained(
    "nateraw/vit-age-classifier"
).to(DEVICE)


def predict_age(image_path):
    """
    Return estimated age (float)
    """
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_idx = logits.argmax(-1).item()

    label = model.config.id2label[predicted_class_idx]

    if "-" in label:
        low, high = label.split("-")
        return (int(low) + int(high)) / 2
    else:
        return float(label)
