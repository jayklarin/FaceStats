"""
embed_clip.py
----------------
Extracts CLIP/Vision Transformer embeddings for FaceStats v4.0.

Responsibilities:
- Load preprocessed images
- Run CLIP or ViT forward pass
- L2-normalize embeddings
- Save embeddings to data/embeddings/embeddings.parquet

Tools:
- torch
- transformers (CLIPProcessor + CLIPModel or ViT)
- PIL
- polars
"""

from pathlib import Path
import torch
import polars as pl
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

IMG_DIR = Path("data/processed/preproc")
EMB_DIR = Path("data/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)

# Lazily initialized globals for single-image inference
_MODEL = None
_PROCESSOR = None


def load_model():
    """Load CLIP model + processor."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def extract_embedding(model, processor, path: Path):
    """Return a 512-D embedding vector for a single image."""
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features[0].cpu().numpy()
    return vec / (vec.norm() + 1e-8)


def run_embedding():
    model, processor = load_model()
    paths = list(IMG_DIR.glob("*.jpg"))
    rows = []

    for p in paths:
        vec = extract_embedding(model, processor, p)
        rows.append({"filename": p.name, "embedding": vec.tolist()})

    df = pl.DataFrame(rows)
    df.write_parquet(EMB_DIR / "embeddings.parquet")
    print(f"Saved embeddings for {len(paths)} images.")
    
def extract_clip_embeddings(input_dir, output_path, model_name="openai/clip-vit-base-patch32"):
    import numpy as np
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    rows = []
    paths = list(input_dir.glob("*.jpg"))

    for p in paths:
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)

        vec = features[0].cpu().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-8)   # <-- FIXED

        rows.append({"filename": p.name, "embedding": vec.tolist()})

    df = pl.DataFrame(rows)
    df.write_parquet(output_path)

    print(f"Saved {len(paths)} embeddings → {output_path}")
    return df


def get_clip_embedding(image_path, model_name="openai/clip-vit-base-patch32"):
    """
    Return a single L2-normalized CLIP embedding for an image path.
    Lazily loads the model/processor once for repeated calls.
    """
    import numpy as np
    global _MODEL, _PROCESSOR

    if _MODEL is None or _PROCESSOR is None:
        _MODEL = CLIPModel.from_pretrained(model_name)
        _PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        _MODEL.eval()

    img = Image.open(image_path).convert("RGB")
    inputs = _PROCESSOR(images=img, return_tensors="pt")
    with torch.no_grad():
        features = _MODEL.get_image_features(**inputs)

    vec = features[0].cpu().numpy()
    return vec / (np.linalg.norm(vec) + 1e-8)

if __name__ == "__main__":
    run_embedding()
