"""
preprocess.py
----------------
Image preprocessing pipeline for FaceStats v4.0.

Responsibilities:
- Load raw images from data/raw/
- Convert to RGB
- Resize to 512×512 (PIL only)
- Save clean processed images to data/processed/preproc/

Tools:
- PIL
- pathlib
"""

from pathlib import Path
from PIL import Image

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/preproc")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(path: Path, outpath: Path) -> None:
    """Load, clean, resize, and save a single image."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((512, 512), Image.BICUBIC)
        img.save(outpath, "JPEG", quality=95)
    except Exception as e:
        print(f"[WARN] Failed: {path} — {e}")


import os
from tqdm import tqdm

def run_preprocessing(input_dir: str, output_dir: str, size: int = 512):
    """
    Preprocess images: load, resize, save.

    Args:
        input_dir (str): Folder containing raw JPEGs.
        output_dir (str): Folder where resized images go.
        size (int): Resize dimension (square).
    """

    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]
    if not images:
        print(f"No images found in {input_dir}")
        return

    count = 0

    for fname in tqdm(images, desc="Preprocessing"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            img = Image.open(input_path).convert("RGB")
            img = img.resize((size, size))
            img.save(output_path, "JPEG")
            count += 1
        except Exception as e:
            print(f"❌ Failed on {fname}: {e}")

    print(f"Completed preprocessing: {count} images.")



if __name__ == "__main__":
    run_preprocessing()
