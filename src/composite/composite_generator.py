"""
composite_generator.py
-------------------------
Generates mean composite faces.

Responsibilities:
- Filter images based on metadata
- Load them with PIL
- Convert to NumPy arrays
- Compute mean or weighted mean
- Save composite image
"""

import numpy as np
from pathlib import Path
from PIL import Image
import polars as pl

MASTER = Path("data/metadata/master.parquet")
IMG_DIR = Path("data/processed/preproc")
OUT_DIR = Path("data/composites")
OUT_DIR.mkdir(exist_ok=True)


def create_composite(filter_expr: str, outname: str):
    df = pl.read_parquet(MASTER).filter(filter_expr)
    imgs = []

    for fname in df["filename"].to_list():
        path = IMG_DIR / fname
        imgs.append(np.array(Image.open(path).convert("RGB"), dtype=np.float32))

    if not imgs:
        print("No images after filtering.")
        return

    mean_img = np.mean(np.stack(imgs), axis=0).astype(np.uint8)
    outpath = OUT_DIR / outname
    Image.fromarray(mean_img).save(outpath)
    print(f"Saved: {outpath}")
