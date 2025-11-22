"""
io_utils.py
--------------
Shared file I/O helpers.

Responsibilities:
- Safe image loaders
- Directory helpers
"""

from pathlib import Path
from PIL import Image


def safe_load_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None
