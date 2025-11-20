# src/utils/file_io.py
"""
file_io.py
----------

Handles file management for the pipeline.

Responsibilities:
- Recursively list images
- Validate directory structure
- Read/write JSON/YAML/CSV/Parquet
- Build consistent filenames

Tools:
- pathlib
- pandas
- json
- yaml

TODO:
- Add file-type detection
"""

from pathlib import Path


def list_images(folder: str):
    return [p for p in Path(folder).iterdir() if p.suffix.lower() in [".jpg", ".png"]]
