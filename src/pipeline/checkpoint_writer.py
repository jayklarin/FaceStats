# src/pipeline/checkpoint_writer.py
"""
checkpoint_writer.py
---------------------

Writes intermediate model outputs (embeddings, age/gender predictions,
ethnicity predictions, attractiveness scores) to chunked Parquet or Arrow files.

Responsibilities:
- Provide a consistent interface for saving intermediate artifacts
- Manage chunk sizes (e.g., every 2,000 images)
- Write outputs to `data/interim/checkpoints/`
- Ensure schema consistency across all pipeline stages

Inputs:
- Dictionaries or DataFrames of model outputs

Outputs:
- Parquet files (chunk_0.parquet, chunk_1.parquet, ...)

Tools:
- pandas
- pyarrow
- pathlib

TODO:
- Implement streaming writer
- Validate saved schemas
"""

import pandas as pd
from pathlib import Path


def write_checkpoint(data: list[dict], out_dir: str, chunk_id: int):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    out_path = Path(out_dir) / f"chunk_{chunk_id}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path
