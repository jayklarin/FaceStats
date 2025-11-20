# src/pipeline/metadata_merge.py
"""
metadata_merge.py
-----------------

Merges all pipeline outputs into a unified master metadata Parquet.

Responsibilities:
- Load all intermediate checkpoint files
- Merge on filename / embedding index
- Combine fields:
    - embedding
    - attractiveness score
    - ethnicity class + probabilities
    - age estimate
    - gender estimate
- Produce final `fs35_master.parquet`

Inputs:
- Parquet files from checkpoint writer

Outputs:
- `data/processed/fs35_master.parquet`

Tools:
- pandas
- pyarrow

TODO:
- Build merge logic
- Validate row counts
- Add column sanity checks
"""

from pathlib import Path
import pandas as pd


def merge_checkpoints(checkpoint_dir: str, output_path: str):
    ckpts = list(Path(checkpoint_dir).glob("chunk_*.parquet"))
    dfs = [pd.read_parquet(f) for f in ckpts]
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    merged.to_parquet(output_path, index=False)
    return merged
