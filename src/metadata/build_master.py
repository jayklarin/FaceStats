"""
build_master.py
------------------
Combines embeddings + attractiveness + age + gender + ethnicity.

Responsibilities:
- Read individual parquet files
- Left join on filename
- Save master metadata table
"""

import polars as pl
from pathlib import Path

OUT = Path("data/metadata/master.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)


def run_merge():
    emb = pl.read_parquet("data/embeddings/embeddings.parquet")
    age = pl.read_parquet("data/attributes/age.parquet")
    gender = pl.read_parquet("data/attributes/gender.parquet")
    eth = pl.read_parquet("data/attributes/ethnicity.parquet")
    scores = pl.read_parquet("data/scores/attractiveness.parquet")

    df = (
        emb
        .join(scores, on="filename", how="left")
        .join(age, on="filename", how="left")
        .join(gender, on="filename", how="left")
        .join(eth, on="filename", how="left")
    )

    df.write_parquet(OUT)
    print(f"Master table saved: {OUT}")
