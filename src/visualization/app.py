"""
app.py
---------
Streamlit dashboard for FaceStats v4.0.

Responsibilities:
- Load master metadata
- Show filters and exploration tools
- Trigger composite generation
"""

import streamlit as st
import polars as pl
from pathlib import Path
from PIL import Image

MASTER = Path("data/metadata/master.parquet")
IMG_DIR = Path("data/processed/preproc")
COMP_DIR = Path("data/composites")


def main():
    st.title("FaceStats v4 Dashboard")

    df = pl.read_parquet(MASTER)

    st.sidebar.header("Filters")
    gender = st.sidebar.selectbox("Gender", ["All"] + df["gender_label"].unique().to_list())
    age = st.sidebar.selectbox("Age Group", ["All"] + df["age_label"].unique().to_list())

    subset = df
    if gender != "All":
        subset = subset.filter(pl.col("gender_label") == gender)
    if age != "All":
        subset = subset.filter(pl.col("age_label") == age)

    st.write(f"Filtered count: {subset.height}")

    # Show composite if exists
    composites = list(COMP_DIR.glob("*.jpg"))
    if composites:
        st.image(str(composites[0]), caption="Latest Composite")

    st.write(subset.head().to_pandas())


if __name__ == "__main__":
    main()
