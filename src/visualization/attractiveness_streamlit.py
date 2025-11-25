"""
Streamlit app: upload a portrait → get attractiveness score.
Uses CLIP embedding + shared attractiveness scorer and reports percentile/decile
against the canonical dataset scores parquet if available.
"""

import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import polars as pl
import streamlit as st
from PIL import Image

# Ensure project root on sys.path when launched via streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attractiveness.scoring import AttractivenessScorer
from src.embeddings.embed_clip import get_clip_embedding

MODEL_PATH = Path("src/models/attractiveness_regressor.pt")
REF_SCORES = Path("data/processed/metadata/attractiveness_scores.parquet")


@st.cache_resource(show_spinner=False)
def load_scorer() -> AttractivenessScorer:
    return AttractivenessScorer(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_reference():
    if not REF_SCORES.exists():
        return None, None
    df_ref = pl.read_parquet(REF_SCORES)
    raw_col = "attractiveness_raw" if "attractiveness_raw" in df_ref.columns else "attractiveness"
    return df_ref, raw_col


def adjust_percentile(scored_df: pl.DataFrame, df_ref: pl.DataFrame, raw_col: str) -> pl.DataFrame:
    if df_ref is None or raw_col is None:
        return scored_df
    raw_val = float(scored_df[raw_col][0]) if raw_col in scored_df.columns else float(scored_df["attractiveness"][0])
    pct = float((df_ref[raw_col] <= raw_val).mean())
    decile = int(np.clip(np.ceil(pct * 10), 1, 10))
    return scored_df.with_columns([
        pl.Series("attractiveness_pct", [pct]),
        pl.Series("attractiveness", [decile]),
    ])


def score_path(path: Path, scorer: AttractivenessScorer, df_ref: pl.DataFrame, raw_col: str) -> pl.DataFrame:
    emb = get_clip_embedding(path)
    emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
    df = pl.DataFrame({"filename": [path.name], "embedding": [emb_list]})
    df = df.with_columns(pl.col("embedding").cast(pl.List(pl.Float32)))
    scored = scorer.score_embeddings(df)
    return adjust_percentile(scored, df_ref, raw_col)


def main():
    st.set_page_config(page_title="Attractiveness Scorer", page_icon="💫")
    st.title("Attractiveness Scorer")
    st.write("Upload 1–5 portraits (JPEG/PNG) to get the model score, percentile, and 1–10 decile.")

    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run inference/training to generate it.")
        return

    df_ref, raw_col = load_reference()
    if df_ref is None:
        st.warning("Reference scores parquet not found; percentile/decile will be based on the single image.")

    scorer = load_scorer()

    uploaded_files = st.file_uploader(
        "Upload portraits",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Select up to 5 images",
    )

    if uploaded_files:
        for uploaded in uploaded_files[:5]:
            img_bytes = uploaded.getvalue()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = Path(tmp.name)

            try:
                scored = score_path(tmp_path, scorer, df_ref, raw_col)
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(Image.open(tmp_path), caption=uploaded.name, width=256)
                with col2:
                    st.write("Result:")
                    st.dataframe(scored.to_pandas(), use_container_width=True)
                    if "attractiveness" in scored.columns:
                        st.metric(
                            label="Attractiveness (decile 1–10)",
                            value=int(scored["attractiveness"][0]),
                        )
                    if "attractiveness_raw" in scored.columns:
                        st.caption(f"Raw score: {float(scored['attractiveness_raw'][0]):.4f}")
            except Exception as e:
                st.error(f"Scoring failed for {uploaded.name}: {e}")
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
    else:
        st.info("Upload one or more JPEG/PNG files to see scores (max 5 shown).")


if __name__ == "__main__":
    main()
