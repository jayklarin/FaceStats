"""
Streamlit app: upload a portrait → get attractiveness score.
Uses CLIP embedding + shared attractiveness scorer and reports percentile/decile
against the canonical dataset scores parquet if available.
"""

import io
import sys
from pathlib import Path
import tempfile
from typing import Optional, Tuple
import numpy as np
import polars as pl
import streamlit as st
from PIL import Image, ImageOps

try:
    import mediapipe as mp  # type: ignore
except ImportError:
    mp = None

# Ensure project root on sys.path when launched via streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attractiveness.scoring import AttractivenessScorer
from src.embeddings.embed_clip import get_clip_embedding

MODEL_PATH = Path("src/models/attractiveness_regressor.pt")
REF_SCORES = Path("data/processed/metadata/attractiveness_scores.parquet")
TARGET_SIZE = 512
FACE_FRACTION_MIN = 0.35


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


def detect_face_bbox(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Return pixel bbox (x0, y0, x1, y1) for the strongest detected face.
    Uses MediaPipe if available; otherwise returns None.
    """
    if mp is None:
        return None

    arr = np.asarray(image.convert("RGB"))
    try:
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4
        ) as detector:
            results = detector.process(arr)
    except Exception:
        return None

    if not results.detections:
        return None

    def _score(det):
        return det.score[0] if det.score else 0.0

    best = max(results.detections, key=_score)
    rel = best.location_data.relative_bounding_box
    h, w = arr.shape[:2]

    x0 = max(0, int(round(rel.xmin * w)))
    y0 = max(0, int(round(rel.ymin * h)))
    x1 = min(w, int(round((rel.xmin + rel.width) * w)))
    y1 = min(h, int(round((rel.ymin + rel.height) * h)))

    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def square_crop_from_bbox(bbox: Tuple[int, int, int, int], img_size: Tuple[int, int], pad: float = 0.35):
    """Build a square crop around the bbox with a bit of padding while staying in frame."""
    x0, y0, x1, y1 = bbox
    w, h = img_size
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + pad)
    side = min(side, w, h)

    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))

    left = max(0, min(left, w - int(side)))
    top = max(0, min(top, h - int(side)))

    side_int = int(round(side))
    return left, top, left + side_int, top + side_int


def center_square_box(img_size: Tuple[int, int]):
    """Fallback square crop centered in the frame."""
    w, h = img_size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return left, top, left + side, top + side


def make_face_compliant(image: Image.Image):
    """
    Ensure the image is 512×512 with the face as the dominant region.
    Returns the processed PIL image and metadata about the adjustment.
    """
    img = ImageOps.exif_transpose(image).convert("RGB")
    bbox = detect_face_bbox(img)

    face_fraction = None
    if bbox:
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        face_fraction = face_area / float(img.width * img.height)

    size_ok = img.size == (TARGET_SIZE, TARGET_SIZE)
    face_ok = face_fraction is not None and face_fraction >= FACE_FRACTION_MIN
    compliant = size_ok and face_ok

    if compliant:
        final_img = img
        message = "Image already 512×512 with a sufficiently large face."
    else:
        if bbox:
            crop_box = square_crop_from_bbox(bbox, img.size, pad=0.35)
            message = "Detected face; auto-cropped and resized to 512×512."
        else:
            crop_box = center_square_box(img.size)
            if mp is None:
                message = "Face detector not installed; center-cropped and resized to 512×512."
            else:
                message = "No face detected; center-cropped and resized to 512×512."
        final_img = img.crop(crop_box).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    meta = {
        "compliant": compliant,
        "face_detected": bbox is not None,
        "face_fraction": face_fraction,
        "message": message,
    }
    return final_img, meta


def prepare_uploaded_image(uploaded_file):
    """Load, validate, and return a temp path for the processed upload."""
    raw_bytes = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(raw_bytes))
    processed_img, meta = make_face_compliant(img)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        processed_img.save(tmp, format="JPEG", quality=95)
        tmp_path = Path(tmp.name)

    return processed_img, tmp_path, meta


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

    if mp is None:
        st.info("Optional dependency `mediapipe` not installed; images will be center-cropped when a face is not detected.")

    if uploaded_files:
        for uploaded in uploaded_files[:5]:
            try:
                processed_img, tmp_path, meta = prepare_uploaded_image(uploaded)
            except Exception as e:
                st.error(f"Failed to load {uploaded.name}: {e}")
                continue

            try:
                scored = score_path(tmp_path, scorer, df_ref, raw_col)
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(processed_img, caption=uploaded.name, width=256)
                    if not meta["compliant"]:
                        st.caption(meta["message"])
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
