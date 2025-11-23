import os
import streamlit as st
import polars as pl
from PIL import Image

# --------------------------------------------
# CONFIG
# --------------------------------------------
IMAGE_DIR = "data/processed/preproc"
OUTPUT_FILE = "data/processed/metadata/manual_labels.csv"

ETHNICITY_OPTIONS = [
    "white",
    "black",
    "latino/hispanic",
    "east_or_southeast_asian",   # combined category
    "indian",
    "middle_eastern",
]

GENDER_OPTIONS = ["male", "female"]

st.set_page_config(page_title="FaceStats — Manual Labeler", layout="wide")

st.title("🖼️ FaceStats Manual Labeling Interface")
st.write("Label gender + ethnicity for your dataset. Saves automatically.")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# --------------------------------------------
# LOAD EXISTING LABELS (ensure schema is valid)
# --------------------------------------------
if os.path.exists(OUTPUT_FILE):
    df_labels = pl.read_csv(OUTPUT_FILE)

    # Fix Null columns by force-casting to Utf8
    for col in ["filename", "gender", "ethnicity"]:
        if df_labels[col].dtype == pl.Null:
            df_labels = df_labels.with_columns(pl.col(col).cast(pl.Utf8))

    labeled_files = set(df_labels["filename"].to_list())

else:
    df_labels = pl.DataFrame({
        "filename": pl.Series([], dtype=pl.Utf8),
        "gender":   pl.Series([], dtype=pl.Utf8),
        "ethnicity": pl.Series([], dtype=pl.Utf8),
    })
    labeled_files = set()

# --------------------------------------------
# COLLECT ALL IMAGES
# --------------------------------------------
all_images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Determine first unlabeled image (resume)
if "idx" not in st.session_state:
    next_idx = 0
    for i, f in enumerate(all_images):
        if f not in labeled_files:
            next_idx = i
            break
    st.session_state.idx = next_idx

idx = st.session_state.idx

# --------------------------------------------
# STOP IF DONE
# --------------------------------------------
if idx >= len(all_images):
    st.success("🎉 All images have been labeled!")
    st.stop()

filename = all_images[idx]
img_path = os.path.join(IMAGE_DIR, filename)

# --------------------------------------------
# UI — DISPLAY IMAGE
# --------------------------------------------
st.subheader(f"Image {idx+1} / {len(all_images)}")
st.image(Image.open(img_path), caption=filename, width=400)

# --------------------------------------------
# UI — LABELS
# --------------------------------------------
gender = st.radio("Gender:", GENDER_OPTIONS, horizontal=True)
ethnicity = st.radio("Ethnicity:", ETHNICITY_OPTIONS)

# --------------------------------------------
# SAVE FUNCTION (now schema-safe)
# --------------------------------------------
def save_label():
    global df_labels

    new_row = pl.DataFrame({
        "filename": [filename],
        "gender": [gender],
        "ethnicity": [ethnicity],
    }).with_columns([
        pl.col("filename").cast(pl.Utf8),
        pl.col("gender").cast(pl.Utf8),
        pl.col("ethnicity").cast(pl.Utf8),
    ])

    # Remove old row if re-labeling
    if filename in labeled_files:
        df_labels = df_labels.filter(pl.col("filename") != filename)

    # Vertical concat (now safe)
    df_labels = pl.concat([df_labels, new_row], how="vertical")

    df_labels.write_csv(OUTPUT_FILE)
    labeled_files.add(filename)

# --------------------------------------------
# BUTTONS
# --------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Save & Next"):
        save_label()
        st.session_state.idx += 1
        st.rerun()

with col2:
    if st.button("⬅️ Back", disabled=(idx == 0)):
        st.session_state.idx -= 1
        st.rerun()

# --------------------------------------------
# PROGRESS BAR
# --------------------------------------------
st.progress(len(labeled_files) / len(all_images))
st.write(f"**Labeled:** {len(labeled_files)} / {len(all_images)}")
