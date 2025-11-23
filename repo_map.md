# FaceStats Repository Map (v4)

Quick reference for what each folder/file is for, rooted at `FaceStats/`.

## Top level
- `LICENSE` — Project license.
- `README.md` — Legacy v3.5 readme.
- `readme2.md` — Updated v4 readme.
- `requirements.txt` — Python dependencies.
- `schematics.ipynb` — Diagrams and repo maps (mermaid).
- `x.ipynb`, `del.ipynb` — Misc notebooks (utility/demo).
- `tools_summary.md` — Tooling snapshot with mermaid map.
- `files_and_folders.md` — This file.

## config/
- Placeholders for configuration files.

## data/
- `raw/` — Source images (sample subset present; TODO full dataset parts).
- `processed/`
  - `embeddings_clip.parquet` — CLIP embeddings (filename + embedding array).
  - `preproc/` — Preprocessed/aligned images (JPEGs).
  - `metadata/`
    - `attributes.parquet` — Attribute predictions (age/gender/ethnicity).
    - `attributes_with_meta.parquet` — Attributes with extra meta columns.
    - `attributes_clean.parquet` — Cleaned subset (single face, confidence filters).
    - `attributes_flags.parquet` — Rows needing review.
    - `attributes_with_clusters.parquet` — Clustered subset with embeddings.
    - `attributes_with_meta.parquet` — Combined attributes/meta table.
    - `attributes_with_labels.parquet` — (optional) merged labels output.
    - `attractiveness_scores.parquet` — Inference scores (if generated).
  - `interim/` — Checkpoints/staging (placeholder).
  - `attributes/`, `embeddings/`, `preprocessed/` — Legacy placeholders (mostly empty).
- `models_insightface/` — InsightFace weights/configs (placeholders).

## notebooks/
- `01_preprocess.ipynb` — Preprocess raw images into `data/processed/preproc/`.
- `02_embeddings.ipynb` — Extract CLIP embeddings to `embeddings_clip.parquet`.
- `03_attributes.ipynb` — Infer attributes; produce attributes/metas/flags.
- `03_labels.ipynb` — Original label notebook (empty/corrupt).
- `04_visualize_attributes.ipynb` — Attribute exploration/plots.
- `04_attractiveness_model.ipynb` — Train attractiveness regressor (synthetic labels placeholder).
- `05_attractiveness_inference.ipynb` — Run regressor on embeddings; save scores.
- `05_ethnicity_clusters.ipynb` — Cluster embeddings/attributes.
- `05_ethnicity_clusters copy.ipynb` — Copy of clustering notebook.
- `01_preprocess...05* data/` — Notebook-specific data subfolder (if any).
- TODOs: `fs07_age_gender_ethnicity_inference.ipynb`, `fs08_fairface_alignment.ipynb`.

## src/
- `attributes/` — Attribute inference helpers.
  - `age_infer.py` — Age inference utilities.
  - `face_attributes.py` — Main attribute pipeline (age/gender/ethnicity).
  - TODO: `ethnicity_infer.py`, `gender_infer.py`.
- `composite/`
  - `composite_generator.py` — Build composite faces (mean/PCA).
  - TODO: filters/explorer modules.
- `data_utils/`
  - `constants.py`, `filters.py`, `io_utils.py` — Data helpers/utilities.
  - TODO: dataset splits, validation.
- `embeddings/`
  - `embed_clip.py` — CLIP embedding extraction script.
  - TODO: openclip variant.
- `learning/`
  - `learning_curves.py` — Plot training curves.
  - TODO: trainers, loss functions.
- `metadata/`
  - `build_master.py` — Master metadata assembly.
- `models/`
  - `attractiveness_model.py` — MLP regressor definition.
  - `train_attractiveness.py` — Scripted training stub.
  - TODO: multi-attribute/fairness models.
- `pipeline/`
  - `preprocess.py` — Preprocessing entry point.
  - `__init__.py`
  - TODO: full pipeline runner.
- `visualization/`
  - `app.py`, `__init__.py`
  - TODO: dashboards for attractiveness, embeddings explorer, composite gallery.
- TODO: `utils/` (logger/config loader).

## models/
- `attractiveness_regressor.pt` — Saved PyTorch state_dict (created by 04_attractiveness_model.ipynb).
