# Mermaid overview
```mermaid
flowchart LR
    N01[01_preprocess.ipynb] --> P1[preprocess.py<br/>preprocess_image, run_preprocessing]
    N02[02_embeddings.ipynb] --> E1[embed_clip.py<br/>load_model, extract_embedding, run_embedding, extract_clip_embeddings, get_clip_embedding]
    N03[03_attributes.ipynb] --> A1[face_attributes.py<br/>get_embedding, infer_attributes]
    N04[04_attractiveness_model.ipynb] --> M1[train_attractiveness.py<br/>EmbeddingDataset, run_training]
    N04 --> M2[attractiveness_model.py<br/>AttractivenessRegressor]
    N05a[05_attractiveness_inference.ipynb] --> M2
    N05b[05_composites.ipynb / 06_composites.ipynb] --> C1[composite_generator.py<br/>filter_images, make_composite, filter_faces, generate_composite]
    N05c[05_ethnicity_clusters.ipynb] --> D1[parquet consumption<br/>(no specific src helpers)]
    Glue[metadata/build_master.py<br/>run_merge] --> D1
```

# Notebook ↔️ Python Interplay

Quick map of how the notebooks call into the Python modules/functions.

| Notebook | Purpose | Python entry points (module → functions/classes) |
| --- | --- | --- |
| `notebooks/01_preprocess.ipynb` | Resize/clean raw images into `data/processed/preproc/` | `src/pipeline/preprocess.py` → `preprocess_image`, `run_preprocessing` |
| `notebooks/02_embeddings.ipynb` | Generate CLIP embeddings parquet | `src/embeddings/embed_clip.py` → `load_model`, `extract_embedding`, `run_embedding`, `extract_clip_embeddings`, `get_clip_embedding` |
| `notebooks/03_attributes.ipynb` | Infer gender/ethnicity per image | `src/attributes/face_attributes.py` → `get_embedding` (pulls CLIP embedding), `infer_attributes` |
| `notebooks/04_attractiveness_model.ipynb` | Train attractiveness regressor | `src/models/train_attractiveness.py` → `EmbeddingDataset`, `run_training`; `src/models/attractiveness_model.py` → `AttractivenessRegressor` |
| `notebooks/04_visualize_attributes.ipynb` | Explore/plot metadata | Consumes parquet outputs; no dedicated helper functions in `src/` |
| `notebooks/05_attractiveness_inference.ipynb` | Score attractiveness with trained model | `src/models/attractiveness_model.py` → `AttractivenessRegressor` (load weights); uses saved model from training |
| `notebooks/05_composites.ipynb` / `06_composites.ipynb` | Filter faces and build composite images | `src/composite/composite_generator.py` → `filter_images`, `make_composite`, `filter_faces`, `generate_composite` |
| `notebooks/05_ethnicity_clusters.ipynb` | Cluster/analyze attributes | Consumes parquet outputs; no dedicated helper functions in `src/` |
| (pipeline glue) | Merge metadata variants | `src/metadata/build_master.py` → `run_merge` |

Data flow snapshot (not exhaustive):
- `01_preprocess` → writes `data/processed/preproc/`
- `02_embeddings` → reads preproc JPGs, writes `data/processed/embeddings/embeddings_clip.parquet`
- `03_attributes` → reads embeddings (via `get_embedding`), writes `data/processed/metadata/attributes*.parquet`
- `04_attractiveness_model` → trains on embeddings; saves `models/attractiveness_regressor.pt`
- `05_attractiveness_inference` → loads model; writes `data/processed/metadata/attractiveness_scores.parquet`
- `05/06_composites` → reads metadata + preproc images; writes `data/processed/composites/…`
