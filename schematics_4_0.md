# schematics-4-0 (FaceStats)

## Notebook → Code & Tools Map
```mermaid
flowchart TB

    classDef notebook fill:#5c7fa6,stroke:#3f5a7b,color:#f2f6fb,font-size:15px,font-weight:bold;
    classDef python fill:#9a80b8,stroke:#6d5789,color:#f7f3fb,font-size:14px,font-weight:bold;
    classDef methods fill:#8cc7ab,stroke:#5e9475,color:#0f2f1f,font-size:13px,font-weight:bold;
    classDef tools fill:#e9c48a,stroke:#b58950,color:#2d1c05,font-size:13px,font-weight:bold;
    classDef files fill:#7fc9aa,stroke:#549578,color:#0f2f1f,font-size:13px,font-weight:bold;

    %% 01_preprocess
    N01["<b>Notebook</b><br>—<br>01_preprocess.ipynb"]
    N01 --> P1["<b>Python</b><br>—<br>preprocess.py"]
    P1 --> T1["<b>Tools</b><br>—<br>Pillow<br>pathlib<br>os<br>tqdm"]
    T1 --> P1M["<div style='text-align:left'><b>Methods</b><br>—<br>preprocess_image(path, outpath)<br><br>run_preprocessing(<br>&nbsp;&nbsp;&nbsp;&nbsp;input_dir<br>&nbsp;&nbsp;&nbsp;&nbsp;output_dir<br>&nbsp;&nbsp;&nbsp;&nbsp;size)</div>"]

    %% 02_embeddings
    N02["<b>Notebook</b><br>—<br>02_embeddings.ipynb"]
    N02 --> E1["<b>Python</b><br>—<br>embed_clip.py"]
    E1 --> T2["<b>Tools</b><br>—<br>torch<br>transformers<br>Pillow<br>polars<br>numpy"]
    T2 --> E1M["<div style='text-align:left'><b>Methods</b><br>—<br>load_model()<br><br>extract_embedding(<br>&nbsp;&nbsp;&nbsp;&nbsp;model<br>&nbsp;&nbsp;&nbsp;&nbsp;processor<br>&nbsp;&nbsp;&nbsp;&nbsp;path)<br><br>run_embedding()<br><br>extract_clip_embeddings(<br>&nbsp;&nbsp;&nbsp;&nbsp;input_dir<br>&nbsp;&nbsp;&nbsp;&nbsp;output_path<br>&nbsp;&nbsp;&nbsp;&nbsp;model_name)<br><br>get_clip_embedding(image_path, model_name)</div>"]

    %% 03_attributes
    N03["<b>Notebook</b><br>—<br>03_attributes.ipynb"]
    N03 --> A1["<b>Python</b><br>—<br>face_attributes.py"]
    A1 --> T3["<b>Tools</b><br>—<br>joblib<br>numpy<br>Pillow"]
    T3 --> A1M["<div style='text-align:left'><b>Methods</b><br>—<br>get_embedding(image_path)<br><br>infer_attributes(image_path)</div>"]

    %% 04_attractiveness_model
    N04["<b>Notebook</b><br>—<br>04_attractiveness_model.ipynb"]
    N04 --> M1["<b>Python</b><br>—<br>train_attractiveness.py"]
    M1 --> T4["<b>Tools</b><br>—<br>torch<br>polars"]
    T4 --> M1M["<div style='text-align:left'><b>Methods</b><br>—<br>run_training()<br><br>EmbeddingDataset<br>&nbsp;&nbsp;&nbsp;&nbsp;Not a file and not a standalone<br>&nbsp;&nbsp;&nbsp;&nbsp;method; you instantiate it,<br>&nbsp;&nbsp;&nbsp;&nbsp;then pass to DataLoader</div>"]

    %% 05_attractiveness_inference
    N05a["<b>Notebook</b><br>—<br>05_attractiveness_inference.ipynb"]
    N05a --> M2["<b>Python</b><br>—<br>attractiveness_model.py"]
    M2 --> T5["<b>Tools</b><br>—<br>torch.nn"]
    T5 --> M2M["<div style='text-align:left'><b>Methods</b><br>—<br>AttractivenessRegressor (class)</div>"]

    %% 05_composites / 06_composites
    N05b["<b>Notebook</b><br>—<br>05_composites.ipynb<br>06_composites.ipynb"]
    N05b --> C1["<b>Python</b><br>—<br>composite_generator.py"]
    C1 --> T6["<b>Tools</b><br>—<br>polars (optional)<br>numpy<br>Pillow<br>os<br>random"]
    T6 --> C1M["<div style='text-align:left'><b>Methods</b><br>—<br>filter_images(<br>&nbsp;&nbsp;&nbsp;&nbsp;df<br>&nbsp;&nbsp;&nbsp;&nbsp;gender<br>&nbsp;&nbsp;&nbsp;&nbsp;ethnicity<br>&nbsp;&nbsp;&nbsp;&nbsp;age_range<br>&nbsp;&nbsp;&nbsp;&nbsp;attractiveness<br>&nbsp;&nbsp;&nbsp;&nbsp;min_results<br>&nbsp;&nbsp;&nbsp;&nbsp;verbose)<br><br>make_composite(<br>&nbsp;&nbsp;&nbsp;&nbsp;df_or_paths<br>&nbsp;&nbsp;&nbsp;&nbsp;n<br>&nbsp;&nbsp;&nbsp;&nbsp;random_sample<br>&nbsp;&nbsp;&nbsp;&nbsp;outname<br>&nbsp;&nbsp;&nbsp;&nbsp;verbose)<br><br>filter_faces(<br>&nbsp;&nbsp;&nbsp;&nbsp;df<br>&nbsp;&nbsp;&nbsp;&nbsp;gender<br>&nbsp;&nbsp;&nbsp;&nbsp;ethnicity<br>&nbsp;&nbsp;&nbsp;&nbsp;age_range<br>&nbsp;&nbsp;&nbsp;&nbsp;attractiveness<br>&nbsp;&nbsp;&nbsp;&nbsp;min_results<br>&nbsp;&nbsp;&nbsp;&nbsp;verbose)<br><br>generate_composite(<br>&nbsp;&nbsp;&nbsp;&nbsp;df<br>&nbsp;&nbsp;&nbsp;&nbsp;gender<br>&nbsp;&nbsp;&nbsp;&nbsp;ethnicity<br>&nbsp;&nbsp;&nbsp;&nbsp;age_range<br>&nbsp;&nbsp;&nbsp;&nbsp;attractiveness<br>&nbsp;&nbsp;&nbsp;&nbsp;sample_size<br>&nbsp;&nbsp;&nbsp;&nbsp;min_results<br>&nbsp;&nbsp;&nbsp;&nbsp;verbose)</div>"]

    %% 05_ethnicity_clusters + metadata build (kept as-is)
    N05c["<b>Notebook</b><br>—<br>05_ethnicity_clusters.ipynb"]
    Glue["<b>Python</b><br>—<br>metadata/build_master.py"]
    N05c --> D1
    Glue --> D1
    D1["<b>Files</b><br>—<br>parquet files (no direct src helpers)"]
    D1 --> T78["<b>Tools</b><br>—<br>polars<br>parquet outputs"]

    class N01,N02,N03,N04,N05a,N05b,N05c notebook;
    class P1,E1,A1,M1,M2,C1,Glue python;
    class P1M,E1M,A1M,M1M,M2M,C1M methods;
    class T1,T2,T3,T4,T5,T6,T78 tools;
    class D1 files;
```


## High-Level Pipeline
```mermaid
flowchart LR
    A["📁 Raw Images<br>data/raw/"] --> B["🧹 Preprocess<br>src/pipeline/preprocess.py → data/processed/preproc/"]
    B --> C["🧠 CLIP Embeddings<br>notebooks/02_embeddings.ipynb → data/processed/embeddings/embeddings_clip.parquet"]
    C --> D["👥 Attributes (sklearn)<br>notebooks/03_attributes.ipynb → data/processed/metadata/attributes.parquet"]
    C --> E["💚 Attractiveness Train<br>notebooks/04_attractiveness_model.ipynb → models/attractiveness_regressor.pt"]
    C --> F["🏃 Attractiveness Inference<br>notebooks/05_attractiveness_inference.ipynb → data/processed/metadata/attractiveness_scores.parquet"]
    D --> G["📊 Enriched Metadata<br>clean/flags/meta/clusters/manual/predictions"]
    F --> G
    G --> H["🎨 Composites & Viz<br>notebooks/05_composites.ipynb, 06_composites.ipynb"]
```

## Embeddings → Attributes
```mermaid
flowchart LR
    A["🖼️ Preprocessed JPGs<br>data/processed/preproc/"] --> B["🤗 CLIPProcessor + CLIPModel<br>src/embeddings/embed_clip.py"]
    B --> C["📄 embeddings_clip.parquet<br>data/processed/embeddings/"]
    C --> D["🎯 sklearn classifiers<br>src/attributes/face_attributes.py"]
    D --> E["📄 attributes.parquet"]
    E --> F["✅ attributes_clean.parquet / 🚩 flags / 🌍 clusters / 🔗 meta/manual/predictions"]
```

## Attractiveness Scoring
```mermaid
flowchart LR
    A["🔢 embeddings_clip.parquet"] --> B["🧠 Train MLP<br>notebooks/04_attractiveness_model.ipynb"]
    B --> C["💾 models/attractiveness_regressor.pt"]
    A --> D["🏃 Inference<br>notebooks/05_attractiveness_inference.ipynb"]
    C --> D
    D --> E["📄 data/processed/metadata/attractiveness_scores.parquet"]
    E --> F["🔗 attractiveness_with_attributes.parquet"]
```

## Composites
```mermaid
flowchart LR
    A["📊 Filtered metadata<br>polars/pandas"] --> B["🗂️ Filenames"]
    B --> C["🖼️ Load images<br>data/processed/preproc/"]
    C --> D["➕ Stack & mean<br>numpy (see src/composite/composite_generator.py)"]
    D --> E["🖼️ Composite image<br>data/processed/composites/"]
```

## Repo Layout (current)
```text
FaceStats/
├── config/
├── data/
│   ├── raw/
│   │   └── fairface/
│   ├── processed/
│   │   ├── preproc/ (image outputs)
│   │   ├── embeddings/
│   │   │   └── embeddings_clip.parquet
│   │   ├── metadata/
│   │   │   ├── attributes.parquet
│   │   │   ├── attributes_clean.parquet
│   │   │   ├── attributes_final.parquet
│   │   │   ├── attributes_flags.parquet
│   │   │   ├── attributes_with_clusters.parquet
│   │   │   ├── attributes_with_meta.parquet
│   │   │   ├── attributes_with_manual.parquet
│   │   │   ├── attributes_with_predictions.parquet
│   │   │   ├── attractiveness_scores.parquet
│   │   │   ├── attractiveness_with_attributes.parquet
│   │   │   ├── fairface_label_structure.parquet
│   │   │   ├── feature_index.json
│   │   │   ├── labels_template.csv
│   │   │   └── manual_labels.csv
│   │   ├── composites/ (e.g., composite_v4_example.jpg)
│   │   └── attractiveness_scores.npy
│   ├── interim/
│   │   ├── checkpoints/
│   │   └── preprocessed/
│   ├── embeddings/
│   ├── models/
│   └── attributes/ (legacy)
├── models/
│   ├── attractiveness_regressor.pt
│   └── gender_clf.pkl
├── models_insightface/
├── notebooks/
│   ├── 01_preprocess.ipynb
│   ├── 02_embeddings.ipynb
│   ├── 03_attributes.ipynb
│   ├── 04_visualize_attributes.ipynb
│   ├── 04_attractiveness_model.ipynb
│   ├── 05_attractiveness_inference.ipynb
│   ├── 05_ethnicity_clusters.ipynb
│   ├── 05_composites.ipynb
│   ├── 06_composites.ipynb
│   └── data/
├── src/
│   ├── pipeline/
│   ├── embeddings/
│   ├── attributes/
│   ├── models/
│   ├── composite/
│   │   └── composite_generator.py
│   ├── data_utils/
│   ├── learning/
│   ├── metadata/
│   └── visualization/
├──current_files.md
├── LICENSE
├── README.md
├── requirements.txt
├── schematics_4_0.ipynb
├── schematics_4_0.md
└── tools_summary.md
```
