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
├── LICENSE
├── README.md
├── repo_map.md
├── requirements.txt
├── schematics.ipynb
├── schematics_4_0.ipynb
├── tools_summary.md
├── del.ipynb
└── x.ipynb
