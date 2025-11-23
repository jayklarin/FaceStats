# ✨ FaceStats Tooling Snapshot

Quick, colorful glance at the stack powering preprocessing, embeddings, attributes, and reporting.

## 🌈 Core Libraries
- 🐍 Python 3.x with `requirements.txt`
- 🔥 `torch` + 🤗 `transformers` for CLIP/ViT embeddings
- 🧮 `numpy`, `sklearn` (classifiers + PCA); `polars` in notebooks (optional install)
- 🖼️ `Pillow` (+ 🌀 `opencv-python` optional) for I/O, resizing, alignment helpers
- ⏱️ `tqdm` progress; 📊 `matplotlib`/`seaborn`; 🗒️ `nbformat` for notebook tweaks

## 🧭 Pipelines at a Glance
- 🧹 Preprocess: load → normalize → resize/alignment → `data/processed/preproc/`
- 🧠 Embeddings: CLIP/ViT forward pass → L2 normalize → `data/processed/embeddings/embeddings_clip.parquet`
- 👥 Attributes: CLIP embeddings → sklearn classifiers → `data/processed/metadata/attributes.parquet` (+ clean/flags/clusters/meta/manual/predictions variants)
- 💚 Attractiveness: MLP regressor → `data/processed/metadata/attractiveness_scores.parquet`
- 📊 Metadata: merges (e.g., `attributes_with_meta.parquet`, `attractiveness_with_attributes.parquet`)
- 🎨 Composites/Analysis: filter metadata, stack images, render composites/reports

## 🎛️ Tool Map (Mermaid)


```mermaid
flowchart LR
    subgraph Preprocess
        P1["🖼️ Pillow"]
        P2["🌀 OpenCV (optional)"]
        P3["🟢 Mediapipe FaceMesh"]
    end

    subgraph Embeddings
        E1["⚡ PyTorch"]
        E2["🤗 Transformers (CLIP/ViT)"]
    end

    subgraph Attributes
        A1["🎯 sklearn classifiers"]
    end

    subgraph Data
        D1["📐 Polars"]
        D2["📦 NumPy"]
    end

    subgraph Modeling
        M1["💚 PyTorch MLP"]
        M2["🔍 sklearn PCA"]
    end

    subgraph Viz
        V1["📊 Matplotlib/Seaborn"]
    end

    P1 & P2 & P3 --> E1
    E1 --> E2
    E2 --> A1
    E2 --> M1
    A1 --> D1
    M1 --> D1
    D1 --> M2
    D1 --> V1
```

## 📝 Notes
- 🖥️ CPU-first by default; plug in GPU-backed PyTorch if available.
- 🧩 No ONNX required; pure PyTorch + Transformers is the baseline.
- 🗂️ Keep paths consistent (`data/raw`, `data/preprocessed`, `embeddings.parquet`, `master.parquet`) to reuse notebooks without edits.
