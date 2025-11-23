# ✨ FaceStats Tooling Snapshot

Quick, colorful glance at the stack powering preprocessing, embeddings, attributes, and reporting.

## 🌈 Core Libraries
- 🐍 Python 3.x with `requirements.txt`
- 🔥 `torch` + 🤗 `transformers` for CLIP/ViT embeddings + attribute models
- 🧮 `polars`, `numpy`, `sklearn` (PCA)
- 🖼️ `Pillow` (+ 🌀 `opencv-python` optional) for I/O, resizing, alignment helpers
- ⏱️ `tqdm` progress; 📊 `matplotlib`/`seaborn`; 🗒️ `nbformat` for notebook tweaks

## 🧭 Pipelines at a Glance
- 🧹 Preprocess: load → normalize → resize/alignment → `data/preprocessed/`
- 🧠 Embeddings: CLIP/ViT forward pass → L2 normalize → `embeddings.parquet`
- 👥 Attributes: HF image-classification pipelines → `attributes.parquet`
- 💚 Attractiveness: small MLP regressor → `scores.parquet`
- 📊 Metadata: merge embeddings/attributes/scores → `master.parquet`
- 🎨 Composites/Analysis: filter metadata, stack images, PCA/means, render composites/reports

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
        A1["🎯 HF Pipelines"]
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
