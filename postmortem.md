# Postmortem: Streamlit Attractiveness Scorer Face Detection + Scoring Drift

## What happened
- The Streamlit app was designed to auto-crop faces before scoring. Locally this worked because mediapipe/opencv were available; on Streamlit Cloud they were missing due to system `libGL` issues and missing packages. That led to center-crops or inconsistent detector fallbacks.
- Score deciles were derived from the training reference parquet. Without consistent crops and without the parquet on cloud, uploads often collapsed to decile 1.
- Detector status/verbosity in the UI was noisy and not helpful to end users.

## Impact
- Cloud users saw all 1s (or near-1) for images outside the training set.
- Crops differed between local and cloud, causing different embeddings/scores for the same image.
- UI showed internal detector warnings.

## Root causes
- Missing/failed detectors on Streamlit Cloud (mediapipe/opencv need `libGL`; facenet-pytorch wasn’t installed).
- Scoring relied on reference parquet; when missing, deciles fell back poorly.
- Detector status surfaced raw dependency errors to users.

## Fixes implemented
- Added GL-free MTCNN detector (facenet-pytorch) and prefer it first; keep mediapipe/opencv as fallbacks.
- Added baked-in training min/max raw score fallback for decile mapping when the parquet is unavailable.
- Suppressed detector status caption in the UI to avoid clutter.
- Gated scoring when no detector is available to avoid garbage crops.

## Remaining risks / next steps
- Ensure Streamlit Cloud install includes `facenet-pytorch==2.5.3` (or mediapipe with `libGL`) so detectors are available; otherwise scoring is blocked.
- Optionally ship `data/processed/metadata/attractiveness_scores.parquet` with the deploy for exact decile alignment.
- If further consistency is needed, lock a single detector (MTCNN) everywhere and drop the others.
