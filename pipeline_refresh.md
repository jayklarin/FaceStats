# Refreshing the pipeline for new images

1) **Prep raw images**  
   - Drop the new JPGs into `data/raw/` (or a subfolder you point the notebooks to).  
   - If you want only the new set, move/archive previous batches or clean old outputs (see step 5).

2) **Preprocess (01_preprocess.ipynb)**  
   - Input: `data/raw/`  
   - Output: `data/processed/preproc/` resized/aligned JPGs.  
   - Verify the image count matches your expected ~10,000.

3) **Embeddings (02_embeddings.ipynb)**  
   - Input: `data/processed/preproc/`  
   - Output: `data/processed/embeddings/embeddings_clip.parquet`  
   - Check row count matches preproc images.

4) **Attributes (03_attributes.ipynb)**  
   - Input: preproc + embeddings via `infer_attributes`  
   - Output: `data/processed/metadata/attributes.parquet` (and derived clean/flags files if the notebook builds them).  
   - Confirm counts match embeddings.

5) **FaceMesh geometry (03_attributes_facemesh.ipynb, optional)**  
   - Input: `data/processed/preproc/`  
   - Output: `data/processed/metadata/attributes_with_meta.parquet` (adds geometry metrics / attractiveness_geom / attractiveness_dist).  
   - If you want to avoid mixing with old data, delete/rename the existing parquet before rerun.

6) **Attractiveness scoring (if using MLP)**  
   - Run any scoring notebook that writes `data/processed/metadata/attractiveness_scores.parquet`.  
   - Ensure joins in composites/visualization pick up the new parquet.

7) **Merge/clean (if applicable)**  
   - If you have a master merge step, rerun it after the above so downstream notebooks load fresh metadata.

8) **Downstream notebooks**  
   - `04_visualize_attributes.ipynb`: re-run with updated metadata; plots should reflect ~10k rows.  
   - `05_composites.ipynb` / `06_composites.ipynb`: re-run filters/composites; set paths to the refreshed parquet files.

9) **Sanity checks**  
   - Row counts: embeddings, attributes, facemesh metadata all ~10,000.  
   - Spot-check a few filenames across parquet outputs.  
   - If counts stay ~700, you’re likely still reading old parquet files—remove/rename them and rerun steps 2–5.
