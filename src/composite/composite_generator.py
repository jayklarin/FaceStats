def filter_faces(
    df,
    gender=None,
    ethnicity=None,
    age_range=None,          # tuple/list: (min_age, max_age)
    attractiveness=None,     # list/range: e.g. [6] or [6, 7, 8]
    min_results=12,          # minimum required faces before fallback triggers
    verbose=False
):
    """
    Robust filter function that guarantees non-empty output using structured fallbacks.
    Handles sparse metadata (unknown gender, unknown ethnicity, null ages).

    Fallback sequence:
        1. Apply strict filters
        2. If < min_results → relax age, then gender, then ethnicity
        3. If still sparse → attractiveness-only sampling
        4. Final fallback → return entire dataframe
    """

    original_df = df.copy()

    # ---------------------------------------
    # Initial strict filter stage
    # ---------------------------------------
    filtered = df.copy()

    if gender is not None:
        filtered = filtered[filtered["gender_final"] == gender]

    if ethnicity is not None:
        filtered = filtered[filtered["ethnicity_final"] == ethnicity]

    if age_range is not None:
        lo, hi = age_range
        filtered = filtered[(filtered["age"] >= lo) & (filtered["age"] <= hi)]

    if attractiveness is not None:
        if len(attractiveness) == 1:
            filtered = filtered[filtered["attractiveness"] == attractiveness[0]]
        elif len(attractiveness) == 2:
            lo, hi = attractiveness
            filtered = filtered[
                (filtered["attractiveness"] >= lo) &
                (filtered["attractiveness"] <= hi)
            ]

    # If strict filtering already returned enough samples → done
    if len(filtered) >= min_results:
        if verbose:
            print(f"[filter_faces] Strict filter returned {len(filtered)} results.")
        return filtered

    # ---------------------------------------
    # FALLBACK 1 — Ignore Age
    # ---------------------------------------
    if age_range is not None:
        tmp = original_df.copy()
        if gender is not None:
            tmp = tmp[tmp["gender_final"] == gender]
        if ethnicity is not None:
            tmp = tmp[tmp["ethnicity_final"] == ethnicity]
        if attractiveness is not None:
            if len(attractiveness) == 1:
                tmp = tmp[tmp["attractiveness"] == attractiveness[0]]
            else:
                lo, hi = attractiveness
                tmp = tmp[
                    (tmp["attractiveness"] >= lo) &
                    (tmp["attractiveness"] <= hi)
                ]

        if len(tmp) >= min_results:
            if verbose:
                print(f"[fallback] Age removed → {len(tmp)} results.")
            return tmp

    # ---------------------------------------
    # FALLBACK 2 — Ignore Gender
    # ---------------------------------------
    if gender is not None:
        tmp = original_df.copy()
        if ethnicity is not None:
            tmp = tmp[tmp["ethnicity_final"] == ethnicity]
        if attractiveness is not None:
            if len(attractiveness) == 1:
                tmp = tmp[tmp["attractiveness"] == attractiveness[0]]
            else:
                lo, hi = attractiveness
                tmp = tmp[
                    (tmp["attractiveness"] >= lo) &
                    (tmp["attractiveness"] <= hi)
                ]

        if len(tmp) >= min_results:
            if verbose:
                print(f"[fallback] Gender removed → {len(tmp)} results.")
            return tmp

    # ---------------------------------------
    # FALLBACK 3 — Ignore Ethnicity
    # ---------------------------------------
    if ethnicity is not None:
        tmp = original_df.copy()
        if attractiveness is not None:
            if len(attractiveness) == 1:
                tmp = tmp[tmp["attractiveness"] == attractiveness[0]]
            else:
                lo, hi = attractiveness
                tmp = tmp[
                    (tmp["attractiveness"] >= lo) &
                    (tmp["attractiveness"] <= hi)
                ]

        if len(tmp) >= min_results:
            if verbose:
                print(f"[fallback] Ethnicity removed → {len(tmp)} results.")
            return tmp

    # ---------------------------------------
    # FALLBACK 4 — Attractiveness-only filter
    # ---------------------------------------
    if attractiveness is not None:
        tmp = original_df.copy()
        if len(attractiveness) == 1:
            tmp = tmp[tmp["attractiveness"] == attractiveness[0]]
        else:
            lo, hi = attractiveness
            tmp = tmp[
                (tmp["attractiveness"] >= lo) &
                (tmp["attractiveness"] <= hi)
            ]

        if len(tmp) >= min_results:
            if verbose:
                print(f"[fallback] Using attractiveness-only filter → {len(tmp)}")
            return tmp

    # ---------------------------------------
    # FINAL FALLBACK — Use everything
    # ---------------------------------------
    if verbose:
        print("[fallback] All filters failed. Using full dataset.")

    return original_df

import os
import numpy as np
from PIL import Image

def generate_composite(
    df,
    gender=None,
    ethnicity=None,
    age_range=None,
    attractiveness=None,
    sample_size=20,
    min_results=12,
    verbose=False
):
    """
    Generate a face composite using filtered faces.
    Steps:
        1. Filter available faces using fallback logic
        2. Randomly sample N faces
        3. Load images from disk
        4. Convert to numpy arrays (float32)
        5. Average pixel values
        6. Return final composite as a PIL.Image

    Requirements:
        df must include 'filename' column
        images must exist in data/processed/preproc/
    """

    # -------------------------------
    # 1. Apply filtering
    # -------------------------------
    filtered = filter_faces(
        df,
        gender=gender,
        ethnicity=ethnicity,
        age_range=age_range,
        attractiveness=attractiveness,
        min_results=min_results,
        verbose=verbose
    )

    if verbose:
        print(f"[generate_composite] Filtered pool: {len(filtered)} faces")

    # -------------------------------
    # 2. Sample faces
    # -------------------------------
    if len(filtered) < sample_size:
        sample_size = len(filtered)  # safety fallback

    sampled = filtered.sample(sample_size, random_state=42)

    if verbose:
        print(f"[generate_composite] Sampling {sample_size} faces")

    # -------------------------------
    # 3. Load images
    # -------------------------------
    images = []
    PREPROC_DIR = "data/processed/preproc"

    for fname in sampled["filename"].tolist():
        path = os.path.join(PREPROC_DIR, fname)

        if not os.path.exists(path):
            if verbose:
                print(f"[WARNING] Missing image: {path}")
            continue

        try:
            img = Image.open(path).convert("RGB")
            images.append(np.array(img, dtype=np.float32))
        except Exception as e:
            if verbose:
                print(f"[ERROR] Could not load {path}: {e}")

    if len(images) == 0:
        raise ValueError("No images loaded; composite failed.")

    # -------------------------------
    # 4. Average the images
    # -------------------------------
    avg = np.mean(np.stack(images), axis=0)
    avg = np.clip(avg, 0, 255).astype(np.uint8)

    # -------------------------------
    # 5. Return PIL image
    # -------------------------------
    return Image.fromarray(avg)


# --------------------------------------------------------------------
# Public API expected by notebooks
# --------------------------------------------------------------------
def filter_images(
    df=None,
    gender=None,
    ethnicity=None,
    age_range=None,
    attractiveness=None,
    min_results=12,
    verbose=False,
):
    """
    Wrapper around filter_faces that accepts either pandas or polars DataFrames.
    If df is None, load attributes + attractiveness scores from disk using polars.
    """
    try:
        import polars as pl  # optional dependency; only used if provided df is polars or df is None
    except ImportError:
        pl = None

    # Lazily load default dataset if none is provided
    if df is None:
        if pl is None:
            raise ImportError("polars is required to auto-load attributes data.")
        df_attr = pl.read_parquet("data/processed/metadata/attributes_clean.parquet")
        df_score = pl.read_parquet("data/processed/metadata/attractiveness_scores.parquet")
        df = df_attr.join(df_score, on="filename")

    was_polars = pl is not None and isinstance(df, pl.DataFrame)
    pdf = df.to_pandas() if was_polars else df.copy()

    filtered = filter_faces(
        pdf,
        gender=gender,
        ethnicity=ethnicity,
        age_range=age_range,
        attractiveness=attractiveness,
        min_results=min_results,
        verbose=verbose,
    )

    if was_polars:
        return pl.from_pandas(filtered)
    return filtered


def make_composite(
    df_or_paths,
    n=20,
    random_sample=True,
    outname=None,
    verbose=False,
):
    """
    Create a composite image from either:
      - a DataFrame (pandas or polars) containing a 'filename' column, or
      - an iterable of filename strings.

    Parameters:
        n (int|None): number of images to include; None means use all.
        random_sample (bool): whether to randomly sample when n is set.
        outname (str|None): if provided, save composite to data/processed/composites/<outname>.
    """
    try:
        import polars as pl  # optional; only used for type detection
    except ImportError:
        pl = None

    # Normalize input into a list of filenames
    if isinstance(df_or_paths, (list, tuple)):
        filenames = list(df_or_paths)
    elif pl is not None and isinstance(df_or_paths, pl.DataFrame):
        filenames = df_or_paths["filename"].to_list()
    else:
        # Assume pandas-like DataFrame
        filenames = df_or_paths["filename"].tolist()

    if not filenames:
        raise ValueError("No filenames provided for composite generation.")

    if n is not None:
        n = min(n, len(filenames))
        if random_sample:
            import random
            random.seed(42)
            filenames = random.sample(filenames, n)
        else:
            filenames = filenames[:n]

    images = []
    PREPROC_DIR = "data/processed/preproc"

    for fname in filenames:
        path = os.path.join(PREPROC_DIR, fname)
        if not os.path.exists(path):
            if verbose:
                print(f"[WARNING] Missing image: {path}")
            continue
        try:
            img = Image.open(path).convert("RGB")
            images.append(np.array(img, dtype=np.float32))
        except Exception as e:
            if verbose:
                print(f"[ERROR] Could not load {path}: {e}")

    if len(images) == 0:
        raise ValueError("No images loaded; composite failed.")

    avg = np.mean(np.stack(images), axis=0)
    avg = np.clip(avg, 0, 255).astype(np.uint8)
    composite = Image.fromarray(avg)

    if outname:
        out_dir = "data/processed/composites"
        os.makedirs(out_dir, exist_ok=True)
        composite.save(os.path.join(out_dir, outname))
        if verbose:
            print(f"[make_composite] Saved → {os.path.join(out_dir, outname)}")

    return composite
