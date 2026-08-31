"""
data_profiler.py — Compute dataset meta-features for SmartViz-ML.
"""
import numpy as np
import pandas as pd


GEO_HINTS = [
    "country", "region", "city", "state", "province", "territory",
    "location", "lat", "lon", "latitude", "longitude", "postal",
    "zip", "continent", "district", "prefecture",
]


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Return a rich profile dict with meta-features and display stats.
    Mirrors extract_meta_features() used in the research notebook.
    """
    n_rows, n_cols = df.shape
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Try to parse object columns that look like dates
    for col in cat_cols[:]:
        try:
            converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if converted.notna().mean() > 0.7:
                date_cols.append(col)
                cat_cols.remove(col)
        except Exception:
            pass

    max_cardinality = (
        max(df[c].nunique() for c in cat_cols) if cat_cols else 0
    )

    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().abs()
        corr_vals = [
            corr_matrix.loc[r, c]
            for r in corr_matrix.index
            for c in corr_matrix.columns
            if r != c and not np.isnan(corr_matrix.loc[r, c])
        ]
        mean_corr = float(np.mean(corr_vals)) if corr_vals else 0.0
    else:
        mean_corr = 0.0

    missing_ratio = float(df.isnull().mean().mean())

    has_temporal = int(len(date_cols) > 0)
    has_geo = int(
        any(
            any(hint in col.lower() for hint in GEO_HINTS)
            for col in df.columns
        )
    )

    # Missing per column
    missing_by_col = df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].to_dict()

    # Cardinalities for categoricals
    cardinalities = {c: int(df[c].nunique()) for c in cat_cols}

    return {
        # ── SmartViz meta-features (fed to the RF model) ──────────────
        "n_rows":           n_rows,
        "n_cols":           n_cols,
        "n_numeric":        len(num_cols),
        "n_categorical":    len(cat_cols),
        "n_datetime":       len(date_cols),
        "max_cardinality":  max_cardinality,
        "mean_correlation": round(mean_corr, 4),
        "missing_ratio":    round(missing_ratio, 4),
        "has_temporal":     has_temporal,
        "has_geo":          has_geo,
        # ── Display extras ────────────────────────────────────────────
        "num_cols":         num_cols,
        "cat_cols":         cat_cols,
        "date_cols":        date_cols,
        "missing_by_col":   missing_by_col,
        "cardinalities":    cardinalities,
        "missing_pct":      round(missing_ratio * 100, 1),
    }
