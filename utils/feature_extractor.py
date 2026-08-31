"""
feature_extractor.py — Build the 18-dim feature vector for SmartViz-ML.
Mirrors encode_query() + META_FEATURE_COLS from the research notebook.
"""
import re
import numpy as np

META_FEATURE_COLS = [
    "n_numeric", "n_categorical", "n_datetime", "max_cardinality",
    "mean_correlation", "missing_ratio", "has_temporal", "has_geo",
]

INTENT_KEYWORDS = {
    "kw_temporal":     r"trend|over time|evolution|monthly|weekly|daily|forecast|growth|decline|quarter|annual|year",
    "kw_rank":         r"top|best|worst|rank|most|least|highest|lowest|leading|bottom",
    "kw_compare":      r"compare|comparison|versus|vs\b|between|difference|differ",
    "kw_proportion":   r"proportion|share|percentage|part|breakdown|contribution|fraction",
    "kw_distribution": r"distribution|spread|range|histogram|frequency|quartile|outlier|variability",
    "kw_correlation":  r"correlation|relationship|association|scatter|link|related",
    "kw_geographic":   r"map|region|country|geographic|location|where|territory|province",
    "kw_segment":      r"segment|cluster|group|profile|category|tier",
    "kw_anomaly":      r"anomaly|outlier|unusual|suspicious|extreme|abnormal|flag",
    "kw_prediction":   r"predict|forecast|future|next|expect|project",
}

FEATURE_COLS = META_FEATURE_COLS + list(INTENT_KEYWORDS.keys())


def encode_query(query: str) -> dict:
    q = query.lower()
    return {kw: int(bool(re.search(pat, q))) for kw, pat in INTENT_KEYWORDS.items()}


def build_feature_vector(profile: dict, query: str) -> np.ndarray:
    """
    Combine dataset meta-features + intent keywords into the 18-dim
    feature vector expected by the trained Random Forest.
    """
    meta = [
        profile["n_numeric"],
        profile["n_categorical"],
        profile["n_datetime"],
        profile["max_cardinality"],
        profile["mean_correlation"],
        profile["missing_ratio"],
        profile["has_temporal"],
        profile["has_geo"],
    ]
    intent = list(encode_query(query).values())
    return np.array(meta + intent, dtype=float).reshape(1, -1)


def intent_summary(query: str) -> dict:
    """Human-readable intent flags for the Explainability page."""
    q = query.lower()
    return {kw: bool(re.search(pat, q)) for kw, pat in INTENT_KEYWORDS.items()}
