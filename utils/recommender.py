"""
recommender.py — Load trained models and run SmartViz inference.
Aligned with notebook §58 export filenames:
  smartviz_best_model.joblib     ← BEST_ML_MODEL
  smartviz_label_encoder.joblib  ← le_viz
  smartviz_rfm_scaler.joblib     ← scaler_rfm
  smartviz_meta.json             ← BEST_MODEL_NAME, metrics, FEATURE_COLS
"""
import json, os, joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils.feature_extractor import FEATURE_COLS, build_feature_vector

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

VIZ_META = {
    "bar_chart":    {"label": "Bar Chart",      "icon": "📊", "use": "Ranking, comparison across categories"},
    "line_chart":   {"label": "Line Chart",     "icon": "📈", "use": "Trends over time"},
    "scatter_plot": {"label": "Scatter Plot",   "icon": "🔵", "use": "Relationships between two numeric variables"},
    "pie_chart":    {"label": "Pie Chart",      "icon": "🥧", "use": "Proportional breakdown (few categories)"},
    "histogram":    {"label": "Histogram",      "icon": "📉", "use": "Distribution of a single numeric variable"},
    "box_plot":     {"label": "Box Plot",       "icon": "📦", "use": "Spread, variability, and outliers"},
    "heatmap":      {"label": "Heatmap",        "icon": "🌡️",  "use": "Correlations across many variables"},
    "choropleth":   {"label": "Choropleth Map", "icon": "🗺️",  "use": "Geographic distribution"},
}


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Load all trained artifacts once. Filenames match notebook §58."""
    paths = {
        "rf":       os.path.join(MODEL_DIR, "smartviz_best_model.joblib"),
        "le":       os.path.join(MODEL_DIR, "smartviz_label_encoder.joblib"),
        "scaler":   os.path.join(MODEL_DIR, "smartviz_rfm_scaler.joblib"),
        "meta":     os.path.join(MODEL_DIR, "smartviz_meta.json"),
        "forecast": os.path.join(MODEL_DIR, "forecasting_model.joblib"),
        "segment":  os.path.join(MODEL_DIR, "segmentation_model.joblib"),
        "anomaly":  os.path.join(MODEL_DIR, "anomaly_model.joblib"),
    }
    models = {}
    for key, path in paths.items():
        if os.path.exists(path):
            try:
                if path.endswith(".json"):
                    with open(path) as f:
                        models[key] = json.load(f)
                else:
                    models[key] = joblib.load(path)
            except Exception:
                pass
    return models


def get_best_model_name(models: dict) -> str:
    """Read BEST_MODEL_NAME from meta.json — never hardcoded."""
    return models.get("meta", {}).get("best_model_name", "Random Forest")


def recommend(profile: dict, query: str, top_n: int = 3) -> list[dict]:
    models = load_models()
    x_df   = pd.DataFrame(build_feature_vector(profile, query), columns=FEATURE_COLS)

    if "rf" in models and "le" in models:
        proba   = models["rf"].predict_proba(x_df)[0]
        classes = models["le"].classes_
        ranked  = sorted(zip(classes, proba), key=lambda t: t[1], reverse=True)
    else:
        ranked = _rule_based_fallback(profile, query)

    results = []
    for rank, (viz_type, conf) in enumerate(ranked[:top_n], 1):
        meta = VIZ_META.get(viz_type, {"label": viz_type, "icon": "📊", "use": ""})
        results.append({
            "rank": rank, "viz_type": viz_type,
            "label": meta["label"], "icon": meta["icon"], "use": meta["use"],
            "confidence": round(float(conf) * 100, 1),
        })
    return results


def _rule_based_fallback(profile: dict, query: str) -> list[tuple]:
    import re
    scores = {v: 0.0 for v in VIZ_META}
    q = query.lower()
    if profile.get("has_temporal"): scores["line_chart"] += 3.0
    if profile.get("has_geo"):      scores["choropleth"] += 3.5
    n_num = profile.get("n_numeric", 0)
    n_cat = profile.get("n_categorical", 0)
    corr  = profile.get("mean_correlation", 0)
    card  = profile.get("max_cardinality", 0)
    if n_num >= 5:             scores["heatmap"]      += 2.5
    if n_num >= 2:             scores["scatter_plot"] += 2.0 + corr; scores["heatmap"] += 1.5
    if n_num == 1 and n_cat == 0: scores["histogram"] += 2.0; scores["box_plot"] += 1.5
    if n_cat >= 1 and n_num >= 1: scores["bar_chart"] += 2.5
    if card <= 6 and n_cat >= 1:  scores["pie_chart"] += 1.0
    if card > 10:                 scores["pie_chart"] -= 2.5
    for pat, charts in {
        r"trend|over time|evolution|monthly|weekly|daily|forecast": ["line_chart"],
        r"top|best|worst|rank|most|least|leading":                  ["bar_chart"],
        r"correlation|relationship|versus|vs\b|scatter":            ["scatter_plot","heatmap"],
        r"proportion|share|percentage|part|breakdown":              ["pie_chart","bar_chart"],
        r"distribution|spread|range|outlier|quartile":              ["histogram","box_plot"],
        r"segment|cluster|group|profile":                           ["bar_chart","scatter_plot"],
        r"map|region|country|geographic|location":                  ["choropleth"],
        r"anomal|unusual|suspicious|extreme":                       ["scatter_plot","box_plot"],
        r"matrix|pairwise|all variables":                           ["heatmap"],
    }.items():
        if re.search(pat, q):
            for ch in charts: scores[ch] += 2.0
    total = sum(scores.values()) or 1.0
    return sorted([(v, s/total) for v, s in scores.items()], key=lambda t: t[1], reverse=True)
