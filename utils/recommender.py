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

    # Always compute rule-based scores — used for blending and override
    rule_ranked = _rule_based_fallback(profile, query)
    rule_scores = {v: s for v, s in rule_ranked}

    if "rf" in models and "le" in models:
        proba   = models["rf"].predict_proba(x_df)[0]
        classes = models["le"].classes_
        rf_scores = dict(zip(classes, proba))

        # ── Context-aware blending ────────────────────────────────────────
        # The RF trained on Pool-A (200 items) overweights keyword signals.
        # Strategy:
        #   • If the RF is overconfident on 1 class (top_prob >= 0.95):
        #       - Use rule-based ranking but display RF confidence for top pick
        #   • Otherwise: blend RF 60% + rules 40%
        import re as _re2
        top_prob = max(rf_scores.values())

        if top_prob >= 0.95:
            # RF is overconfident — use rules for ranking, RF for confidence
            ranked = rule_ranked
            # Assign RF confidence to the rule-based top pick, scale rest
            if ranked:
                top_rule_viz = ranked[0][0]
                top_rf_conf  = rf_scores.get(top_rule_viz, top_prob)
                # Scale all confidences relative to the top
                max_rule = ranked[0][1] or 1.0
                ranked = [(v, (s / max_rule) * max(top_rf_conf, 0.5))
                          for v, s in ranked]
                total_r = sum(s for _, s in ranked) or 1.0
                ranked  = [(v, s / total_r) for v, s in ranked]
        else:
            # Calibrated blend: RF 60% + rules 40%
            blended = {v: 0.6 * rf_scores.get(v, 0.0) + 0.4 * rule_scores.get(v, 0.0)
                       for v in VIZ_META}
            total_b = sum(blended.values()) or 1.0
            ranked  = sorted([(v, s/total_b) for v, s in blended.items()],
                              key=lambda t: t[1], reverse=True)
    else:
        ranked = rule_ranked

    # Normalise so top confidence is always in [40%, 99%] range
    scores_list = [s for _, s in ranked[:top_n]]
    max_s = max(scores_list) if scores_list else 1.0
    if max_s > 0:
        ranked_norm = [(v, s / max_s) for v, s in ranked]
    else:
        ranked_norm = ranked

    results = []
    for rank, (viz_type, conf) in enumerate(ranked_norm[:top_n], 1):
        # Map to [0%, 99%] — never show 100% unless truly certain
        display_conf = min(round(float(conf) * 99, 1), 99.0)
        meta = VIZ_META.get(viz_type, {"label": viz_type, "icon": "📊", "use": ""})
        results.append({
            "rank": rank, "viz_type": viz_type,
            "label": meta["label"], "icon": meta["icon"], "use": meta["use"],
            "confidence": display_conf,
        })
    return results


def _rule_based_fallback(profile: dict, query: str) -> list[tuple]:
    import re
    scores = {v: 0.0 for v in VIZ_META}
    q = query.lower()
    if profile.get("has_temporal"): scores["line_chart"] += 5.0
    if profile.get("has_geo"):      scores["choropleth"] += 6.0
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
        r"map|region|country|geographic|location|territoire":       ["choropleth"],
        r"anomal|unusual|suspicious|extreme":                       ["scatter_plot","box_plot"],
        r"matrix|pairwise|all variables":                           ["heatmap"],
    }.items():
        if re.search(pat, q):
            for ch in charts: scores[ch] += 2.0
    total = sum(scores.values()) or 1.0
    return sorted([(v, s/total) for v, s in scores.items()], key=lambda t: t[1], reverse=True)
