"""
explainability.py — Rule-based explanation + SHAP feature importance display.
"""
import numpy as np
import pandas as pd

from utils.feature_extractor import FEATURE_COLS, intent_summary, META_FEATURE_COLS


INTENT_LABELS = {
    "kw_temporal":     "Temporal / time-series intent",
    "kw_rank":         "Ranking intent",
    "kw_compare":      "Comparison intent",
    "kw_proportion":   "Proportional breakdown intent",
    "kw_distribution": "Distribution / spread intent",
    "kw_correlation":  "Correlation / relationship intent",
    "kw_geographic":   "Geographic intent",
    "kw_segment":      "Segmentation intent",
    "kw_anomaly":      "Anomaly / outlier intent",
    "kw_prediction":   "Predictive intent",
}

META_LABELS = {
    "n_numeric":        "Numeric columns",
    "n_categorical":    "Categorical columns",
    "n_datetime":       "Datetime columns",
    "max_cardinality":  "Max category cardinality",
    "mean_correlation": "Mean inter-variable correlation",
    "missing_ratio":    "Missing value ratio",
    "has_temporal":     "Temporal data present",
    "has_geo":          "Geographic data present",
}

VIZ_REASONING = {
    "bar_chart": [
        ("n_categorical ≥ 1",  "Categorical variable detected — bar chart handles discrete labels well"),
        ("n_numeric ≥ 1",      "Numeric measure available for bar height"),
        ("kw_rank",            "Ranking intent detected in your question"),
        ("kw_compare",         "Comparison intent detected in your question"),
        ("max_cardinality",    "Cardinality supports readable bar chart"),
    ],
    "line_chart": [
        ("has_temporal",       "Datetime variable detected — line chart is ideal for time series"),
        ("kw_temporal",        "Temporal intent detected: 'trend', 'over time', etc."),
        ("n_numeric ≥ 1",      "Numeric measure available for the y-axis"),
    ],
    "scatter_plot": [
        ("n_numeric ≥ 2",      "Two or more numeric variables enable scatter axes"),
        ("kw_correlation",     "Relationship / correlation intent detected"),
        ("mean_correlation",   "Variables show non-trivial correlation signal"),
    ],
    "pie_chart": [
        ("n_categorical ≥ 1",  "Categorical variable detected"),
        ("kw_proportion",      "Proportional breakdown intent detected"),
        ("max_cardinality ≤ 8","Low cardinality — pie chart remains readable"),
    ],
    "histogram": [
        ("n_numeric ≥ 1",      "Single numeric variable available for distribution"),
        ("kw_distribution",    "Distribution intent detected in your question"),
        ("n_categorical = 0",  "No grouping variable — histogram suits univariate analysis"),
    ],
    "box_plot": [
        ("n_numeric ≥ 1",      "Numeric variable with spread to display"),
        ("kw_distribution",    "Variability / spread intent detected"),
        ("n_categorical ≥ 1",  "Grouping variable enables comparison across boxes"),
    ],
    "heatmap": [
        ("n_numeric ≥ 5",      "Many numeric variables — heatmap shows all pairwise correlations"),
        ("kw_correlation",     "Correlation intent detected"),
        ("mean_correlation",   "Variables show inter-correlation signal"),
    ],
    "choropleth": [
        ("has_geo",            "Geographic variable detected"),
        ("kw_geographic",      "Geographic intent detected: 'map', 'region', 'country', etc."),
        ("n_numeric ≥ 1",      "Numeric measure available to colour the map"),
    ],
}


def explain_recommendation(profile: dict, query: str, viz_type: str) -> dict:
    """
    Return structured explanation for why viz_type was recommended.
    """
    intent = intent_summary(query)
    reasons = []

    # Intent-based reasons
    for kw, active in intent.items():
        if active:
            reasons.append({
                "type":  "intent",
                "label": INTENT_LABELS.get(kw, kw),
                "value": "detected",
                "positive": True,
            })

    # Meta-feature reasons
    feature_highlights = {
        "has_temporal":    (profile.get("has_temporal", 0) == 1,    "Temporal data present"),
        "has_geo":         (profile.get("has_geo", 0) == 1,         "Geographic variable detected"),
        "n_numeric":       (profile.get("n_numeric", 0) >= 2,       f"{profile.get('n_numeric',0)} numeric columns"),
        "n_categorical":   (profile.get("n_categorical", 0) >= 1,   f"{profile.get('n_categorical',0)} categorical columns"),
        "max_cardinality": (profile.get("max_cardinality", 0) <= 12,f"Cardinality: {profile.get('max_cardinality',0)}"),
    }
    for key, (cond, label) in feature_highlights.items():
        if cond:
            reasons.append({"type": "meta", "label": label, "value": "✓", "positive": True})

    # Viz-specific reasoning
    viz_reasons = VIZ_REASONING.get(viz_type, [])

    return {
        "intent_flags": intent,
        "reasons":      reasons,
        "viz_reasons":  viz_reasons,
        "profile":      profile,
    }


def shap_bar_data(models: dict, x_vec: np.ndarray) -> pd.DataFrame | None:
    """
    Compute mean |SHAP| values if shap is available and RF is loaded.
    Returns a DataFrame with columns [feature, importance].
    """
    try:
        import shap
        rf = models.get("rf")
        if rf is None:
            return None
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(x_vec)
        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)[0]
        elif shap_vals.ndim == 3:
            mean_abs = np.abs(shap_vals).mean(axis=2)[0]
        else:
            mean_abs = np.abs(shap_vals)[0]
        df = pd.DataFrame({"feature": FEATURE_COLS, "importance": mean_abs})
        return df.sort_values("importance", ascending=False).head(12)
    except Exception:
        return None
