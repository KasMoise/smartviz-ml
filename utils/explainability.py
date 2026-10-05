"""
explainability.py — Permutation Importance (replaces SHAP).
Mirrors notebook §35: uses sklearn.inspection.permutation_importance.
No external SHAP dependency — works reliably on Streamlit Cloud.
"""
import numpy as np
import pandas as pd
from utils.feature_extractor import FEATURE_COLS, intent_summary

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

VIZ_REASONING = {
    "bar_chart":    [("Categorical variable detected", "Bar chart handles discrete labels well"),
                     ("Numeric measure available", "Used as bar height / length"),
                     ("Ranking or comparison intent", "Detected in your question")],
    "line_chart":   [("Datetime variable detected", "Line chart is ideal for time series"),
                     ("Temporal intent detected", "Words like 'trend', 'over time', 'monthly'"),
                     ("Numeric measure available", "Used as the y-axis")],
    "scatter_plot": [("Two or more numeric variables", "Enable scatter axes"),
                     ("Correlation intent detected", "Words like 'relationship', 'link', 'association'"),
                     ("Non-trivial correlation signal", "Variables show some co-movement")],
    "pie_chart":    [("Categorical variable detected", "Provides the slice labels"),
                     ("Proportion intent detected", "Words like 'share', 'percentage', 'breakdown'"),
                     ("Low cardinality", "Pie chart remains readable with few categories")],
    "histogram":    [("Single numeric variable", "Available for distribution analysis"),
                     ("Distribution intent detected", "Words like 'distribution', 'spread', 'range'"),
                     ("No grouping variable", "Histogram suits univariate analysis")],
    "box_plot":     [("Numeric variable with spread", "Enables box-and-whisker display"),
                     ("Variability intent detected", "Words like 'variability', 'spread', 'outlier'"),
                     ("Grouping variable available", "Enables comparison across boxes")],
    "heatmap":      [("Many numeric variables (≥5)", "Heatmap shows all pairwise correlations"),
                     ("Correlation intent detected", "Words like 'correlate', 'matrix', 'pairwise'"),
                     ("Inter-variable correlation signal", "Variables show mutual dependency")],
    "choropleth":   [("Geographic variable detected", "Country / region column found"),
                     ("Geographic intent detected", "Words like 'map', 'region', 'country'"),
                     ("Numeric measure available", "Used to colour the map regions")],
}


def explain_recommendation(profile: dict, query: str, viz_type: str) -> dict:
    """Return structured explanation for a recommendation."""
    intent = intent_summary(query)
    reasons = [
        {"type": "intent", "label": INTENT_LABELS.get(kw, kw), "positive": True}
        for kw, active in intent.items() if active
    ]
    feature_highlights = {
        "has_temporal":   (profile.get("has_temporal", 0) == 1,    "Temporal data present"),
        "has_geo":        (profile.get("has_geo", 0) == 1,         "Geographic variable detected"),
        "n_numeric":      (profile.get("n_numeric", 0) >= 2,       f"{profile.get('n_numeric',0)} numeric columns"),
        "n_categorical":  (profile.get("n_categorical", 0) >= 1,   f"{profile.get('n_categorical',0)} categorical columns"),
        "max_cardinality":(profile.get("max_cardinality", 0) <= 12,f"Cardinality: {profile.get('max_cardinality',0)}"),
    }
    for key, (cond, label) in feature_highlights.items():
        if cond:
            reasons.append({"type": "meta", "label": label, "positive": True})
    return {
        "intent_flags": intent,
        "reasons":      reasons,
        "viz_reasons":  VIZ_REASONING.get(viz_type, []),
        "profile":      profile,
    }


def permutation_importance_data(models: dict, X_test: np.ndarray,
                                 y_test: np.ndarray) -> pd.DataFrame | None:
    """
    Compute permutation importance on Pool C test set.
    Mirrors notebook §35: n_repeats=30, scoring='accuracy'.
    Returns DataFrame with columns [feature, importance, std].
    """
    try:
        from sklearn.inspection import permutation_importance
        import pandas as pd

        rf = models.get("rf")
        le = models.get("le")
        if rf is None or le is None or X_test is None or y_test is None:
            return None

        # Cap to 200 samples for speed on the web
        n = min(200, len(X_test))
        np.random.seed(42)
        idx = np.random.choice(len(X_test), n, replace=False)
        X_s, y_s = X_test[idx], y_test[idx]

        result = permutation_importance(
            rf, X_s, y_s,
            n_repeats=10,        # reduced for web (notebook uses 30)
            random_state=42,
            n_jobs=1,
            scoring="accuracy",
        )
        df = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": result.importances_mean,
            "std":        result.importances_std,
        })
        return df.sort_values("importance", ascending=False).head(12)
    except Exception:
        return None


def impurity_importance_data(models: dict) -> pd.DataFrame | None:
    """
    Gini (impurity) importance from the RF model.
    Available immediately — no test data needed.
    Mirrors the second panel in notebook §35.
    """
    try:
        import pandas as pd
        rf = models.get("rf")
        if rf is None:
            return None
        core = rf[-1] if hasattr(rf, "named_steps") else rf
        if not hasattr(core, "feature_importances_"):
            return None
        df = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": core.feature_importances_,
        })
        return df.sort_values("importance", ascending=False).head(12)
    except Exception:
        return None
