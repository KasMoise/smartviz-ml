"""
SmartViz-ML — Core Engine
Shared logic for meta-feature extraction, intent encoding,
rule-based baseline, and model inference.
Imported by both app.py and the training script.
"""

import re
import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
VIZ_TYPES = [
    "line_chart", "bar_chart", "scatter_plot", "pie_chart",
    "histogram", "box_plot", "heatmap", "choropleth",
]

VIZ_LABELS = {
    "line_chart":   "Line Chart",
    "bar_chart":    "Bar Chart",
    "scatter_plot": "Scatter Plot",
    "pie_chart":    "Pie Chart",
    "histogram":    "Histogram",
    "box_plot":     "Box Plot",
    "heatmap":      "Heatmap",
    "choropleth":   "Choropleth Map",
}

SEGMENT_NAMES = ["VIP", "Loyal", "Potential", "Occasional", "At-Risk"]

VIZ_ICONS = {
    "line_chart":   "📈",
    "bar_chart":    "📊",
    "scatter_plot": "🔵",
    "pie_chart":    "🥧",
    "histogram":    "📉",
    "box_plot":     "📦",
    "heatmap":      "🌡️",
    "choropleth":   "🗺️",
}

META_FEATURE_COLS = [
    "n_numeric", "n_categorical", "n_datetime", "max_cardinality",
    "mean_correlation", "missing_ratio", "has_temporal", "has_geo",
]

INTENT_KEYWORDS = {
    "kw_temporal":     r"trend|over time|evolution|monthly|weekly|daily|forecast|growth|decline",
    "kw_rank":         r"top|best|worst|rank|most|least|highest|lowest|leading",
    "kw_compare":      r"compare|comparison|versus|vs\b|between|difference",
    "kw_proportion":   r"proportion|share|percentage|part|breakdown|contribution",
    "kw_distribution": r"distribution|spread|range|histogram|frequency|quartile|outlier",
    "kw_correlation":  r"correlation|relationship|association|scatter|link",
    "kw_geographic":   r"map|region|country|geographic|location|where",
    "kw_segment":      r"segment|cluster|group|profile|category",
    "kw_anomaly":      r"anomaly|outlier|unusual|suspicious|extreme|abnormal",
    "kw_prediction":   r"predict|forecast|future|next|expect",
}

FEATURE_COLS = META_FEATURE_COLS + list(INTENT_KEYWORDS.keys())

KEYWORD_MAP = {
    r"trend|over time|evolution|monthly|weekly|daily|forecast": ["line_chart"],
    r"top|best|worst|rank|most|least|leading":                  ["bar_chart"],
    r"correlation|relationship|versus|vs\b|link|associate":     ["scatter_plot", "heatmap"],
    r"proportion|share|percentage|part|breakdown":              ["pie_chart", "bar_chart"],
    r"distribution|spread|range|quartile|outlier|skew":         ["histogram", "box_plot"],
    r"segment|cluster|group|profile":                           ["bar_chart", "scatter_plot"],
    r"map|region|country|geographic|location":                  ["choropleth"],
    r"anomal|unusual|suspicious|extreme|abnormal":              ["scatter_plot", "box_plot"],
    r"matrix|pairwise|all variables|overview":                  ["heatmap"],
}

VIZ_INSIGHTS = {
    "line_chart": {
        "title": "Trend Analysis",
        "desc": "Reveals how a metric evolves over time. Look for seasonality, inflection points, and anomalous periods.",
        "when": "Use when you have a time dimension and want to show change.",
        "caution": "Avoid when data is not time-ordered or intervals are irregular.",
    },
    "bar_chart": {
        "title": "Ranking / Comparison",
        "desc": "Compares discrete categories by a quantitative measure. Ideal for identifying top performers.",
        "when": "Use when comparing groups or ranking items.",
        "caution": "Truncating the y-axis can exaggerate differences.",
    },
    "scatter_plot": {
        "title": "Relationship Exploration",
        "desc": "Shows joint distribution of two numeric variables. Reveals correlations and clusters.",
        "when": "Use when exploring co-movement between two continuous variables.",
        "caution": "Correlation does not imply causation.",
    },
    "pie_chart": {
        "title": "Proportional Breakdown",
        "desc": "Shows part-to-whole relationships for a small number of categories.",
        "when": "Use when you have ≤6 categories and want to show composition.",
        "caution": "Avoid with many categories — use a bar chart instead.",
    },
    "histogram": {
        "title": "Distribution Profiling",
        "desc": "Shows the frequency distribution of a single numeric variable. Reveals skewness and outliers.",
        "when": "Use to understand the shape and spread of your data.",
        "caution": "Bin width choice significantly affects interpretation.",
    },
    "box_plot": {
        "title": "Spread & Outlier Review",
        "desc": "Summarises distribution via quartiles. Excellent for comparing variability across groups.",
        "when": "Use when comparing spread across multiple groups.",
        "caution": "Hides the underlying distribution shape.",
    },
    "heatmap": {
        "title": "Correlation Matrix",
        "desc": "Shows pairwise relationships across many variables simultaneously.",
        "when": "Use to scan for inter-variable correlations in multi-dimensional data.",
        "caution": "High correlation may indicate redundancy, not causation.",
    },
    "choropleth": {
        "title": "Geographic Overview",
        "desc": "Maps a metric onto geographic regions. Reveals spatial patterns and regional disparities.",
        "when": "Use when your data has a meaningful geographic dimension.",
        "caution": "Results depend heavily on the geographic aggregation level chosen.",
    },
}


# ── Feature extraction ───────────────────────────────────────────────────────

def encode_query(query: str) -> dict:
    q = query.lower()
    return {kw: int(bool(re.search(pat, q))) for kw, pat in INTENT_KEYWORDS.items()}


def extract_meta_features(df: pd.DataFrame) -> dict:
    num_cols  = df.select_dtypes("number").columns.tolist()
    cat_cols  = df.select_dtypes(["object", "category", "string"]).columns.tolist()
    date_cols = df.select_dtypes("datetime").columns.tolist()
    max_card  = max((df[c].nunique() for c in cat_cols), default=0)
    if len(num_cols) >= 2:
        cm = df[num_cols].corr().abs()
        mc = float(np.mean([cm.loc[r, c] for r in cm.index for c in cm.columns if r != c]))
    else:
        mc = 0.0
    return {
        "n_numeric":        len(num_cols),
        "n_categorical":    len(cat_cols),
        "n_datetime":       len(date_cols),
        "max_cardinality":  max_card,
        "mean_correlation": mc,
        "missing_ratio":    float(df.isnull().mean().mean()),
        "has_temporal":     int(len(date_cols) > 0),
        "has_geo":          int(any("country" in c.lower() for c in df.columns)),
    }


def build_feature_vector(meta: dict, query: str) -> np.ndarray:
    kw = encode_query(query)
    vec = [meta.get(f, 0) for f in META_FEATURE_COLS] + list(kw.values())
    return np.array(vec, dtype=float).reshape(1, -1)


def feature_vector_df(meta: dict, query: str) -> pd.DataFrame:
    return pd.DataFrame(build_feature_vector(meta, query), columns=FEATURE_COLS)


# ── Rule-based baseline ──────────────────────────────────────────────────────

def rule_rank(meta: dict, query: str) -> list:
    scores = {v: 0.0 for v in VIZ_TYPES}
    q = query.lower()
    if meta.get("has_temporal", 0): scores["line_chart"]  += 3.0
    if meta.get("has_geo", 0):      scores["choropleth"]  += 3.5
    nn   = meta.get("n_numeric", 0)
    nc   = meta.get("n_categorical", 0)
    corr = meta.get("mean_correlation", 0)
    card = meta.get("max_cardinality", 0)
    if nn >= 5:              scores["heatmap"]      += 2.5
    if nn >= 2:              scores["scatter_plot"] += 2.0 + corr;  scores["heatmap"] += 1.5
    if nn == 1 and nc == 0:  scores["histogram"]    += 2.0;         scores["box_plot"] += 1.5
    if nc >= 1 and nn >= 1:  scores["bar_chart"]    += 2.5
    if card <= 6 and nc >= 1: scores["pie_chart"]   += 1.0
    if card > 10:             scores["pie_chart"]   -= 2.5
    for pat, charts in KEYWORD_MAP.items():
        if re.search(pat, q):
            for ch in charts:
                scores[ch] += 2.0
    return sorted(VIZ_TYPES, key=lambda v: scores[v], reverse=True)


# ── Model inference ──────────────────────────────────────────────────────────

def ml_rank(model, le, meta: dict, query: str) -> list:
    xdf   = feature_vector_df(meta, query)
    proba = model.predict_proba(xdf)[0]
    return [le.classes_[i] for i in np.argsort(proba)[::-1]], proba


def recommend(model, le, meta: dict, query: str, top_n: int = 3) -> list:
    ranked, proba = ml_rank(model, le, meta, query)
    results = []
    for viz in ranked[:top_n]:
        idx   = list(le.classes_).index(viz)
        score = float(proba[idx])
        info  = VIZ_INSIGHTS.get(viz, {})
        results.append({
            "viz":     viz,
            "label":   VIZ_LABELS.get(viz, viz),
            "icon":    VIZ_ICONS.get(viz, ""),
            "score":   score,
            "pct":     round(score * 100, 1),
            "title":   info.get("title", ""),
            "desc":    info.get("desc", ""),
            "when":    info.get("when", ""),
            "caution": info.get("caution", ""),
        })
    return results
