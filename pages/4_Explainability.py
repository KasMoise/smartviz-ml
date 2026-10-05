import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="SmartViz-ML · Explainability", page_icon="🔎", layout="wide")

from utils.recommender import load_models, get_best_model_name, VIZ_META
from utils.feature_extractor import build_feature_vector, FEATURE_COLS
from utils.explainability import (explain_recommendation, permutation_importance_data,
                                   impurity_importance_data, VIZ_REASONING)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#1e293b; }
.page-sub   { color:#64748b; font-size:0.95rem; margin-bottom:1.5rem; }
.explain-card { background:white; border-radius:14px; padding:1.5rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(108,99,255,0.07);
    border:1px solid #f1f0ff; margin-bottom:1rem; }
.reason-row { display:flex; align-items:flex-start; gap:0.75rem;
    padding:0.65rem 0; border-bottom:1px solid #f1f0ff; }
.reason-row:last-child { border-bottom:none; }
.reason-label { font-weight:600; color:#1e293b; font-size:0.88rem; }
.reason-sub   { color:#64748b; font-size:0.8rem; margin-top:0.1rem; }
.feature-row  { display:flex; align-items:center; justify-content:space-between;
    padding:0.5rem 0; border-bottom:1px solid #f8f8ff; font-size:0.85rem; }
.feature-row:last-child { border-bottom:none; }
.feature-name { color:#475569; font-weight:500; }
.feature-val  { font-weight:700; color:#1e293b; }
.chain-box    { background:#f8f7ff; border:1px solid #e2e0ff; border-radius:10px;
    padding:0.5rem 1rem; font-size:0.85rem; font-weight:600; color:#5b52d6; text-align:center; }
.no-data-msg  { background:#fff7ed; border:1px solid #fed7aa; border-radius:12px;
    padding:1.5rem; text-align:center; color:#c2410c; }
.badge-pi { background:#d1fae5; color:#065f46; border-radius:6px;
    padding:0.2rem 0.6rem; font-size:0.75rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🔎 Explainability</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Understand <em>why</em> SmartViz recommended a particular visualization — permutation importance, impurity importance, intent detection, and the full decision chain.</div>', unsafe_allow_html=True)

recs         = st.session_state.get("recs")
profile      = st.session_state.get("profile")
active_query = st.session_state.get("active_query", "")
models       = load_models()
model_name   = get_best_model_name(models)

if not recs or not profile:
    st.markdown('<div class="no-data-msg"><div style="font-size:2rem">💡</div><div style="font-weight:600;margin-top:0.5rem">No recommendation yet</div><div style="font-size:0.85rem;margin-top:0.25rem">Go to <strong>Visualization Recommender</strong> first.</div></div>', unsafe_allow_html=True)
    st.stop()

top_rec     = recs[0]
explanation = explain_recommendation(profile, active_query, top_rec["viz_type"])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:linear-gradient(135deg,#f8f7ff,#f0f9ff);border-radius:14px;'
    f'padding:1.25rem 1.5rem;border-left:4px solid #6C63FF;margin-bottom:1.5rem">'
    f'<div style="color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Top recommendation</div>'
    f'<div style="font-size:1.5rem;margin:0.25rem 0">{top_rec["icon"]} <strong>{top_rec["label"]}</strong></div>'
    f'<div style="color:#6C63FF;font-weight:700;font-size:1.1rem">{top_rec["confidence"]}% confidence</div>'
    f'<div style="color:#64748b;font-size:0.85rem;margin-top:0.35rem">Model: <strong>{model_name}</strong> · Query: <em>"{active_query}"</em></div>'
    f'</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1])

with left:
    # ── Detected intents ──────────────────────────────────────────────────
    st.markdown("#### Detected analytical intents")
    intent_map = {
        "kw_temporal":    ("⏱", "Temporal intent",     "trend, over time, monthly, evolution"),
        "kw_rank":        ("🏆","Ranking intent",       "top, best, worst, rank, highest"),
        "kw_compare":     ("⚖️","Comparison intent",    "compare, versus, difference, between"),
        "kw_proportion":  ("🥧","Proportion intent",    "share, percentage, breakdown, fraction"),
        "kw_distribution":("📊","Distribution intent",  "distribution, spread, range, outlier"),
        "kw_correlation": ("🔗","Correlation intent",   "correlation, relationship, link, scatter"),
        "kw_geographic":  ("🗺️","Geographic intent",    "map, region, country, location"),
        "kw_segment":     ("👥","Segmentation intent",  "segment, cluster, group, profile"),
        "kw_anomaly":     ("🚨","Anomaly intent",       "anomaly, unusual, suspicious, extreme"),
        "kw_prediction":  ("🔮","Prediction intent",    "predict, forecast, future, next"),
    }
    rows_html = ""
    for kw, (icon, label, hint) in intent_map.items():
        detected = explanation["intent_flags"].get(kw, False)
        color    = "#22c55e" if detected else "#e2e8f0"
        txt_col  = "#1e293b" if detected else "#94a3b8"
        rows_html += (
            f'<div class="reason-row"><span style="font-size:1.1rem;color:{color}">{"✓" if detected else "○"}</span>'
            f'<div><div class="reason-label" style="color:{txt_col}">{icon} {label}</div>'
            f'<div class="reason-sub">{hint}</div></div></div>'
        )
    st.markdown(f'<div class="explain-card">{rows_html}</div>', unsafe_allow_html=True)

with right:
    # ── Dataset meta-features ─────────────────────────────────────────────
    st.markdown("#### Dataset meta-features")
    meta_rows = [
        ("Numeric columns",     profile["n_numeric"]),
        ("Categorical columns", profile["n_categorical"]),
        ("Datetime columns",    profile["n_datetime"]),
        ("Max cardinality",     f'{profile["max_cardinality"]:,}'),
        ("Mean correlation",    f'{profile["mean_correlation"]:.3f}'),
        ("Missing ratio",       f'{profile["missing_ratio"]:.1%}'),
        ("Temporal data",       "Yes ✓" if profile["has_temporal"] else "No"),
        ("Geographic variable", "Yes ✓" if profile["has_geo"]      else "No"),
    ]
    rows_html = "".join(
        f'<div class="feature-row"><span class="feature-name">{n}</span><span class="feature-val">{v}</span></div>'
        for n, v in meta_rows
    )
    st.markdown(f'<div class="explain-card">{rows_html}</div>', unsafe_allow_html=True)

    # ── Why this chart ────────────────────────────────────────────────────
    st.markdown(f"#### Why {top_rec['label']}?")
    reasons = VIZ_REASONING.get(top_rec["viz_type"], [])
    reason_html = "".join(
        f'<div class="reason-row"><span style="color:#6C63FF;font-size:1.1rem">▸</span>'
        f'<div><div class="reason-label">{r[0]}</div><div class="reason-sub">{r[1]}</div></div></div>'
        for r in reasons
    )
    st.markdown(f'<div class="explain-card">{reason_html}</div>', unsafe_allow_html=True)

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("#### Feature Importance")
st.markdown(
    '<span class="badge-pi">✓ Permutation Importance (sklearn) — mirrors notebook §35</span>'
    ' — how much accuracy drops when each feature is randomly shuffled.',
    unsafe_allow_html=True,
)

tab_perm, tab_gini = st.tabs(["📉 Permutation Importance", "🌳 Impurity (Gini) Importance"])

with tab_perm:
    # Try permutation importance using stored Pool C test data
    X_test = st.session_state.get("X_test_b")
    y_test = st.session_state.get("y_test_b")
    perm_df = permutation_importance_data(models, X_test, y_test)

    if perm_df is not None and not perm_df.empty:
        fig = px.bar(
            perm_df, x="importance", y="feature", orientation="h",
            error_x="std",
            color="importance",
            color_continuous_scale=["#c7d2fe","#6C63FF","#312e81"],
            labels={"importance": "Mean accuracy drop when shuffled", "feature": ""},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0),
            font=dict(family="Inter,sans-serif",size=11),
            yaxis=dict(autorange="reversed"), xaxis=dict(gridcolor="#f0f0f8"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"n_repeats=10, n_samples≤200, scoring='accuracy' · Model: {model_name}")
    else:
        # Fallback: show active feature flags for this query
        x_vec   = build_feature_vector(profile, active_query)[0]
        feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Active": x_vec})
        feat_df = feat_df[feat_df["Active"] > 0].sort_values("Active", ascending=False)
        if not feat_df.empty:
            fig = px.bar(feat_df, x="Active", y="Feature", orientation="h",
                         color="Active", color_continuous_scale=["#c7d2fe","#6C63FF"],
                         labels={"Active": "Feature value", "Feature": ""})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0),
                               font=dict(family="Inter,sans-serif",size=11),
                               yaxis=dict(autorange="reversed"), xaxis=dict(gridcolor="#f0f0f8"))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Full permutation importance requires Pool C test data (X_test_b, y_test_b) stored in session state from the notebook. Showing active feature flags instead.")

with tab_gini:
    gini_df = impurity_importance_data(models)
    if gini_df is not None and not gini_df.empty:
        fig = px.bar(
            gini_df, x="importance", y="feature", orientation="h",
            color="importance",
            color_continuous_scale=["#fde68a","#f59e0b","#92400e"],
            labels={"importance": "Mean decrease in impurity (Gini)", "feature": ""},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0),
            font=dict(family="Inter,sans-serif",size=11),
            yaxis=dict(autorange="reversed"), xaxis=dict(gridcolor="#f0f0f8"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Gini impurity importance from the trained {model_name}. Available immediately — no test data needed.")
    else:
        st.info("Gini importance requires the trained model file `smartviz_best_model.joblib`. Run §58 in the notebook to export it.")

st.divider()

# ── Full decision chain ───────────────────────────────────────────────────────
st.markdown("#### Full decision chain")
chain = [
    ("📂 Dataset",        f'{profile["n_rows"]:,} rows · {profile["n_cols"]} cols'),
    ("🔢 Meta-features",  f'{profile["n_numeric"]} num · {profile["n_categorical"]} cat · {"temporal" if profile["has_temporal"] else "no time"} · {"geo" if profile["has_geo"] else "no geo"}'),
    ("🔤 Intent keywords",", ".join(k.replace("kw_","") for k,v in explanation["intent_flags"].items() if v) or "none detected"),
    (f"🤖 {model_name}",  f"18-dim feature vector → {len(VIZ_META)} class probabilities"),
    (f'{top_rec["icon"]} {top_rec["label"]}', f'{top_rec["confidence"]}% confidence'),
]
for i, (label, val) in enumerate(chain):
    st.markdown(
        f'<div class="chain-box"><strong>{label}</strong><br>'
        f'<span style="font-weight:400;font-size:0.78rem;color:#6b7280">{val}</span></div>',
        unsafe_allow_html=True,
    )
    if i < len(chain) - 1:
        st.markdown('<div style="color:#c7d2fe;text-align:center;font-size:1.2rem;margin:-0.1rem 0">↓</div>', unsafe_allow_html=True)
