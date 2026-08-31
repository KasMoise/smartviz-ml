import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SmartViz-ML · Explainability", page_icon="🔎", layout="wide")

from utils.recommender import load_models, VIZ_META
from utils.feature_extractor import build_feature_vector, FEATURE_COLS
from utils.explainability import explain_recommendation, shap_bar_data

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#1e293b; }
.page-sub   { color:#64748b; font-size:0.95rem; margin-bottom:1.5rem; }
.reason-row {
    display:flex; align-items:flex-start; gap:0.75rem;
    padding:0.75rem 0; border-bottom:1px solid #f1f0ff;
}
.reason-row:last-child { border-bottom:none; }
.reason-icon { font-size:1.1rem; flex-shrink:0; margin-top:0.1rem; }
.reason-label { font-weight:600; color:#1e293b; font-size:0.88rem; }
.reason-sub   { color:#64748b; font-size:0.8rem; margin-top:0.1rem; }
.explain-card {
    background:white; border-radius:14px; padding:1.5rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(108,99,255,0.07);
    border:1px solid #f1f0ff; margin-bottom:1rem;
}
.feature-row { display:flex; align-items:center; justify-content:space-between;
    padding:0.5rem 0; border-bottom:1px solid #f8f8ff; font-size:0.85rem; }
.feature-row:last-child { border-bottom:none; }
.feature-name { color:#475569; font-weight:500; }
.feature-val  { font-weight:700; color:#1e293b; }
.shap-label   { font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; font-weight:600; margin-bottom:0.5rem; }
.chain-step   { display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem; }
.chain-box    { background:#f8f7ff; border:1px solid #e2e0ff; border-radius:10px;
    padding:0.5rem 1rem; font-size:0.85rem; font-weight:600; color:#5b52d6; flex:1; text-align:center; }
.chain-arrow  { color:#c7d2fe; font-size:1.2rem; }
.no-data-msg  { background:#fff7ed; border:1px solid #fed7aa; border-radius:12px;
    padding:1.5rem; text-align:center; color:#c2410c; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🔎 Explainability</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Understand <em>why</em> SmartViz recommended a particular visualization — feature contributions, intent detection, and SHAP values.</div>', unsafe_allow_html=True)

# ── Check state ───────────────────────────────────────────────────────────────
recs        = st.session_state.get("recs")
profile     = st.session_state.get("profile")
active_query= st.session_state.get("active_query", "")

if not recs or not profile:
    st.markdown("""
    <div class="no-data-msg">
        <div style="font-size:2rem">💡</div>
        <div style="font-weight:600;margin-top:0.5rem">No recommendation yet</div>
        <div style="font-size:0.85rem;margin-top:0.25rem">Go to <strong>Visualization Recommender</strong> and ask a question first.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

top_rec = recs[0]
explanation = explain_recommendation(profile, active_query, top_rec["viz_type"])
models = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:linear-gradient(135deg,#f8f7ff,#f0f9ff);border-radius:14px;padding:1.25rem 1.5rem;border-left:4px solid #6C63FF;margin-bottom:1.5rem">'
    f'<div style="color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Top recommendation</div>'
    f'<div style="font-size:1.5rem;margin:0.25rem 0">{top_rec["icon"]} <strong>{top_rec["label"]}</strong></div>'
    f'<div style="color:#6C63FF;font-weight:700;font-size:1.1rem">{top_rec["confidence"]}% confidence</div>'
    f'<div style="color:#64748b;font-size:0.85rem;margin-top:0.35rem">Query: <em>"{active_query}"</em></div>'
    f'</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1])

with left:
    # ── Detected intents ──────────────────────────────────────────────────
    st.markdown("#### Detected analytical intents")
    intent_label_map = {
        "kw_temporal":    ("⏱", "Temporal intent",    "Words like 'trend', 'over time', 'monthly', 'evolution'"),
        "kw_rank":        ("🏆","Ranking intent",     "Words like 'top', 'best', 'worst', 'rank', 'highest'"),
        "kw_compare":     ("⚖️","Comparison intent",  "Words like 'compare', 'versus', 'difference', 'between'"),
        "kw_proportion":  ("🥧","Proportion intent",  "Words like 'share', 'percentage', 'breakdown', 'fraction'"),
        "kw_distribution":("📊","Distribution intent","Words like 'distribution', 'spread', 'range', 'outlier'"),
        "kw_correlation": ("🔗","Correlation intent", "Words like 'correlation', 'relationship', 'link', 'scatter'"),
        "kw_geographic":  ("🗺️","Geographic intent",  "Words like 'map', 'region', 'country', 'location'"),
        "kw_segment":     ("👥","Segmentation intent","Words like 'segment', 'cluster', 'group', 'profile'"),
        "kw_anomaly":     ("🚨","Anomaly intent",     "Words like 'anomaly', 'unusual', 'suspicious', 'extreme'"),
        "kw_prediction":  ("🔮","Prediction intent",  "Words like 'predict', 'forecast', 'future', 'next'"),
    }

    rows_html = ""
    for kw, (icon, label, hint) in intent_label_map.items():
        detected = explanation["intent_flags"].get(kw, False)
        color    = "#22c55e" if detected else "#e2e8f0"
        txt_col  = "#1e293b" if detected else "#94a3b8"
        rows_html += (
            f'<div class="reason-row">'
            f'<span style="font-size:1.1rem;color:{color}">'
            f'{"✓" if detected else "○"}</span>'
            f'<div><div class="reason-label" style="color:{txt_col}">{icon} {label}</div>'
            f'<div class="reason-sub">{hint}</div></div>'
            f'</div>'
        )
    st.markdown(f'<div class="explain-card">{rows_html}</div>', unsafe_allow_html=True)

with right:
    # ── Dataset meta-features ─────────────────────────────────────────────
    st.markdown("#### Dataset meta-features")
    meta_display = [
        ("Numeric columns",      profile["n_numeric"],                   "n_numeric"),
        ("Categorical columns",  profile["n_categorical"],               "n_categorical"),
        ("Datetime columns",     profile["n_datetime"],                  "n_datetime"),
        ("Max cardinality",      f'{profile["max_cardinality"]:,}',      "max_cardinality"),
        ("Mean correlation",     f'{profile["mean_correlation"]:.3f}',   "mean_correlation"),
        ("Missing ratio",        f'{profile["missing_ratio"]:.1%}',      "missing_ratio"),
        ("Temporal data",        "Yes ✓" if profile["has_temporal"] else "No", "has_temporal"),
        ("Geographic variable",  "Yes ✓" if profile["has_geo"]      else "No", "has_geo"),
    ]
    rows_html = "".join(
        f'<div class="feature-row">'
        f'<span class="feature-name">{name}</span>'
        f'<span class="feature-val">{val}</span>'
        f'</div>'
        for name, val, _ in meta_display
    )
    st.markdown(f'<div class="explain-card">{rows_html}</div>', unsafe_allow_html=True)

    # ── Recommendation-specific reasoning ──────────────────────────────────
    st.markdown(f"#### Why {top_rec['label']}?")
    from utils.explainability import VIZ_REASONING
    reasons = VIZ_REASONING.get(top_rec["viz_type"], [])
    reason_html = "".join(
        f'<div class="reason-row">'
        f'<span class="reason-icon" style="color:#6C63FF">▸</span>'
        f'<div><div class="reason-label">{r[0]}</div>'
        f'<div class="reason-sub">{r[1]}</div></div>'
        f'</div>'
        for r in reasons
    )
    st.markdown(f'<div class="explain-card">{reason_html}</div>', unsafe_allow_html=True)

st.divider()

# ── SHAP feature importance ───────────────────────────────────────────────────
st.markdown("#### SHAP Feature Importance")
st.markdown("*Mean |SHAP value| across all visualization classes — higher = more influential in the recommendation.*")

x_vec = build_feature_vector(profile, active_query)
shap_df = shap_bar_data(models, x_vec)

if shap_df is not None and not shap_df.empty:
    fig_shap = px.bar(
        shap_df, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale=["#c7d2fe","#6C63FF","#312e81"],
        labels={"importance": "Mean |SHAP value|", "feature": ""},
    )
    fig_shap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="Inter,sans-serif", size=11),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(gridcolor="#f0f0f8"),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    # Fallback: show feature vector as a simple bar
    feat_labels = FEATURE_COLS
    feat_vals   = x_vec[0]
    fallback_df = pd.DataFrame({"Feature": feat_labels, "Value": feat_vals})
    fallback_df = fallback_df[fallback_df["Value"] > 0].sort_values("Value", ascending=False)

    if not fallback_df.empty:
        fig_fb = px.bar(
            fallback_df, x="Value", y="Feature", orientation="h",
            color="Value", color_continuous_scale=["#c7d2fe","#6C63FF"],
            labels={"Value": "Feature value (active = 1)", "Feature": ""},
        )
        fig_fb.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0),
            font=dict(family="Inter,sans-serif",size=11),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(gridcolor="#f0f0f8"),
        )
        st.plotly_chart(fig_fb, use_container_width=True)
        st.caption("SHAP model not loaded — showing active feature flags. Export and load `smartviz_rf.joblib` for full SHAP explanation.")
    else:
        st.info("No active features detected for SHAP display.")

st.divider()

# ── Full decision chain ───────────────────────────────────────────────────────
st.markdown("#### Full decision chain")

chain = [
    ("📂 Dataset",          f'{profile["n_rows"]:,} rows · {profile["n_cols"]} cols'),
    ("🔢 Meta-features",   f'{profile["n_numeric"]} num · {profile["n_categorical"]} cat · {"temporal" if profile["has_temporal"] else "no time"} · {"geo" if profile["has_geo"] else "no geo"}'),
    ("🔤 Intent keywords", ", ".join(k.replace("kw_","") for k,v in explanation["intent_flags"].items() if v) or "none detected"),
    ("🤖 Random Forest",   f'18-dim feature vector → {len(list(__import__("utils.recommender", fromlist=["VIZ_META"]).VIZ_META))} class probabilities'),
    (f'{top_rec["icon"]} {top_rec["label"]}', f'{top_rec["confidence"]}% confidence'),
]

for step_label, step_val in chain:
    st.markdown(
        f'<div class="chain-step">'
        f'<div class="chain-box"><strong>{step_label}</strong><br>'
        f'<span style="font-weight:400;font-size:0.78rem;color:#6b7280">{step_val}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if step_label != chain[-1][0]:
        st.markdown('<div style="color:#c7d2fe;text-align:center;font-size:1.2rem;margin:-0.25rem 0">↓</div>',
                    unsafe_allow_html=True)
