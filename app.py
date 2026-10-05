"""
SmartViz-ML — Main entry point.
Run with: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="SmartViz-ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "SmartViz-ML — Explainable ML for Business Visualization (v8)",
    },
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0e2e 0%, #1a1560 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #e2e0ff !important; }
[data-testid="stSidebar"] a { color: #c7d2fe !important; }
[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: rgba(108,99,255,0.25) !important;
    border-radius: 10px;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(108,99,255,0.15) !important;
    border-radius: 10px;
}

/* Brand in sidebar */
.sidebar-brand {
    font-family: 'Inter', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 0.5rem 0 0.25rem;
    letter-spacing: -0.01em;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: #6366f1 !important;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 0.75rem 0;
}
.sidebar-section {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6366f1 !important;
    padding: 0.25rem 0;
}
.sidebar-status {
    font-size: 0.75rem;
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    font-weight: 600;
    display: inline-block;
    margin-top: 0.25rem;
}
.status-ok   { background: rgba(34,197,94,0.15); color: #4ade80 !important; }
.status-warn { background: rgba(251,191,36,0.15); color: #fbbf24 !important; }

/* Main content area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Global button style */
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #6C63FF, #48CAE4);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    padding: 0.55rem 1.5rem;
    transition: opacity 0.2s ease, transform 0.15s ease;
}
.stButton > button[data-testid="baseButton-primary"]:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #6C63FF;
    border-bottom-color: #6C63FF;
}

/* Metrics */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #f1f0ff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    color: #6C63FF;
}

/* Plotly chart background */
.js-plotly-plot { border-radius: 12px; }

/* Inputs */
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    border-radius: 10px;
    border: 1.5px solid #e2e0ff;
    font-family: 'Inter', sans-serif;
}
.stTextInput input:focus, .stSelectbox select:focus {
    border-color: #6C63FF;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.12);
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #f1f0ff;
    border-radius: 12px;
    background: white;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f8f7ff; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6C63FF; }

/* Responsive tweaks */
@media (max-width: 768px) {
    .main .block-container { padding: 1rem 0.75rem 2rem; }
    [data-testid="stSidebar"] { min-width: 240px !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🧠 SmartViz-ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Explainable AI · Business Analytics</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Dataset status
    df      = st.session_state.get("df")
    profile = st.session_state.get("profile")

    st.markdown('<div class="sidebar-section">Dataset</div>', unsafe_allow_html=True)
    if df is not None and profile is not None:
        st.markdown(
            f'<span class="sidebar-status status-ok">'
            f'✓ {profile["n_rows"]:,} rows · {profile["n_cols"]} cols'
            f'</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="sidebar-status status-warn">No dataset loaded</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Model status
    from utils.recommender import load_models, get_best_model_name
    models = load_models()
    meta   = models.get("meta", {})
    mname  = get_best_model_name(models)
    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    if "rf" in models:
        st.markdown(
            f'<span class="sidebar-status status-ok">✓ {mname} loaded</span>',
            unsafe_allow_html=True,
        )
        if meta:
            st.markdown(
                f'<span style="font-size:0.72rem;color:#818cf8 !important">' +
                f'Pool C Acc@1 {meta.get("pool_c_acc1",0):.1%}' +
                f'</span>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<span class="sidebar-status status-warn">Demo mode — run §58 in notebook</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Last query
    last_q = st.session_state.get("active_query", "")
    if last_q:
        st.markdown('<div class="sidebar-section">Last question</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span style="font-size:0.78rem;color:#a5b4fc !important;font-style:italic">'
            f'"{last_q[:60]}{"…" if len(last_q)>60 else ""}"'
            f'</span>',
            unsafe_allow_html=True,
        )
        recs = st.session_state.get("recs")
        if recs:
            st.markdown(
                f'<span style="font-size:0.78rem;color:#67e8f9 !important">'
                f'→ {recs[0]["icon"]} {recs[0]["label"]} ({recs[0]["confidence"]}%)'
                f'</span>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:0.68rem;color:#4338ca !important">'
        'SmartViz-ML · Research prototype<br>'
        'Master candidate — UP Diliman, Philippines<br>'
        'BAEL framework</span>',
        unsafe_allow_html=True,
    )

# ── Home content ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.home-hero {
    background: linear-gradient(135deg, #0f0e2e 0%, #1a1560 60%, #0c4a6e 100%);
    border-radius: 24px;
    padding: clamp(2rem, 5vw, 3.5rem);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.home-hero::before {
    content: '';
    position: absolute;
    top: -40%; right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(108,99,255,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.home-title {
    font-family: 'Inter', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc 0%, #67e8f9 50%, #ffffff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}
.home-sub {
    color: #94a3b8;
    font-size: clamp(0.9rem, 2vw, 1.1rem);
    max-width: 600px;
    line-height: 1.6;
    margin-bottom: 2rem;
}
.home-cta {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, #6C63FF, #48CAE4);
    color: white !important; font-weight: 700; font-size: 0.95rem;
    border-radius: 12px; padding: 0.75rem 1.75rem;
    text-decoration: none; transition: opacity 0.2s ease, transform 0.15s ease;
}
.home-cta:hover { opacity: 0.9; transform: translateY(-1px); }
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.feature-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #f1f0ff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 16px rgba(108,99,255,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(108,99,255,0.14);
}
.feature-icon { font-size: 2rem; margin-bottom: 0.75rem; }
.feature-title { font-weight: 700; color: #1e293b; font-size: 0.95rem; margin-bottom: 0.35rem; }
.feature-desc  { color: #64748b; font-size: 0.82rem; line-height: 1.5; }
.paper-banner {
    background: linear-gradient(135deg, #fdf4ff, #faf5ff);
    border: 1px solid #e9d5ff;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 2rem;
}
.paper-icon { font-size: 2rem; flex-shrink: 0; }
.paper-title { font-weight: 700; color: #6b21a8; font-size: 0.95rem; }
.paper-sub   { color: #94a3b8; font-size: 0.8rem; margin-top: 0.15rem; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="home-hero">
    <div class="home-title">SmartViz-ML</div>
    <div class="home-sub">
        Explainable Machine Learning for Automated Business Visualization
        and Decision Support — Ask a question in plain English, get the
        right chart with a scientific explanation.
    </div>
    <div style="display:flex;gap:0.75rem;flex-wrap:wrap">
        <a class="home-cta" href="/Data_Upload">📂 Upload your data →</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Paper banner
st.markdown("""
<div class="paper-banner">
    <div class="paper-icon">📄</div>
    <div>
        <div class="paper-title">SmartViz-ML: An Explainable Machine Learning Framework for Automated Business Visualization and Decision Support</div>
        <div class="paper-sub">Research paper in preparation · BAEL Doctoral Framework · UP Diliman, Philippines</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature cards
st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
features = [
    ("📂", "Data Profiling",         "Auto-detect columns, types, cardinality, correlations and geographic variables from any CSV or Excel file."),
    ("💡", "Visualization Recommender","Ask a question in plain English. The trained Random Forest ranks all 8 chart types by probability."),
    ("📊", "Chart Generation",        "See the actual chart generated from your data — not just a label, a real interactive Plotly figure."),
    ("🔎", "Explainability (SHAP)",   "Understand why each visualization was recommended via SHAP feature importance and intent detection."),
    ("📋", "Business Insights",       "Get hedged, calibrated plain-language summaries of what the data suggests."),
    ("📝", "Human Evaluation Form",   "Collect real evaluator ratings (Likert 1–5) for the paper's human study — exports to CSV."),
    ("📈", "Sales Forecasting",       "Rolling time-series forecasting using the trained model from the research notebook."),
    ("🚨", "Anomaly Detection",       "Isolation Forest flags statistically unusual transactions for manual review."),
]
for icon, title, desc in features:
    st.markdown(
        f'<div class="feature-card">'
        f'<div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{title}</div>'
        f'<div class="feature-desc">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)

# Model results table — primary model highlighted dynamically from meta.json
import pandas as pd
from utils.recommender import load_models, get_best_model_name
_models  = load_models()
_meta    = _models.get("meta", {})
_mname   = get_best_model_name(_models)
_acc1    = _meta.get("pool_c_acc1", 0.9500) if _meta else 0.9500
_mrr     = _meta.get("pool_c_mrr",  0.9688) if _meta else 0.9688

st.markdown(f"### 📊 Model performance (Pool C — 300 independent queries)")

perf_data = {
    "Model":      [f"{_mname} ★","LightGBM","XGBoost","Logistic Regression","Rule-based baseline"],
    "Accuracy@1": [_acc1,  0.9068, 0.8941, 0.7288, 0.7754],
    "Accuracy@3": [0.9800, 0.9788, 0.9703, 0.9831, 0.9576],
    "MRR":        [_mrr,   0.9449, 0.9314, 0.8499, 0.8707],
    "NDCG@3":     [0.9700, 0.9511, 0.9372, 0.8815, 0.8876],
    "Macro-F1":   [0.9302, 0.9083, 0.8700, 0.6457, 0.6872],
}
perf_df = pd.DataFrame(perf_data)

st.dataframe(
    perf_df.style
        .format({c: "{:.4f}" for c in ["Accuracy@1","Accuracy@3","MRR","NDCG@3","Macro-F1"]})
        .background_gradient(subset=["MRR"], cmap="Blues")
        .set_properties(**{"font-family":"Inter,sans-serif","font-size":"13px"}),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    f"★ {_mname} selected as primary model (from smartviz_meta.json). "
    f"Pool D (held-out, 78 queries): Acc@1≈0.9615, MRR≈0.9776 — gap within ±0.05 of Pool C."
)
