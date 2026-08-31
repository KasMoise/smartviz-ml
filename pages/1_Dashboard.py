import streamlit as st

st.set_page_config(page_title="SmartViz-ML · Dashboard", page_icon="🧠", layout="wide")

from utils.recommender import load_models, VIZ_META

# ── Load models silently ───────────────────────────────────────────────────────
models = load_models()
model_loaded = "rf" in models

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: clamp(0.95rem, 2vw, 1.15rem);
    color: #64748b;
    font-weight: 400;
    margin-bottom: 2rem;
    max-width: 620px;
}
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(108,99,255,0.07);
    border: 1px solid #f1f0ff;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 24px rgba(108,99,255,0.14);
}
.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #6C63FF;
    line-height: 1;
}
.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
    font-weight: 500;
}
.viz-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f8f7ff;
    border: 1px solid #e2e0ff;
    border-radius: 100px;
    padding: 0.35rem 0.85rem;
    font-size: 0.82rem;
    color: #5b52d6;
    font-weight: 500;
    margin: 0.2rem;
}
.step-card {
    background: white;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    border-left: 4px solid #6C63FF;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 0.75rem;
}
.step-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #e2e0ff;
    line-height: 1;
}
.step-title { font-weight: 600; color: #1e293b; font-size: 0.95rem; }
.step-desc  { color: #64748b; font-size: 0.85rem; margin-top: 0.15rem; }
.badge-ok   { background:#d1fae5; color:#065f46; border-radius:100px; padding:0.2rem 0.7rem; font-size:0.78rem; font-weight:600; }
.badge-warn { background:#fef3c7; color:#92400e; border-radius:100px; padding:0.2rem 0.7rem; font-size:0.78rem; font-weight:600; }
.domain-tag {
    background: linear-gradient(135deg,#f0f9ff,#e0f2fe);
    color: #0369a1;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
col_hero, col_badge = st.columns([3, 1])
with col_hero:
    st.markdown('<div class="hero-title">SmartViz-ML</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Explainable Machine Learning for Automated Business '
        'Visualization and Decision Support</div>',
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    if model_loaded:
        st.markdown('<span class="badge-ok">✓ Model loaded</span>', unsafe_allow_html=True)
        meta = models.get("meta", {})
        if meta:
            st.markdown(
                f'<span style="font-size:0.75rem;color:#94a3b8">RF · '
                f'Acc@1 {meta.get("pool_c_acc1",0):.1%} · '
                f'MRR {meta.get("pool_c_mrr",0):.1%}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<span class="badge-warn">⚠ Demo mode — export models first</span>',
                    unsafe_allow_html=True)

st.divider()

# ── Stats row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    (c1, "8",   "Chart types"),
    (c2, "5",   "Business domains"),
    (c3, "300", "Benchmark queries"),
    (c4, "94.5%","Pool C Accuracy@1"),
    (c5, "96.9%","Pool C MRR"),
]
for col, num, lbl in stats:
    with col:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-number">{num}</div>'
            f'<div class="stat-label">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Supported visualizations ─────────────────────────────────────────────────
st.markdown("#### Supported visualization types")
pills_html = "".join(
    f'<span class="viz-pill">{meta["icon"]} {meta["label"]}</span>'
    for meta in VIZ_META.values()
)
st.markdown(pills_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Domains ──────────────────────────────────────────────────────────────────
st.markdown("#### Business domains")
d1, d2, d3, d4, d5 = st.columns(5)
for col, domain, icon in zip(
    [d1, d2, d3, d4, d5],
    ["Retail & E-commerce", "Banking & Finance", "Healthcare",
     "Manufacturing", "Human Resources"],
    ["🛒", "🏦", "🏥", "🏭", "👥"],
):
    with col:
        st.markdown(f'<div class="domain-tag">{icon}<br>{domain}</div>',
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Quick-start steps ────────────────────────────────────────────────────────
st.markdown("#### How to use SmartViz-ML")
steps = [
    ("1", "📂 Upload your data",
     "CSV or Excel — SmartViz automatically profiles your dataset."),
    ("2", "💡 Ask a business question",
     "Type what you want to know in plain English."),
    ("3", "📊 Review the recommendations",
     "SmartViz ranks visualization types by model confidence."),
    ("4", "🔎 Explore the explanation",
     "See why each visualization was recommended (SHAP + intent)."),
    ("5", "📋 Read the business insight",
     "Get a plain-language summary calibrated to your data."),
]
for num, title, desc in steps:
    st.markdown(
        f'<div class="step-card">'
        f'<div class="step-num">{num}</div>'
        f'<div class="step-title">{title}</div>'
        f'<div class="step-desc">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Model info ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ About the model"):
    st.markdown("""
**Algorithm** — Random Forest classifier (300 trees, `random_state=42`)

**Features** — 18-dimensional vector:
- 8 dataset meta-features (n_numeric, n_categorical, n_datetime, max_cardinality,
  mean_correlation, missing_ratio, has_temporal, has_geo)
- 10 analytical intent keywords (temporal, rank, compare, proportion, distribution,
  correlation, geographic, segment, anomaly, prediction)

**Training** — Pool A (200 synthetically constructed queries across 8 chart types)

**Evaluation** — Pool C (300 independently written queries, 5 business domains)

**Key results (Pool C)**
| Metric | Random Forest | Rule-based baseline |
|--------|:---:|:---:|
| Accuracy@1 | 0.9500 | 0.7754 |
| Accuracy@3 | 0.9800 | 0.9576 |
| MRR | 0.9688 | 0.8707 |
| NDCG@3 | 0.9700 | 0.8876 |
| Macro-F1 | 0.9302 | 0.6872 |

**Explainability** — SHAP TreeExplainer on Random Forest (top features: max_cardinality,
missing_ratio, mean_correlation, n_datetime)

**Paper** — SmartViz-ML: An Explainable Machine Learning Framework for Automated
Business Visualization and Decision Support (in preparation)
""")
