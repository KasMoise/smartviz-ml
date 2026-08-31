import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SmartViz-ML · Business Insights", page_icon="📋", layout="wide")

from utils.recommender import load_models
from utils.visualizations import generate_chart
from utils.insights import generate_insight

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#1e293b; }
.page-sub   { color:#64748b; font-size:0.95rem; margin-bottom:1.5rem; }
.insight-card {
    background:white; border-radius:16px; padding:1.5rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(108,99,255,0.07);
    border:1px solid #f1f0ff; margin-bottom:1rem;
}
.insight-header { font-weight:700; color:#1e293b; font-size:1.05rem; margin-bottom:0.75rem; }
.insight-text   { color:#334155; font-size:0.9rem; line-height:1.7; }
.insight-caution {
    background:#fff7ed; border:1px solid #fed7aa; border-radius:8px;
    padding:0.6rem 1rem; font-size:0.78rem; color:#92400e; margin-top:1rem;
}
.kpi-card {
    background:linear-gradient(135deg,#f8f7ff,#f0f9ff);
    border-radius:14px; padding:1.25rem;
    border:1px solid #e2e0ff; text-align:center;
}
.kpi-val   { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#6C63FF; }
.kpi-label { font-size:0.78rem; color:#94a3b8; font-weight:500; text-transform:uppercase; letter-spacing:0.06em; margin-top:0.25rem; }
.rec-chip  {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:#d1fae5; color:#065f46; border-radius:100px;
    padding:0.3rem 0.75rem; font-size:0.8rem; font-weight:600; margin:0.2rem;
}
.warn-chip {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:#fef3c7; color:#92400e; border-radius:100px;
    padding:0.3rem 0.75rem; font-size:0.8rem; font-weight:600; margin:0.2rem;
}
.no-data-msg { background:#fff7ed; border:1px solid #fed7aa; border-radius:12px;
    padding:1.5rem; text-align:center; color:#c2410c; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📋 Business Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Plain-language summaries calibrated to your data. All insights are descriptive — verify before operational use.</div>', unsafe_allow_html=True)

# ── Check state ───────────────────────────────────────────────────────────────
df          = st.session_state.get("df")
profile     = st.session_state.get("profile")
recs        = st.session_state.get("recs")
active_query= st.session_state.get("active_query", "")

if df is None or recs is None:
    st.markdown("""
    <div class="no-data-msg">
        <div style="font-size:2rem">💡</div>
        <div style="font-weight:600;margin-top:0.5rem">No recommendation yet</div>
        <div style="font-size:0.85rem;margin-top:0.25rem">Complete <strong>Data Upload</strong> → <strong>Recommender</strong> first.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

top_rec = recs[0]

# ── KPI summary ───────────────────────────────────────────────────────────────
st.markdown("### Dataset summary")
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

kpi_cols = st.columns(4)
kpi_data = [(f"{len(df):,}", "Total rows")]

if num_cols:
    primary_num = next(
        (c for c in num_cols if any(k in c.lower() for k in ["revenue","amount","sales","value","total"])),
        num_cols[0],
    )
    total = df[primary_num].sum()
    avg   = df[primary_num].mean()
    kpi_data += [
        (f"{total:,.0f}", f"Total {primary_num}"),
        (f"{avg:,.1f}",   f"Avg {primary_num}"),
    ]

if cat_cols:
    primary_cat = next(
        (c for c in cat_cols if any(k in c.lower() for k in ["product","category","country","segment","type"])),
        cat_cols[0],
    )
    kpi_data.append((str(df[primary_cat].nunique()), f"Unique {primary_cat}"))

for col, (val, label) in zip(kpi_cols, kpi_data[:4]):
    with col:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-val">{val}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Insight for top recommendation ───────────────────────────────────────────
st.markdown("### Insight for your question")
st.markdown(f'*"{active_query}"*')

insight = generate_insight(df, top_rec["viz_type"], active_query)
st.markdown(
    f'<div class="insight-card">'
    f'<div class="insight-header">{top_rec["icon"]} {top_rec["label"]} — {top_rec["confidence"]}% confidence</div>'
    f'<div class="insight-text">{insight}</div>'
    f'<div class="insight-caution">⚠ This insight is descriptive. Correlation does not imply causation. Validate findings before taking operational action.</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Top-N breakdown chart ────────────────────────────────────────────────────
st.markdown("### Explore your data")

if cat_cols and num_cols:
    col_sel, metric_sel = st.columns([1, 1])
    with col_sel:
        dim = st.selectbox("Group by", cat_cols, key="insight_dim")
    with metric_sel:
        metric = st.selectbox("Measure", num_cols, key="insight_metric")

    top_n = st.slider("Show top N", 5, 20, 10, key="insight_topn")
    agg = df.groupby(dim)[metric].sum().nlargest(top_n).reset_index()

    fig = px.bar(
        agg, x=metric, y=dim, orientation="h",
        color=metric, color_continuous_scale=["#c7d2fe","#6C63FF","#312e81"],
        labels={metric: metric, dim: dim},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False, margin=dict(l=0,r=0,t=20,b=0),
        yaxis=dict(autorange="reversed", gridcolor="#f0f0f8"),
        xaxis=dict(gridcolor="#f0f0f8"),
        font=dict(family="Inter,sans-serif", size=11),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Auto-generated finding
    top_item  = agg.iloc[0]
    top_share = top_item[metric] / agg[metric].sum() * 100
    n_top5    = min(5, len(agg))
    top5_share= agg.head(n_top5)[metric].sum() / agg[metric].sum() * 100

    chip_class = "rec-chip" if top_share < 50 else "warn-chip"
    chip_icon  = "✓" if top_share < 50 else "⚠"

    st.markdown(
        f'<div class="insight-card">'
        f'<div class="insight-header">🔍 Automated finding</div>'
        f'<div class="insight-text">'
        f'<strong>{top_item[dim]}</strong> leads {dim} by {metric} with '
        f'<strong>{top_item[metric]:,.0f}</strong> '
        f'({top_share:.1f}% of the displayed total). '
        f'The top {n_top5} groups account for <strong>{top5_share:.1f}%</strong> of total {metric}.'
        f'</div>'
        f'<div style="margin-top:0.75rem">'
        f'<span class="{chip_class}">{chip_icon} {"Balanced distribution" if top_share < 50 else "High concentration"}</span>'
        f'</div>'
        f'<div class="insight-caution">This is an automated summary. Review the full data before drawing conclusions.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── All-insights panel ────────────────────────────────────────────────────────
with st.expander("📊 Insights for all recommended visualizations"):
    for rec in recs:
        ins = generate_insight(df, rec["viz_type"], active_query)
        st.markdown(
            f'<div class="insight-card">'
            f'<div class="insight-header">{rec["icon"]} {rec["label"]} ({rec["confidence"]}%)</div>'
            f'<div class="insight-text">{ins}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
