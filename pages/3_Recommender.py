import streamlit as st
import pandas as pd

st.set_page_config(page_title="SmartViz-ML · Recommender", page_icon="💡", layout="wide")

from utils.recommender import recommend, VIZ_META
from utils.visualizations import generate_chart
from utils.insights import generate_insight
from utils.explainability import explain_recommendation

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#1e293b; }
.page-sub   { color:#64748b; font-size:0.95rem; margin-bottom:1.5rem; }
.rec-card {
    background:white; border-radius:16px; padding:1.5rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 6px 24px rgba(108,99,255,0.08);
    border:1px solid #f1f0ff; margin-bottom:1rem; cursor:pointer;
    transition:transform 0.15s ease, box-shadow 0.15s ease;
}
.rec-card:hover { transform:translateY(-2px); box-shadow:0 6px 32px rgba(108,99,255,0.16); }
.rec-card.selected { border:2px solid #6C63FF; background:#fafafe; }
.rec-rank  { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; color:#c7d2fe; }
.rec-icon  { font-size:2.2rem; margin:0.25rem 0; }
.rec-label { font-weight:700; color:#1e293b; font-size:1rem; }
.rec-use   { color:#94a3b8; font-size:0.8rem; margin-top:0.15rem; }
.conf-bar-bg { background:#f1f0ff; border-radius:100px; height:8px; margin-top:0.75rem; }
.conf-bar-fg { background:linear-gradient(90deg,#6C63FF,#48CAE4); border-radius:100px; height:8px; }
.conf-pct  { font-family:'Syne',sans-serif; font-weight:800; font-size:1.1rem; color:#6C63FF; margin-top:0.3rem; }
.chart-section { background:white; border-radius:16px; padding:1.5rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(108,99,255,0.07);
    border:1px solid #f1f0ff; }
.insight-box { background:linear-gradient(135deg,#f8f7ff 0%,#f0f9ff 100%);
    border-radius:14px; padding:1.25rem 1.5rem; border-left:4px solid #6C63FF;
    margin-top:1rem; }
.query-example { background:#f8f7ff; border:1px solid #e2e0ff; border-radius:10px;
    padding:0.5rem 1rem; font-size:0.83rem; color:#5b52d6; cursor:pointer;
    display:inline-block; margin:0.2rem; font-weight:500;
    transition:background 0.15s; }
.query-example:hover { background:#ede9ff; }
.intent-tag { display:inline-flex; align-items:center; gap:0.3rem;
    background:#f0fdf4; color:#166534; border-radius:6px;
    padding:0.2rem 0.6rem; font-size:0.75rem; font-weight:600; margin:0.15rem; }
.no-data-msg { background:#fff7ed; border:1px solid #fed7aa; border-radius:12px;
    padding:1.5rem; text-align:center; color:#c2410c; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">💡 Visualization Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Ask a business question in plain English — SmartViz ranks visualization types by model confidence.</div>', unsafe_allow_html=True)

# ── Check data ────────────────────────────────────────────────────────────────
df      = st.session_state.get("df")
profile = st.session_state.get("profile")

if df is None or profile is None:
    st.markdown("""
    <div class="no-data-msg">
        <div style="font-size:2rem">📂</div>
        <div style="font-weight:600;margin-top:0.5rem">No dataset loaded yet</div>
        <div style="font-size:0.85rem;margin-top:0.25rem">Go to <strong>Data Upload</strong> to load your data first.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLES = [
    "Which products generate the most revenue?",
    "How did our sales evolve over the last year?",
    "Which customer segments are most valuable?",
    "Is there a relationship between price and quantity sold?",
    "What is the distribution of order values?",
    "Which countries generate the most revenue?",
    "Show me pairwise correlations across all metrics",
    "Flag transactions that look unusual",
]

st.markdown("**Try an example:**")
example_html = "".join(
    f'<span class="query-example" onclick="void(0)">▶ {q}</span>'
    for q in EXAMPLES
)
st.markdown(example_html, unsafe_allow_html=True)

selected_example = st.selectbox(
    "Or pick from the list",
    [""] + EXAMPLES,
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Query input ───────────────────────────────────────────────────────────────
default_query = selected_example if selected_example else ""
query = st.text_input(
    "Your business question",
    value=default_query,
    placeholder="e.g. Which products generate the most revenue?",
    help="Ask anything about your data in plain English.",
)

run = st.button("✨ Recommend visualizations", type="primary", use_container_width=False)

if not run and not st.session_state.get("last_query"):
    st.stop()

if run and query.strip():
    st.session_state["last_query"] = query

active_query = st.session_state.get("last_query", query)

if not active_query:
    st.warning("Please enter a business question.")
    st.stop()

# ── Inference ─────────────────────────────────────────────────────────────────
with st.spinner("Running SmartViz-ML…"):
    recs = recommend(profile, active_query, top_n=3)
    explanation = explain_recommendation(profile, active_query, recs[0]["viz_type"])

# Save for Explainability page
st.session_state["recs"]        = recs
st.session_state["explanation"] = explanation
st.session_state["active_query"]= active_query

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Recommended visualizations")
st.markdown(f'*Question: "{active_query}"*')

# ── Recommendation cards ──────────────────────────────────────────────────────
medal = ["🥇", "🥈", "🥉"]
cols  = st.columns(3)

for i, (rec, col) in enumerate(zip(recs, cols)):
    with col:
        conf_w = int(rec["confidence"])
        st.markdown(
            f'<div class="rec-card{"  selected" if i==0 else ""}">'
            f'<div class="rec-rank">{medal[i]} #{rec["rank"]}</div>'
            f'<div class="rec-icon">{rec["icon"]}</div>'
            f'<div class="rec-label">{rec["label"]}</div>'
            f'<div class="rec-use">{rec["use"]}</div>'
            f'<div class="conf-bar-bg"><div class="conf-bar-fg" style="width:{min(conf_w,100)}%"></div></div>'
            f'<div class="conf-pct">{rec["confidence"]}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Detected intents ──────────────────────────────────────────────────────────
active_intents = [k for k, v in explanation["intent_flags"].items() if v]
if active_intents:
    intent_label_map = {
        "kw_temporal":"⏱ Temporal","kw_rank":"🏆 Ranking","kw_compare":"⚖️ Comparison",
        "kw_proportion":"🥧 Proportion","kw_distribution":"📊 Distribution",
        "kw_correlation":"🔗 Correlation","kw_geographic":"🗺️ Geographic",
        "kw_segment":"👥 Segment","kw_anomaly":"🚨 Anomaly","kw_prediction":"🔮 Prediction",
    }
    tags_html = "".join(
        f'<span class="intent-tag">✓ {intent_label_map.get(k, k)}</span>'
        for k in active_intents
    )
    st.markdown(f"**Detected intents:** {tags_html}", unsafe_allow_html=True)

st.divider()

# ── Visualization selector + chart ───────────────────────────────────────────
st.markdown("### Generated visualization")

viz_choice = st.radio(
    "Select visualization to display",
    [f'{r["icon"]} {r["label"]} ({r["confidence"]}%)' for r in recs],
    horizontal=True,
    label_visibility="collapsed",
)

chosen_idx  = [f'{r["icon"]} {r["label"]} ({r["confidence"]}%)' for r in recs].index(viz_choice)
chosen_rec  = recs[chosen_idx]

with st.spinner("Generating chart…"):
    fig = generate_chart(df, chosen_rec["viz_type"], active_query)

st.markdown(
    f'<div class="chart-section">'
    f'<div style="font-weight:700;color:#1e293b;font-size:1.05rem;margin-bottom:0.5rem">'
    f'{chosen_rec["icon"]} {chosen_rec["label"]}'
    f'<span style="color:#94a3b8;font-size:0.8rem;font-weight:400;margin-left:0.5rem">confidence {chosen_rec["confidence"]}%</span>'
    f'</div>',
    unsafe_allow_html=True,
)

if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chart could not be generated for this dataset. Try a different visualization type.")

st.markdown("</div>", unsafe_allow_html=True)

# ── Business insight ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.spinner("Generating business insight…"):
    insight = generate_insight(df, chosen_rec["viz_type"], active_query)

st.markdown(
    f'<div class="insight-box">'
    f'<div style="font-weight:700;color:#1e293b;font-size:0.9rem;margin-bottom:0.5rem">📋 Business Insight</div>'
    f'<div style="color:#334155;font-size:0.9rem;line-height:1.6">{insight}</div>'
    f'<div style="color:#94a3b8;font-size:0.75rem;margin-top:0.75rem">⚠ This insight is descriptive, not causal. Verify findings before operational decisions.</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.info("🔎 Go to **Explainability** to see *why* SmartViz made this recommendation.")
