import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

st.set_page_config(page_title="SmartViz-ML · Human Evaluation", page_icon="📝", layout="wide")

from utils.recommender import recommend, VIZ_META
from utils.visualizations import generate_chart

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title  { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#1e293b; }
.page-sub    { color:#64748b; font-size:0.95rem; margin-bottom:1.5rem; }
.step-header { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:#1e293b; margin-bottom:0.5rem; }
.question-card {
    background:linear-gradient(135deg,#f8f7ff,#f0f9ff);
    border-radius:16px; padding:1.5rem; border-left:4px solid #6C63FF;
    margin-bottom:1rem;
}
.question-text { font-size:1.05rem; font-weight:600; color:#1e293b; margin-bottom:0.25rem; }
.question-meta { font-size:0.8rem; color:#94a3b8; }
.rec-display {
    background:white; border-radius:14px; padding:1.25rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(108,99,255,0.07);
    border:1px solid #f1f0ff; text-align:center; margin-bottom:1rem;
}
.likert-label { font-weight:600; color:#1e293b; font-size:0.9rem; margin-bottom:0.35rem; }
.likert-hint  { font-size:0.75rem; color:#94a3b8; margin-bottom:0.5rem; }
.progress-bar-bg { background:#f1f0ff; border-radius:100px; height:10px; margin-bottom:1rem; }
.progress-bar-fg { background:linear-gradient(90deg,#6C63FF,#48CAE4); border-radius:100px; height:10px; transition:width 0.3s ease; }
.consent-box { background:#f8f7ff; border:1px solid #e2e0ff; border-radius:14px; padding:1.5rem; margin-bottom:1rem; }
.badge-ok    { background:#d1fae5; color:#065f46; border-radius:100px; padding:0.25rem 0.75rem; font-size:0.8rem; font-weight:600; }
.submit-success { background:linear-gradient(135deg,#d1fae5,#a7f3d0); border-radius:16px;
    padding:2rem; text-align:center; border:1px solid #6ee7b7; }
</style>
""", unsafe_allow_html=True)

# ── Evaluation queries (15 from Pool C human eval set) ───────────────────────
EVAL_QUERIES = [
    {"id":"HQ01","domain":"Retail",        "question":"Show me how our total sales have changed quarter by quarter",     "ground_truth":"line_chart",   "profile":"temporal_num"},
    {"id":"HQ02","domain":"Retail",        "question":"Which five countries generated the most orders last year?",        "ground_truth":"bar_chart",    "profile":"cat_rank"},
    {"id":"HQ03","domain":"Retail",        "question":"What does the distribution of individual order values look like?", "ground_truth":"histogram",    "profile":"num_dist"},
    {"id":"HQ04","domain":"Retail",        "question":"Do customers who buy more frequently also spend more per visit?",  "ground_truth":"scatter_plot", "profile":"num_num"},
    {"id":"HQ05","domain":"Retail",        "question":"What fraction of revenue comes from repeat versus new buyers?",    "ground_truth":"pie_chart",    "profile":"cat_prop"},
    {"id":"HQ06","domain":"Retail",        "question":"How are VIP, loyal, and at-risk customers distributed?",          "ground_truth":"bar_chart",    "profile":"cluster_cat"},
    {"id":"HQ07","domain":"Retail",        "question":"Show total revenue by country on a world map",                    "ground_truth":"choropleth",   "profile":"geo_num"},
    {"id":"HQ08","domain":"Retail",        "question":"Summarise the variability in delivery lead times",                "ground_truth":"box_plot",     "profile":"num_dist"},
    {"id":"HQ09","domain":"Retail",        "question":"Give me an overview of how all our KPIs relate to each other",   "ground_truth":"heatmap",      "profile":"num_num_multi"},
    {"id":"HQ10","domain":"Retail",        "question":"Is there a seasonal pattern in weekly order volumes?",            "ground_truth":"line_chart",   "profile":"temporal_num"},
    {"id":"HQ11","domain":"Retail",        "question":"What are the ten highest-grossing product categories?",           "ground_truth":"bar_chart",    "profile":"cat_num"},
    {"id":"HQ12","domain":"Retail",        "question":"Show the full range of unit prices across our catalogue",         "ground_truth":"histogram",    "profile":"num_dist"},
    {"id":"HQ13","domain":"Retail",        "question":"Break down our cost structure by expense category",               "ground_truth":"pie_chart",    "profile":"cat_prop"},
    {"id":"HQ14","domain":"Retail",        "question":"Visualise the RFM space to identify natural customer clusters",  "ground_truth":"scatter_plot", "profile":"cluster_num"},
    {"id":"HQ15","domain":"Retail",        "question":"Display pairwise correlations for all RFM dimensions",           "ground_truth":"heatmap",      "profile":"num_num_multi"},
]

META_PROFILES = {
    "temporal_num": dict(n_numeric=2, n_categorical=0, n_datetime=1, max_cardinality=0, mean_correlation=0.4, missing_ratio=0.01, has_temporal=1, has_geo=0),
    "cat_rank":     dict(n_numeric=1, n_categorical=1, n_datetime=0, max_cardinality=15, mean_correlation=0.05, missing_ratio=0.0, has_temporal=0, has_geo=0),
    "num_dist":     dict(n_numeric=1, n_categorical=0, n_datetime=0, max_cardinality=0, mean_correlation=0.0, missing_ratio=0.03, has_temporal=0, has_geo=0),
    "num_num":      dict(n_numeric=2, n_categorical=0, n_datetime=0, max_cardinality=0, mean_correlation=0.65, missing_ratio=0.01, has_temporal=0, has_geo=0),
    "cat_prop":     dict(n_numeric=1, n_categorical=1, n_datetime=0, max_cardinality=5, mean_correlation=0.05, missing_ratio=0.0, has_temporal=0, has_geo=0),
    "cluster_cat":  dict(n_numeric=1, n_categorical=2, n_datetime=0, max_cardinality=6, mean_correlation=0.2, missing_ratio=0.01, has_temporal=0, has_geo=0),
    "geo_num":      dict(n_numeric=1, n_categorical=1, n_datetime=0, max_cardinality=40, mean_correlation=0.1, missing_ratio=0.05, has_temporal=0, has_geo=1),
    "cat_num":      dict(n_numeric=1, n_categorical=1, n_datetime=0, max_cardinality=12, mean_correlation=0.1, missing_ratio=0.02, has_temporal=0, has_geo=0),
    "num_num_multi":dict(n_numeric=5, n_categorical=0, n_datetime=0, max_cardinality=0, mean_correlation=0.55, missing_ratio=0.02, has_temporal=0, has_geo=0),
    "cluster_num":  dict(n_numeric=3, n_categorical=1, n_datetime=0, max_cardinality=5, mean_correlation=0.3, missing_ratio=0.01, has_temporal=0, has_geo=0),
}

LIKERT_DIMS   = ["Relevance", "Readability", "Usefulness", "Interpretability"]
LIKERT_HINTS  = {
    "Relevance":       "1 = completely wrong chart  →  5 = perfectly appropriate",
    "Readability":     "1 = very hard to read  →  5 = immediately clear",
    "Usefulness":      "1 = no business value  →  5 = highly actionable",
    "Interpretability":"1 = impossible to interpret  →  5 = trivial to interpret",
}
OVERALL_LABEL = "Overall satisfaction"
RESP_FILE     = "human_eval_responses.csv"

st.markdown('<div class="page-title">📝 Human Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Rate SmartViz-ML recommendations to provide real data for the research paper. Your responses are saved to <code>human_eval_responses.csv</code>.</div>', unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "eval_step" not in st.session_state:
    st.session_state["eval_step"] = "consent"
if "eval_responses" not in st.session_state:
    st.session_state["eval_responses"] = []
if "eval_q_idx" not in st.session_state:
    st.session_state["eval_q_idx"] = 0
if "participant_info" not in st.session_state:
    st.session_state["participant_info"] = {}

step = st.session_state["eval_step"]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Consent & participant info
# ─────────────────────────────────────────────────────────────────────────────
if step == "consent":
    st.markdown("#### Step 1 — Consent & Participant Information")
    st.markdown("""
    <div class="consent-box">
    <strong>Study purpose</strong><br>
    You will evaluate 15 visualization recommendations made by SmartViz-ML for business questions.
    For each question, you will see the recommended chart type (with the actual chart) and rate it
    on four Likert scales (1–5).<br><br>
    <strong>Data use</strong><br>
    Responses are collected for academic research only. No personally identifying information is stored
    beyond your participant ID and professional background.<br><br>
    <strong>Duration</strong><br>
    Approximately 10–15 minutes.
    </div>
    """, unsafe_allow_html=True)

    with st.form("consent_form"):
        pid = st.text_input("Participant ID", placeholder="e.g. P001",
                            help="Choose any ID that is not your real name.")
        background = st.selectbox(
            "Professional background",
            ["Business / Management", "Banking / Finance", "Data Science / Analytics",
             "Healthcare", "Manufacturing / Engineering", "Human Resources", "Other"],
        )
        experience = st.selectbox(
            "Years of experience with data / analytics",
            ["< 1 year", "1–3 years", "3–5 years", "5–10 years", "10+ years"],
        )
        consent = st.checkbox("I have read the above and agree to participate.")
        submitted = st.form_submit_button("Start evaluation →", type="primary")

        if submitted:
            if not pid.strip():
                st.error("Please enter a participant ID.")
            elif not consent:
                st.error("Please tick the consent checkbox to proceed.")
            else:
                st.session_state["participant_info"] = {
                    "participant_id": pid.strip(),
                    "background":     background,
                    "experience":     experience,
                }
                st.session_state["eval_step"]  = "questions"
                st.session_state["eval_q_idx"] = 0
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Evaluation questions
# ─────────────────────────────────────────────────────────────────────────────
elif step == "questions":
    q_idx   = st.session_state["eval_q_idx"]
    n_total = len(EVAL_QUERIES)

    # Progress
    pct = int((q_idx / n_total) * 100)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">'
        f'<div style="flex:1">'
        f'<div class="progress-bar-bg"><div class="progress-bar-fg" style="width:{pct}%"></div></div>'
        f'</div>'
        f'<div style="font-size:0.85rem;font-weight:600;color:#6C63FF;white-space:nowrap">{q_idx}/{n_total}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    q = EVAL_QUERIES[q_idx]
    profile = META_PROFILES.get(q["profile"], META_PROFILES["cat_num"])

    # Get SmartViz recommendation
    recs = recommend(profile, q["question"], top_n=3)
    top_rec = recs[0]

    # Question card
    st.markdown(
        f'<div class="question-card">'
        f'<div class="question-text">"{q["question"]}"</div>'
        f'<div class="question-meta">Query {q["id"]} · Domain: {q["domain"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Recommendation display
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown(
            f'<div class="rec-display">'
            f'<div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:#94a3b8;margin-bottom:0.5rem">SmartViz-ML recommends</div>'
            f'<div style="font-size:2.5rem">{top_rec["icon"]}</div>'
            f'<div style="font-weight:800;font-size:1.1rem;color:#1e293b;margin:0.25rem 0">{top_rec["label"]}</div>'
            f'<div style="color:#6C63FF;font-weight:700;font-size:1rem">{top_rec["confidence"]}% confidence</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Runner-up
        if len(recs) > 1:
            st.markdown(
                f'<div style="font-size:0.78rem;color:#94a3b8;text-align:center">'
                f'Also considered: {recs[1]["icon"]} {recs[1]["label"]} ({recs[1]["confidence"]}%)</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        # Generate chart from demo data if available
        df = st.session_state.get("df")
        if df is not None:
            fig = generate_chart(df, top_rec["viz_type"], q["question"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Chart preview not available for this visualization type.")
        else:
            st.info("Load a dataset on the **Data Upload** page to see actual charts here.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Likert form
    with st.form(f"eval_form_{q_idx}"):
        st.markdown("**Please rate this recommendation:**")

        ratings = {}
        cols = st.columns(len(LIKERT_DIMS))
        for col, dim in zip(cols, LIKERT_DIMS):
            with col:
                st.markdown(f'<div class="likert-label">{dim}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="likert-hint">{LIKERT_HINTS[dim]}</div>', unsafe_allow_html=True)
                ratings[dim] = st.radio(
                    dim, [1, 2, 3, 4, 5],
                    index=3,  # default 4
                    horizontal=False,
                    label_visibility="collapsed",
                    key=f"{dim}_{q_idx}",
                )

        overall = st.slider("Overall satisfaction (1–5)", 1, 5, 4, key=f"overall_{q_idx}")
        comments = st.text_area("Comments (optional)", key=f"comments_{q_idx}",
                                 placeholder="Any observations about this recommendation…",
                                 height=80)

        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            submitted = st.form_submit_button(
                "Next →" if q_idx < n_total - 1 else "Submit evaluation",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            row = {
                "participant_id":    st.session_state["participant_info"]["participant_id"],
                "background":        st.session_state["participant_info"]["background"],
                "experience":        st.session_state["participant_info"]["experience"],
                "query_id":          q["id"],
                "domain":            q["domain"],
                "business_question": q["question"],
                "ground_truth_viz":  q["ground_truth"],
                "recommended_viz":   top_rec["viz_type"],
                "confidence":        top_rec["confidence"],
                "rank1_viz":         recs[0]["viz_type"],
                "rank2_viz":         recs[1]["viz_type"] if len(recs) > 1 else "",
                "rank3_viz":         recs[2]["viz_type"] if len(recs) > 2 else "",
                "Relevance":         ratings["Relevance"],
                "Readability":       ratings["Readability"],
                "Usefulness":        ratings["Usefulness"],
                "Interpretability":  ratings["Interpretability"],
                "overall_satisfaction": overall,
                "comments":          comments,
                "timestamp":         datetime.now().isoformat(),
            }
            st.session_state["eval_responses"].append(row)

            if q_idx < n_total - 1:
                st.session_state["eval_q_idx"] += 1
                st.rerun()
            else:
                st.session_state["eval_step"] = "submit"
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Submit & save
# ─────────────────────────────────────────────────────────────────────────────
elif step == "submit":
    responses = st.session_state.get("eval_responses", [])
    pid       = st.session_state["participant_info"].get("participant_id","unknown")

    # Save via storage module (Google Sheets or CSV fallback)
    from utils.storage import save_response, responses_as_csv
    backend = save_response(responses[-1]) if responses else 'csv'
    for r in responses[:-1]:
        save_response(r)

    # Summary stats
    dims = ["Relevance","Readability","Usefulness","Interpretability","overall_satisfaction"]
    means = {d: pd.DataFrame(responses)[d].mean() for d in dims}

    st.markdown(
        f'<div class="submit-success">'
        f'<div style="font-size:3rem">🎉</div>'
        f'<div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#065f46;margin:0.5rem 0">Thank you, {pid}!</div>'
        f'<div style="color:#047857;font-size:0.9rem">Your {len(responses)} responses have been saved to <code>{RESP_FILE}</code>.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Your scores")

    metric_cols = st.columns(5)
    for col, dim in zip(metric_cols, dims):
        with col:
            label = dim.replace("_"," ").title()
            st.metric(label, f"{means[dim]:.2f} / 5")

    # Download button
    st.markdown("<br>", unsafe_allow_html=True)
    from utils.storage import responses_as_csv
    st.download_button(
        "⬇️ Download all responses CSV",
        data=responses_as_csv(),
        file_name=f"human_eval_responses_{pid}.csv",
        mime="text/csv",
    )

    if st.button("🔄 Evaluate again (new participant)"):
        for key in ["eval_step","eval_responses","eval_q_idx","participant_info"]:
            st.session_state.pop(key, None)
        st.rerun()
