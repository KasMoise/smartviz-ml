import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="SmartViz-ML · Data Upload", page_icon="📂", layout="wide")

from utils.data_profiler import profile_dataset

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#1e293b; }
.page-sub   { color:#64748b; font-size:0.95rem; margin-bottom:1.5rem; }
.profile-card {
    background:white; border-radius:14px; padding:1.25rem 1.5rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(108,99,255,0.07);
    border:1px solid #f1f0ff; margin-bottom:0.75rem;
}
.metric-row { display:flex; align-items:center; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid #f8f8ff; }
.metric-row:last-child { border-bottom:none; }
.metric-name { color:#64748b; font-size:0.85rem; font-weight:500; }
.metric-val  { font-weight:700; color:#1e293b; font-size:0.9rem; }
.badge-yes   { background:#d1fae5; color:#065f46; border-radius:6px; padding:0.15rem 0.5rem; font-size:0.75rem; font-weight:600; }
.badge-no    { background:#f1f5f9; color:#94a3b8; border-radius:6px; padding:0.15rem 0.5rem; font-size:0.75rem; font-weight:600; }
.badge-warn  { background:#fef3c7; color:#92400e; border-radius:6px; padding:0.15rem 0.5rem; font-size:0.75rem; font-weight:600; }
.col-tag     { display:inline-block; background:#f0f9ff; color:#0369a1; border-radius:6px; padding:0.15rem 0.5rem; font-size:0.75rem; margin:0.1rem; font-weight:500; }
.col-tag-cat { background:#fdf4ff; color:#7e22ce; }
.col-tag-dt  { background:#f0fdf4; color:#166534; }
.upload-area { border:2px dashed #c7d2fe; border-radius:16px; padding:2.5rem; text-align:center; background:#fafafe; }
.upload-icon { font-size:3rem; margin-bottom:0.5rem; }
.upload-hint { color:#94a3b8; font-size:0.85rem; margin-top:0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📂 Data Upload & Profiling</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Upload your dataset — SmartViz automatically extracts the features needed for visualization recommendation.</div>', unsafe_allow_html=True)

# ── Source selector ───────────────────────────────────────────────────────────
source = st.radio(
    "Choose your data source",
    ["📤 Upload my dataset", "🛒 Try Online Retail II demo"],
    horizontal=True,
    label_visibility="collapsed",
)

df = None

if source == "📤 Upload my dataset":
    st.markdown("""
    <div class="upload-area">
        <div class="upload-icon">📁</div>
        <div style="font-weight:600;color:#475569">Drop your file here</div>
        <div class="upload-hint">CSV or Excel (.xlsx) · up to 200 MB</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv", "xlsx", "xls"],
                                 label_visibility="collapsed")
    if uploaded:
        with st.spinner("Reading file…"):
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded, engine="openpyxl")
                else:
                    df = pd.read_csv(uploaded, low_memory=False)
                st.success(f"✓ Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")

else:
    # Synthetic demo — always available on cloud and locally
    import os
    np.random.seed(42)
    n = 3000
    dates = pd.date_range("2022-01-01", periods=n, freq="6h")
    products  = ["REGENCY CAKESTAND 3 TIER","WHITE HANGING HEART T-LIGHT","JUMBO BAG RED WHITE SPOTTY",
                  "PARTY BUNTING","ASSORTED COLOUR BIRD ORNAMENT","PAPER CRAFT LITTLE BIRDIE",
                  "MEDIUM CERAMIC TOP STORAGE JAR","PAPER CHAIN KIT 50s CHRISTMAS"]
    countries = ["United Kingdom","Germany","France","Netherlands","Spain","Australia","Belgium","Japan"]
    df = pd.DataFrame({
        "Invoice":     [f"INV{i:05d}" for i in range(n)],
        "Description": np.random.choice(products, n),
        "Country":     np.random.choice(countries, n, p=[0.55,0.10,0.08,0.07,0.06,0.05,0.05,0.04]),
        "Quantity":    np.random.randint(1, 80, n),
        "Price":       np.random.choice([1.25,2.10,2.95,4.15,6.75,6.95,12.75], n),
        "InvoiceDate": dates,
        "Customer ID": np.random.randint(12346, 18288, n),
    })
    df["Revenue"] = (df["Quantity"] * df["Price"]).round(2)
    st.success(f"✓ Demo dataset loaded — {len(df):,} transactions · 8 products · 8 countries")
    st.info("📊 This is a realistic synthetic dataset. Upload your own CSV/Excel for real analysis.")

# ── Profile ───────────────────────────────────────────────────────────────────
if df is not None:
    st.session_state["df"] = df

    with st.spinner("Profiling dataset…"):
        profile = profile_dataset(df)
    st.session_state["profile"] = profile

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Dataset Profile")

    # ── Summary metrics ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        (c1, "🗂️", f"{profile['n_rows']:,}", "Rows"),
        (c2, "📐", str(profile["n_cols"]), "Columns"),
        (c3, "⚠️", f"{profile['missing_pct']}%", "Missing values"),
        (c4, "🔗", f"{profile['mean_correlation']:.2f}", "Mean correlation"),
    ]
    for col, icon, val, label in tiles:
        with col:
            st.markdown(
                f'<div class="profile-card" style="text-align:center">'
                f'<div style="font-size:1.8rem">{icon}</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:#6C63FF">{val}</div>'
                f'<div style="color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;font-weight:500">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Column breakdown")
        def yn(v): return f'<span class="badge-yes">Yes</span>' if v else f'<span class="badge-no">No</span>'

        rows_html = "".join([
            f'<div class="metric-row"><span class="metric-name">Numeric columns</span><span class="metric-val">{profile["n_numeric"]}</span></div>',
            f'<div class="metric-row"><span class="metric-name">Categorical columns</span><span class="metric-val">{profile["n_categorical"]}</span></div>',
            f'<div class="metric-row"><span class="metric-name">Datetime columns</span><span class="metric-val">{profile["n_datetime"]}</span></div>',
            f'<div class="metric-row"><span class="metric-name">Max cardinality</span><span class="metric-val">{profile["max_cardinality"]:,}</span></div>',
            f'<div class="metric-row"><span class="metric-name">Temporal data</span><span class="metric-val">{yn(profile["has_temporal"])}</span></div>',
            f'<div class="metric-row"><span class="metric-name">Geographic variable</span><span class="metric-val">{yn(profile["has_geo"])}</span></div>',
        ])
        st.markdown(f'<div class="profile-card">{rows_html}</div>', unsafe_allow_html=True)

        # Missing values
        if profile["missing_by_col"]:
            st.markdown("#### Missing values by column")
            miss_df = pd.Series(profile["missing_by_col"]).sort_values(ascending=False).reset_index()
            miss_df.columns = ["Column", "Missing"]
            miss_df["Pct"] = (miss_df["Missing"] / profile["n_rows"] * 100).round(1)
            fig_miss = px.bar(
                miss_df.head(10), x="Column", y="Pct",
                color="Pct", color_continuous_scale="Reds",
                labels={"Pct": "Missing %"},
            )
            fig_miss.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, margin=dict(l=0,r=0,t=20,b=0),
                font=dict(family="Inter,sans-serif",size=11),
            )
            st.plotly_chart(fig_miss, use_container_width=True)

    with right:
        st.markdown("#### Column names")
        num_tags = "".join(f'<span class="col-tag">{c}</span>' for c in profile["num_cols"])
        cat_tags = "".join(f'<span class="col-tag col-tag-cat">{c}</span>' for c in profile["cat_cols"])
        dt_tags  = "".join(f'<span class="col-tag col-tag-dt">{c}</span>'  for c in profile["date_cols"])

        st.markdown(
            f'<div class="profile-card">'
            f'<div style="margin-bottom:0.5rem"><span style="font-size:0.75rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em">Numeric</span><br>{num_tags or "<em style=\'color:#cbd5e1\'>None</em>"}</div>'
            f'<div style="margin-bottom:0.5rem"><span style="font-size:0.75rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em">Categorical</span><br>{cat_tags or "<em style=\'color:#cbd5e1\'>None</em>"}</div>'
            f'<div><span style="font-size:0.75rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em">Datetime</span><br>{dt_tags or "<em style=\'color:#cbd5e1\'>None</em>"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Cardinality chart for categoricals
        if profile["cardinalities"]:
            st.markdown("#### Category cardinalities")
            card_df = pd.Series(profile["cardinalities"]).sort_values(ascending=False).reset_index()
            card_df.columns = ["Column","Cardinality"]
            fig_card = px.bar(
                card_df.head(10), x="Column", y="Cardinality",
                color="Cardinality", color_continuous_scale="Blues",
            )
            fig_card.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, margin=dict(l=0,r=0,t=20,b=0),
                font=dict(family="Inter,sans-serif",size=11),
                xaxis=dict(gridcolor="#f0f0f8"), yaxis=dict(gridcolor="#f0f0f8"),
            )
            st.plotly_chart(fig_card, use_container_width=True)

    # ── Data preview ──────────────────────────────────────────────────────
    with st.expander("📋 Preview first 100 rows"):
        st.dataframe(df.head(100), use_container_width=True)

    st.success("✅ Dataset profiled and ready. Go to **💡 Visualization Recommender** to ask your question.")
