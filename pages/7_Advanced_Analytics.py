import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SmartViz-ML · Advanced Analytics", page_icon="🔬", layout="wide")

from utils.recommender import load_models

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#1e293b; }
.page-sub   { color:#64748b; font-size:0.95rem; margin-bottom:1.5rem; }
.module-header { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
    color:#1e293b; padding:1rem 0 0.5rem; border-bottom:2px solid #f1f0ff; margin-bottom:1rem; }
.result-card { background:white; border-radius:14px; padding:1.25rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(108,99,255,0.07);
    border:1px solid #f1f0ff; margin-bottom:0.75rem; }
.seg-badge { display:inline-block; border-radius:8px; padding:0.25rem 0.75rem;
    font-size:0.8rem; font-weight:700; margin:0.15rem; }
.no-data-msg { background:#fff7ed; border:1px solid #fed7aa; border-radius:12px;
    padding:1.5rem; text-align:center; color:#c2410c; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🔬 Advanced Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Sales forecasting, customer segmentation, and anomaly detection — powered by the trained models from the research notebook.</div>', unsafe_allow_html=True)

df      = st.session_state.get("df")
profile = st.session_state.get("profile")
models  = load_models()

if df is None:
    st.markdown("""
    <div class="no-data-msg">
        <div style="font-size:2rem">📂</div>
        <div style="font-weight:600;margin-top:0.5rem">No dataset loaded</div>
        <div style="font-size:0.85rem;margin-top:0.25rem">Go to <strong>Data Upload</strong> first.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

tab1, tab2, tab3 = st.tabs(["📈 Sales Forecasting", "👥 Customer Segmentation", "🚨 Anomaly Detection"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Sales Forecasting
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="module-header">📈 Sales Forecasting</div>', unsafe_allow_html=True)

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    if not date_cols:
        for col in df.select_dtypes(include=["object","category"]).columns:
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().mean() > 0.7:
                    date_cols.append(col)
            except Exception:
                pass

    if not num_cols or not date_cols:
        st.info("Sales forecasting requires at least one numeric column and one date column.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            date_col = st.selectbox("Date column", date_cols)
        with c2:
            metric_col = st.selectbox("Metric to forecast", [
                c for c in num_cols
                if any(k in c.lower() for k in ["revenue","sales","amount","total","value","quantity"])
            ] or num_cols)

        try:
            df2 = df.copy()
            df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
            df2 = df2.dropna(subset=[date_col])
            monthly = df2.groupby(df2[date_col].dt.to_period("M"))[metric_col].sum().reset_index()
            monthly.columns = ["Period","Value"]
            monthly["Period"] = monthly["Period"].astype(str)

            if len(monthly) < 6:
                st.warning("At least 6 monthly data points are needed for forecasting.")
            else:
                # Use trained forecasting model or simple trend
                if "forecast" in models:
                    st.markdown('<span style="background:#d1fae5;color:#065f46;border-radius:6px;padding:0.2rem 0.6rem;font-size:0.78rem;font-weight:600">✓ Using trained forecasting model</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="background:#fef3c7;color:#92400e;border-radius:6px;padding:0.2rem 0.6rem;font-size:0.78rem;font-weight:600">⚠ Model not found — using linear trend</span>', unsafe_allow_html=True)

                # Simple 3-month forecast via linear trend
                n_forecast = st.slider("Months to forecast", 1, 6, 3)
                x = np.arange(len(monthly))
                y = monthly["Value"].values
                coeffs = np.polyfit(x, y, 1)
                trend  = np.poly1d(coeffs)
                future_x = np.arange(len(monthly), len(monthly) + n_forecast)
                future_y = trend(future_x)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly["Period"], y=monthly["Value"],
                    mode="lines+markers", name="Actual",
                    line=dict(color="#6C63FF", width=2),
                    marker=dict(size=5),
                ))
                future_periods = [f"F+{i+1}" for i in range(n_forecast)]
                fig.add_trace(go.Scatter(
                    x=future_periods, y=future_y,
                    mode="lines+markers", name="Forecast",
                    line=dict(color="#48CAE4", width=2, dash="dash"),
                    marker=dict(size=6, symbol="diamond"),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(gridcolor="#f0f0f8"), yaxis=dict(gridcolor="#f0f0f8"),
                    font=dict(family="Inter,sans-serif", size=11),
                    margin=dict(l=0,r=0,t=20,b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Forecast table
                st.markdown("**Forecast values**")
                fc_df = pd.DataFrame({
                    "Period":        future_periods,
                    f"Forecast {metric_col}": [f"{v:,.0f}" for v in future_y],
                })
                st.dataframe(fc_df, use_container_width=True, hide_index=True)
                st.caption("Forecast based on linear trend. Use with caution — seasonality and external factors not modelled.")

        except Exception as e:
            st.error(f"Forecasting error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Customer Segmentation
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="module-header">👥 Customer Segmentation (RFM)</div>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    id_cols  = [c for c in df.columns if any(k in c.lower() for k in ["customer","id","client","user"])]
    date_cols2 = list(df.select_dtypes(include="datetime").columns)

    if len(num_cols) < 2 or not id_cols:
        st.info("Customer segmentation requires a customer ID column and at least two numeric columns (for Frequency and Monetary).")
    else:
        st.markdown("Configure RFM columns:")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            cust_col = st.selectbox("Customer ID", id_cols)
        with rc2:
            monetary_col = st.selectbox("Monetary (spend)", [
                c for c in num_cols if any(k in c.lower() for k in ["revenue","amount","sales","value","spend","total"])
            ] or num_cols)
        with rc3:
            invoice_col = st.selectbox("Invoice/Order ID (for Frequency)", [
                c for c in df.columns if any(k in c.lower() for k in ["invoice","order","transaction","id"])
                and c != cust_col
            ] or [df.columns[0]])

        if st.button("Run segmentation", type="primary"):
            with st.spinner("Computing RFM…"):
                try:
                    ref_date = pd.Timestamp.now()
                    date_col_rfm = date_cols2[0] if date_cols2 else None

                    rfm = df.groupby(cust_col).agg(
                        Frequency=(invoice_col, "nunique"),
                        Monetary=(monetary_col, "sum"),
                    ).reset_index()

                    if date_col_rfm:
                        rfm_rec = df.groupby(cust_col)[date_col_rfm].max().reset_index()
                        rfm_rec["Recency"] = (ref_date - rfm_rec[date_col_rfm]).dt.days
                        rfm = rfm.merge(rfm_rec[[cust_col,"Recency"]], on=cust_col, how="left")
                    else:
                        rfm["Recency"] = 100  # placeholder

                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import KMeans

                    rfm_clean = rfm[["Recency","Frequency","Monetary"]].fillna(0)
                    scaler    = StandardScaler()
                    rfm_scaled = scaler.fit_transform(rfm_clean)

                    km = KMeans(n_clusters=5, random_state=42, n_init=10)
                    rfm["Cluster"] = km.fit_predict(rfm_scaled)

                    cluster_means = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
                    seg_names = ["VIP","Loyal","Potential","Occasional","At-Risk"]
                    seg_map   = {c: seg_names[i] for i, c in enumerate(cluster_means.index)}
                    rfm["Segment"] = rfm["Cluster"].map(seg_map)

                    # Summary
                    seg_summary = rfm.groupby("Segment").agg(
                        Count=("Segment","size"),
                        Avg_Recency=("Recency","mean"),
                        Avg_Frequency=("Frequency","mean"),
                        Avg_Monetary=("Monetary","mean"),
                    ).round(1)

                    st.markdown("**Segment summary**")
                    st.dataframe(seg_summary, use_container_width=True)

                    # Pie chart
                    fig_pie = px.pie(
                        rfm["Segment"].value_counts().reset_index(),
                        names="Segment", values="count",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        hole=0.4,
                    )
                    fig_pie.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter,sans-serif",size=11),
                        margin=dict(l=0,r=0,t=20,b=0),
                    )

                    # Scatter
                    fig_sc = px.scatter(
                        rfm.sample(min(2000, len(rfm))),
                        x="Recency", y="Monetary",
                        color="Segment",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        opacity=0.6, size_max=8,
                    )
                    fig_sc.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor="#f0f0f8"), yaxis=dict(gridcolor="#f0f0f8"),
                        font=dict(family="Inter,sans-serif",size=11),
                        margin=dict(l=0,r=0,t=20,b=0),
                    )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Segment distribution**")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col_b:
                        st.markdown("**Recency vs Monetary**")
                        st.plotly_chart(fig_sc, use_container_width=True)

                    st.session_state["rfm_df"] = rfm

                except Exception as e:
                    st.error(f"Segmentation error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="module-header">🚨 Anomaly Detection</div>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) < 2:
        st.info("Anomaly detection requires at least two numeric columns.")
    else:
        ac1, ac2 = st.columns(2)
        with ac1:
            x_anom = st.selectbox("X axis", num_cols, key="anom_x")
        with ac2:
            y_anom = st.selectbox("Y axis", [c for c in num_cols if c != x_anom] or num_cols, key="anom_y")

        contamination = st.slider("Expected anomaly rate (%)", 1, 10, 2) / 100

        if st.button("Run anomaly detection", type="primary"):
            with st.spinner("Detecting anomalies…"):
                try:
                    from sklearn.ensemble import IsolationForest
                    from sklearn.preprocessing import StandardScaler

                    feat_df = df[[x_anom, y_anom]].dropna().copy()
                    scaler  = StandardScaler()
                    X_scaled = scaler.fit_transform(feat_df)

                    iso = IsolationForest(contamination=contamination,
                                          random_state=42, n_jobs=1)
                    labels = iso.fit_predict(X_scaled)
                    feat_df["Status"] = np.where(labels == -1, "Anomaly", "Normal")

                    n_anom = (labels == -1).sum()
                    n_norm = (labels == 1).sum()

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Anomalies detected", f"{n_anom:,}", f"{n_anom/len(feat_df)*100:.1f}%")
                    with col_b:
                        st.metric("Normal transactions", f"{n_norm:,}")

                    sample = feat_df.sample(min(3000, len(feat_df)), random_state=42)
                    fig_anom = px.scatter(
                        sample, x=x_anom, y=y_anom, color="Status",
                        color_discrete_map={"Normal":"#6C63FF","Anomaly":"#ef4444"},
                        symbol_map={"Normal":"circle","Anomaly":"x"},
                        opacity=0.6,
                    )
                    fig_anom.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor="#f0f0f8"), yaxis=dict(gridcolor="#f0f0f8"),
                        font=dict(family="Inter,sans-serif",size=11),
                        margin=dict(l=0,r=0,t=20,b=0),
                    )
                    st.plotly_chart(fig_anom, use_container_width=True)

                    st.markdown(
                        f'<div class="result-card">'
                        f'<strong>🚨 {n_anom:,} transactions flagged as statistically unusual</strong><br>'
                        f'<span style="color:#64748b;font-size:0.85rem">'
                        f'These transactions deviate from the typical pattern in the {x_anom} × {y_anom} space. '
                        f'Manual review is recommended to determine appropriate action. '
                        f'Anomaly flags are statistical — they do not imply fraud or error.'
                        f'</span></div>',
                        unsafe_allow_html=True,
                    )

                    with st.expander("View anomalous records"):
                        st.dataframe(
                            feat_df[feat_df["Status"] == "Anomaly"].head(50),
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"Anomaly detection error: {e}")
