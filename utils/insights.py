"""
insights.py — Generate calibrated business insights from DataFrame + recommendation.
No causal or fraud attribution claims.
"""
import pandas as pd
import numpy as np


INSIGHT_TEMPLATES = {
    "bar_chart": (
        "📊 **Ranking insight** — {top_label} leads with {top_val}, "
        "representing approximately {top_pct:.0f}% of the total. "
        "The top {n_top} categories account for {top_n_pct:.0f}% of the total. "
        "This pattern suggests concentrating operational attention on high-performing segments."
    ),
    "line_chart": (
        "📈 **Trend insight** — The metric shows a "
        "{direction} trajectory over the observed period. "
        "The peak value ({peak_val}) occurs at {peak_period}. "
        "Cross-reference with operational events to identify potential drivers."
    ),
    "scatter_plot": (
        "🔵 **Relationship insight** — The two variables show "
        "{strength} co-movement (correlation ≈ {corr:.2f}). "
        "Note: correlation does not imply causation. "
        "Further investigation is recommended before drawing operational conclusions."
    ),
    "pie_chart": (
        "🥧 **Proportion insight** — {top_label} accounts for the largest share "
        "({top_pct:.0f}%). "
        "{'High concentration in one segment may warrant diversification review.' if top_pct > 50 else 'The distribution appears relatively balanced across segments.'}"
    ),
    "histogram": (
        "📉 **Distribution insight** — Values range from {v_min:.2f} to {v_max:.2f} "
        "(median: {median:.2f}). "
        "The distribution is {skew_desc}. "
        "Extreme values at the tails may merit closer inspection."
    ),
    "box_plot": (
        "📦 **Variability insight** — The interquartile range spans {iqr:.2f}, "
        "indicating {var_desc} variability. "
        "{outlier_note}"
    ),
    "heatmap": (
        "🌡️ **Correlation insight** — The strongest positive correlation is between "
        "{max_pair} (r = {max_corr:.2f}). "
        "Highly correlated variables may contain redundant information for predictive modelling."
    ),
    "choropleth": (
        "🗺️ **Geographic insight** — {top_label} shows the highest value ({top_val}). "
        "Geographic concentration suggests potential for targeted regional strategies."
    ),
}


def generate_insight(df: pd.DataFrame, viz_type: str, query: str = "") -> str:
    """Return a hedged, calibrated business insight string."""
    if df is None or df.empty:
        return "_No data available to generate insight._"

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    x_col = cat_cols[0]  if cat_cols  else (num_cols[0]  if num_cols  else df.columns[0])
    y_col = num_cols[0]  if num_cols  else df.columns[-1]
    dt_col = date_cols[0] if date_cols else None

    for col in num_cols:
        if any(k in col.lower() for k in ["revenue","amount","sales","value","total","profit"]):
            y_col = col; break
    for col in cat_cols:
        if any(k in col.lower() for k in ["product","category","name","type","segment","country","region"]):
            x_col = col; break

    try:
        if viz_type == "bar_chart":
            agg = df.groupby(x_col)[y_col].sum()
            top_label = agg.idxmax()
            top_val   = f"{agg.max():,.0f}"
            top_pct   = agg.max() / agg.sum() * 100
            n_top     = min(5, len(agg))
            top_n_pct = agg.nlargest(n_top).sum() / agg.sum() * 100
            return INSIGHT_TEMPLATES["bar_chart"].format(
                top_label=top_label, top_val=top_val,
                top_pct=top_pct, n_top=n_top, top_n_pct=top_n_pct
            )

        elif viz_type == "line_chart":
            col = dt_col if dt_col else None
            series = df[y_col].dropna()
            peak_val = f"{series.max():,.0f}"
            if col:
                try:
                    df2 = df.copy()
                    df2[col] = pd.to_datetime(df2[col], errors="coerce")
                    agg = df2.groupby(col)[y_col].sum()
                    peak_period = str(agg.idxmax())[:10]
                    direction = "upward" if agg.iloc[-1] > agg.iloc[0] else "downward"
                except Exception:
                    peak_period = "the observed peak"; direction = "variable"
            else:
                peak_period = f"index {series.idxmax()}"; direction = "variable"
            return INSIGHT_TEMPLATES["line_chart"].format(
                direction=direction, peak_val=peak_val, peak_period=peak_period
            )

        elif viz_type == "scatter_plot":
            if len(num_cols) >= 2:
                corr = df[num_cols[0]].corr(df[num_cols[1]])
                strength = ("strong" if abs(corr) > 0.6
                            else "moderate" if abs(corr) > 0.3
                            else "weak")
            else:
                corr = 0.0; strength = "unclear"
            return INSIGHT_TEMPLATES["scatter_plot"].format(strength=strength, corr=corr)

        elif viz_type == "pie_chart":
            agg = df.groupby(x_col)[y_col].sum()
            top_label = agg.idxmax()
            top_pct   = agg.max() / agg.sum() * 100
            note = ("High concentration in one segment may warrant diversification review."
                    if top_pct > 50 else
                    "The distribution appears relatively balanced across segments.")
            return (f"🥧 **Proportion insight** — **{top_label}** accounts for the largest share "
                    f"(**{top_pct:.0f}%**). {note}")

        elif viz_type == "histogram":
            s = df[y_col].dropna()
            skew = s.skew()
            skew_desc = ("right-skewed (long tail of high values)"
                         if skew > 0.5 else
                         "left-skewed (long tail of low values)"
                         if skew < -0.5 else "approximately symmetric")
            return INSIGHT_TEMPLATES["histogram"].format(
                v_min=s.min(), v_max=s.max(), median=s.median(), skew_desc=skew_desc
            )

        elif viz_type == "box_plot":
            s = df[y_col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            n_out = int((s > upper).sum() + (s < q1 - 1.5 * iqr).sum())
            var_desc = "high" if iqr > s.std() else "moderate"
            outlier_note = (f"{n_out} statistical outlier(s) detected — manual review recommended."
                            if n_out > 0 else "No extreme outliers detected.")
            return INSIGHT_TEMPLATES["box_plot"].format(
                iqr=iqr, var_desc=var_desc, outlier_note=outlier_note
            )

        elif viz_type == "heatmap":
            if len(num_cols) >= 2:
                corr_m = df[num_cols[:12]].corr().abs()
                np.fill_diagonal(corr_m.values, 0)
                idx = np.unravel_index(corr_m.values.argmax(), corr_m.shape)
                max_pair = f"{corr_m.columns[idx[0]]} & {corr_m.columns[idx[1]]}"
                max_corr = corr_m.values[idx]
            else:
                max_pair = "the two variables"; max_corr = 0.0
            return INSIGHT_TEMPLATES["heatmap"].format(max_pair=max_pair, max_corr=max_corr)

        elif viz_type == "choropleth":
            geo_col = next(
                (c for c in cat_cols if any(k in c.lower() for k in ["country","region","state"])),
                x_col
            )
            agg = df.groupby(geo_col)[y_col].sum()
            top_label = agg.idxmax()
            top_val   = f"{agg.max():,.0f}"
            return INSIGHT_TEMPLATES["choropleth"].format(top_label=top_label, top_val=top_val)

    except Exception:
        pass

    return "_Unable to generate a specific insight for this dataset and visualization type._"
