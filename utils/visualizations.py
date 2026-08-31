"""
visualizations.py — Generate Plotly charts from a DataFrame + recommendation.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

PALETTE = px.colors.qualitative.Set2
ACCENT  = "#6C63FF"


def _pick_cols(df: pd.DataFrame, viz_type: str, query: str = ""):
    """Heuristically pick x/y columns from the DataFrame."""
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    q = query.lower()

    # Date columns: try parsing strings
    if not date_cols:
        for col in cat_cols[:]:
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if converted.notna().mean() > 0.7:
                    date_cols.append(col)
            except Exception:
                pass

    x_col  = cat_cols[0]  if cat_cols  else (num_cols[0] if num_cols else df.columns[0])
    y_col  = num_cols[0]  if num_cols  else df.columns[-1]
    dt_col = date_cols[0] if date_cols else None

    # Revenue / amount heuristic
    for col in num_cols:
        if any(k in col.lower() for k in ["revenue","amount","sales","price","value","total","profit"]):
            y_col = col
            break
    for col in cat_cols:
        if any(k in col.lower() for k in ["product","category","name","type","segment","country","region"]):
            x_col = col
            break

    return x_col, y_col, dt_col, num_cols, cat_cols


def generate_chart(df: pd.DataFrame, viz_type: str, query: str = "") -> go.Figure | None:
    """Return a Plotly figure appropriate for viz_type, or None if impossible."""
    if df is None or df.empty:
        return None

    x_col, y_col, dt_col, num_cols, cat_cols = _pick_cols(df, viz_type, query)

    try:
        if viz_type == "bar_chart":
            agg = df.groupby(x_col)[y_col].sum().nlargest(15).reset_index()
            fig = px.bar(agg, x=y_col, y=x_col, orientation="h",
                         color=y_col, color_continuous_scale="Blues",
                         labels={y_col: y_col, x_col: x_col})
            fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})

        elif viz_type == "line_chart":
            if dt_col:
                try:
                    df2 = df.copy()
                    df2[dt_col] = pd.to_datetime(df2[dt_col], errors="coerce")
                    df2 = df2.dropna(subset=[dt_col])
                    df2 = df2.sort_values(dt_col)
                    agg = df2.groupby(dt_col)[y_col].sum().reset_index()
                    fig = px.line(agg, x=dt_col, y=y_col, markers=True,
                                  color_discrete_sequence=[ACCENT])
                except Exception:
                    fig = px.line(df.head(200), y=y_col, color_discrete_sequence=[ACCENT])
            else:
                fig = px.line(df.head(200), y=y_col, color_discrete_sequence=[ACCENT])

        elif viz_type == "scatter_plot":
            y2 = num_cols[1] if len(num_cols) > 1 else y_col
            sample = df[[x_col if x_col in num_cols else y_col, y2]].dropna().head(500)
            x_s = sample.columns[0]
            fig = px.scatter(sample, x=x_s, y=y2,
                             color_discrete_sequence=[ACCENT], opacity=0.6,
                             trendline="ols")

        elif viz_type == "pie_chart":
            agg = df.groupby(x_col)[y_col].sum().nlargest(8).reset_index()
            fig = px.pie(agg, names=x_col, values=y_col,
                         color_discrete_sequence=PALETTE, hole=0.35)
            fig.update_traces(textposition="inside", textinfo="percent+label")

        elif viz_type == "histogram":
            fig = px.histogram(df[y_col].dropna().head(5000), x=y_col,
                               nbins=40, color_discrete_sequence=[ACCENT])
            fig.update_layout(bargap=0.05)

        elif viz_type == "box_plot":
            if cat_cols:
                top_cats = df[x_col].value_counts().head(8).index
                df2 = df[df[x_col].isin(top_cats)]
                fig = px.box(df2, x=x_col, y=y_col,
                             color=x_col, color_discrete_sequence=PALETTE)
            else:
                fig = px.box(df, y=y_col, color_discrete_sequence=[ACCENT])

        elif viz_type == "heatmap":
            corr = df[num_cols[:12]].corr().round(2)
            fig = px.imshow(corr, color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, text_auto=True, aspect="auto")

        elif viz_type == "choropleth":
            geo_col = next(
                (c for c in cat_cols if any(k in c.lower() for k in ["country","region","state","territory"])),
                x_col,
            )
            agg = df.groupby(geo_col)[y_col].sum().reset_index()
            fig = px.choropleth(agg, locations=geo_col, locationmode="country names",
                                color=y_col, color_continuous_scale="Blues",
                                labels={y_col: y_col})
        else:
            return None

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            xaxis=dict(gridcolor="#e8e8f0"),
            yaxis=dict(gridcolor="#e8e8f0"),
        )
        return fig

    except Exception as e:
        return None
