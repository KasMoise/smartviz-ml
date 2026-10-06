"""
visualizations.py — Generate Plotly charts from a DataFrame + recommendation.
Query-aware column selection: the user's question drives which column is used.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

PALETTE = px.colors.qualitative.Set2
ACCENT  = "#6C63FF"

# ── Keyword maps: query words → column name hints ─────────────────────────────
# Order matters: more specific first.
DIMENSION_HINTS = [
    # Product / item
    (["product","item","sku","article","description","goods"],
     ["product","description","item","sku","article","name","goods"]),
    # Customer / segment
    (["customer","client","segment","buyer","purchaser"],
     ["customer","segment","client","buyer"]),
    # Category
    (["categor","type","class","group","kind"],
     ["category","type","class","group","kind"]),
    # Geography — only when the query explicitly asks for it
    (["country","region","location","geographic","map","territory","state","province"],
     ["country","region","state","territory","location","province"]),
    # Department / team
    (["department","team","division","unit","ward","branch"],
     ["department","team","division","unit","ward","branch"]),
    # Supplier / vendor
    (["supplier","vendor","manufacturer"],
     ["supplier","vendor","manufacturer"]),
    # Time — fallback to date column when query is temporal
    (["time","date","month","week","year","quarter","period","trend"],
     []),  # handled separately via date_cols
]

METRIC_HINTS = [
    ["revenue","income","sales","turnover"],
    ["amount","value","total","sum"],
    ["profit","margin","earning"],
    ["quantity","qty","units","volume","count"],
    ["price","cost","rate","fee"],
    ["score","rating","index","kpi"],
]


def _pick_cols(df: pd.DataFrame, viz_type: str, query: str = ""):
    """
    Query-aware column picker.
    Analyses the user's question to select the most appropriate
    dimension (x) and metric (y) columns.
    """
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    q = query.lower()

    # Try to detect date columns hiding as strings
    if not date_cols:
        for col in cat_cols[:]:
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().mean() > 0.7:
                    date_cols.append(col)
            except Exception:
                pass

    # ── 1. Pick metric (y_col) from query ─────────────────────────────────────
    y_col = num_cols[0] if num_cols else (df.columns[-1])
    for hint_words in METRIC_HINTS:
        if any(w in q for w in hint_words):
            # Find a numeric column whose name matches
            for col in num_cols:
                if any(w in col.lower() for w in hint_words):
                    y_col = col
                    break
            else:
                # Query mentions the concept but no matching column name —
                # keep searching for best numeric column
                continue
            break
    # Fallback: prefer Revenue > Amount > Quantity > first numeric
    if y_col == num_cols[0] if num_cols else None:
        for col in num_cols:
            if any(k in col.lower() for k in ["revenue","amount","sales","value","total","profit"]):
                y_col = col
                break

    # ── 2. Pick dimension (x_col) from query ──────────────────────────────────
    x_col = cat_cols[0] if cat_cols else (num_cols[0] if num_cols else df.columns[0])

    matched = False
    for query_words, col_hints in DIMENSION_HINTS:
        if any(w in q for w in query_words):
            # Find a categorical column whose name matches the hints
            for hint in col_hints:
                for col in cat_cols:
                    if hint in col.lower():
                        x_col = col
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break

    # ── 3. If no match from query, use smart fallback priority ────────────────
    if not matched and cat_cols:
        PRIORITY_ORDER = [
            ["product","description","item","sku","article","name","goods"],
            ["category","type","class","group","kind"],
            ["segment","tier","level"],
            ["department","team","division"],
            ["country","region","state","territory"],  # geo last in fallback
        ]
        for hints in PRIORITY_ORDER:
            for col in cat_cols:
                if any(h in col.lower() for h in hints):
                    x_col = col
                    matched = True
                    break
            if matched:
                break

    dt_col = date_cols[0] if date_cols else None
    return x_col, y_col, dt_col, num_cols, cat_cols


def generate_chart(df: pd.DataFrame, viz_type: str, query: str = "") -> go.Figure | None:
    """Return a Plotly figure appropriate for viz_type, or None if impossible."""
    if df is None or df.empty:
        return None

    x_col, y_col, dt_col, num_cols, cat_cols = _pick_cols(df, viz_type, query)

    try:
        if viz_type == "bar_chart":
            agg = df.groupby(x_col)[y_col].sum().nlargest(15).reset_index()
            fig = px.bar(
                agg, x=y_col, y=x_col, orientation="h",
                color=y_col, color_continuous_scale="Blues",
                labels={y_col: y_col, x_col: x_col},
                title=f"{y_col} by {x_col}",
            )
            fig.update_layout(
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
            )

        elif viz_type == "line_chart":
            if dt_col:
                try:
                    df2 = df.copy()
                    df2[dt_col] = pd.to_datetime(df2[dt_col], errors="coerce")
                    df2 = df2.dropna(subset=[dt_col])
                    df2 = df2.sort_values(dt_col)
                    agg = df2.groupby(df2[dt_col].dt.to_period("M"))[y_col].sum().reset_index()
                    agg[dt_col] = agg[dt_col].astype(str)
                    fig = px.line(
                        agg, x=dt_col, y=y_col, markers=True,
                        color_discrete_sequence=[ACCENT],
                        title=f"{y_col} over time",
                    )
                except Exception:
                    fig = px.line(
                        df.head(200), y=y_col,
                        color_discrete_sequence=[ACCENT],
                    )
            else:
                fig = px.line(df.head(200), y=y_col, color_discrete_sequence=[ACCENT])

        elif viz_type == "scatter_plot":
            # For scatter: prefer two numeric columns
            y2 = num_cols[1] if len(num_cols) > 1 else y_col
            x_num = y_col if y_col in num_cols else (num_cols[0] if num_cols else x_col)
            sample = df[[x_num, y2]].dropna().head(500)
            fig = px.scatter(
                sample, x=x_num, y=y2,
                color_discrete_sequence=[ACCENT], opacity=0.6,
                trendline="ols",
                title=f"{x_num} vs {y2}",
            )

        elif viz_type == "pie_chart":
            agg = df.groupby(x_col)[y_col].sum().nlargest(8).reset_index()
            fig = px.pie(
                agg, names=x_col, values=y_col,
                color_discrete_sequence=PALETTE, hole=0.35,
                title=f"{y_col} breakdown by {x_col}",
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")

        elif viz_type == "histogram":
            fig = px.histogram(
                df[y_col].dropna().head(5000), x=y_col,
                nbins=40, color_discrete_sequence=[ACCENT],
                title=f"Distribution of {y_col}",
            )
            fig.update_layout(bargap=0.05)

        elif viz_type == "box_plot":
            if cat_cols:
                top_cats = df[x_col].value_counts().head(8).index
                df2 = df[df[x_col].isin(top_cats)]
                fig = px.box(
                    df2, x=x_col, y=y_col,
                    color=x_col, color_discrete_sequence=PALETTE,
                    title=f"{y_col} distribution by {x_col}",
                )
            else:
                fig = px.box(df, y=y_col, color_discrete_sequence=[ACCENT])

        elif viz_type == "heatmap":
            corr = df[num_cols[:12]].corr().round(2)
            fig = px.imshow(
                corr, color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, text_auto=True, aspect="auto",
                title="Correlation matrix",
            )

        elif viz_type == "choropleth":
            # For choropleth, always use a geographic column regardless of query
            geo_col = next(
                (c for c in cat_cols
                 if any(k in c.lower() for k in ["country","region","state","territory"])),
                x_col,
            )
            agg = df.groupby(geo_col)[y_col].sum().reset_index()
            fig = px.choropleth(
                agg, locations=geo_col, locationmode="country names",
                color=y_col, color_continuous_scale="Blues",
                labels={y_col: y_col},
                title=f"{y_col} by {geo_col}",
            )
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

    except Exception:
        return None
