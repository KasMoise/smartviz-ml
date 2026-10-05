"""
SmartViz-ML — Training Script
Run once to produce models/ artifacts consumed by app.py.
Usage:
    python train.py --data online_retail_II.xlsx
    python train.py --demo   # trains on synthetic data (no dataset needed)
"""

import argparse, json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
import joblib

from engine import (
    VIZ_TYPES, FEATURE_COLS, META_FEATURE_COLS, INTENT_KEYWORDS,
    SEGMENT_NAMES, encode_query, extract_meta_features,
)

RANDOM_STATE  = 42
N_CLUSTERS    = 5
N_TS_SPLITS   = 5
TS_TEST_SIZE  = 3
MODEL_DIR     = "models"

np.random.seed(RANDOM_STATE)

# ── Meta-profiles ────────────────────────────────────────────────────────────
META_PROFILES = {
    "temporal_num":  dict(n_numeric=2,n_categorical=0,n_datetime=1,max_cardinality=0, mean_correlation=0.4,missing_ratio=0.01,has_temporal=1,has_geo=0),
    "temporal_cat":  dict(n_numeric=1,n_categorical=2,n_datetime=1,max_cardinality=8, mean_correlation=0.2,missing_ratio=0.02,has_temporal=1,has_geo=0),
    "cat_num":       dict(n_numeric=1,n_categorical=1,n_datetime=0,max_cardinality=12,mean_correlation=0.1,missing_ratio=0.02,has_temporal=0,has_geo=0),
    "cat_prop":      dict(n_numeric=1,n_categorical=1,n_datetime=0,max_cardinality=5, mean_correlation=0.05,missing_ratio=0.0,has_temporal=0,has_geo=0),
    "cat_prop_many": dict(n_numeric=1,n_categorical=1,n_datetime=0,max_cardinality=25,mean_correlation=0.05,missing_ratio=0.0,has_temporal=0,has_geo=0),
    "num_dist":      dict(n_numeric=1,n_categorical=0,n_datetime=0,max_cardinality=0, mean_correlation=0.0,missing_ratio=0.03,has_temporal=0,has_geo=0),
    "num_dist_multi":dict(n_numeric=1,n_categorical=1,n_datetime=0,max_cardinality=4, mean_correlation=0.15,missing_ratio=0.02,has_temporal=0,has_geo=0),
    "num_num":       dict(n_numeric=2,n_categorical=0,n_datetime=0,max_cardinality=0, mean_correlation=0.65,missing_ratio=0.01,has_temporal=0,has_geo=0),
    "num_num_multi": dict(n_numeric=5,n_categorical=0,n_datetime=0,max_cardinality=0, mean_correlation=0.55,missing_ratio=0.02,has_temporal=0,has_geo=0),
    "geo_num":       dict(n_numeric=1,n_categorical=1,n_datetime=0,max_cardinality=40,mean_correlation=0.1,missing_ratio=0.05,has_temporal=0,has_geo=1),
    "cluster_cat":   dict(n_numeric=1,n_categorical=2,n_datetime=0,max_cardinality=6, mean_correlation=0.2,missing_ratio=0.01,has_temporal=0,has_geo=0),
    "cluster_num":   dict(n_numeric=3,n_categorical=1,n_datetime=0,max_cardinality=5, mean_correlation=0.3,missing_ratio=0.01,has_temporal=0,has_geo=0),
    "num_anom":      dict(n_numeric=2,n_categorical=0,n_datetime=0,max_cardinality=0, mean_correlation=0.4,missing_ratio=0.04,has_temporal=0,has_geo=0),
    "cat_rank":      dict(n_numeric=1,n_categorical=1,n_datetime=0,max_cardinality=15,mean_correlation=0.05,missing_ratio=0.0,has_temporal=0,has_geo=0),
}

SEED_RULES = [
    ("temporal_num","trend over time monthly weekly evolution forecast","line_chart",False),
    ("temporal_num","sales revenue growth decline prediction","line_chart",False),
    ("temporal_cat","category performance comparison trend","line_chart",True),
    ("cat_num","top best worst rank compare most least","bar_chart",False),
    ("cat_num","revenue category product country segment","bar_chart",False),
    ("cat_num","average mean median per group","bar_chart",True),
    ("cat_prop","proportion share percentage breakdown contribution","pie_chart",True),
    ("cat_prop_many","proportion share percentage many categories","bar_chart",True),
    ("num_dist","distribution spread range histogram frequency","histogram",False),
    ("num_dist","quartile median outlier interquartile whisker","box_plot",False),
    ("num_dist_multi","distribution compare groups spread variability","box_plot",True),
    ("num_num","correlation relationship two variables scatter","scatter_plot",False),
    ("num_num_multi","correlation matrix all variables heatmap","heatmap",False),
    ("num_num","versus relationship link association","scatter_plot",True),
    ("geo_num","map region country geographic location","choropleth",False),
    ("geo_num","sales country revenue geographic distribution","choropleth",True),
    ("cluster_cat","segment cluster group customer profile","bar_chart",True),
    ("cluster_num","segment scatter rfm recency frequency monetary","scatter_plot",True),
    ("num_anom","anomaly outlier unusual suspicious extreme","scatter_plot",True),
    ("cat_rank","ranking leaderboard top bottom sorted","bar_chart",False),
]

NOISE = {
    "n_numeric":(0,1),"n_categorical":(0,1),"n_datetime":(0,0),
    "max_cardinality":(0,5),"mean_correlation":(-0.05,0.05),
    "missing_ratio":(-0.005,0.005),"has_temporal":(0,0),"has_geo":(0,0),
}


def build_training_pool():
    rows = []
    for pk, intent_str, viz, ambig in SEED_RULES:
        prof  = META_PROFILES.get(pk, META_PROFILES["cat_num"])
        words = intent_str.split()
        n_var = 12 if not ambig else 8
        for _ in range(n_var):
            np.random.shuffle(words)
            q = " ".join(words[:min(5, len(words))])
            pert = {}
            for feat, (lo, hi) in NOISE.items():
                base = prof[feat]
                if isinstance(base, float):
                    pert[feat] = float(np.clip(base + np.random.uniform(lo, hi), 0, 1))
                else:
                    pert[feat] = int(max(0, base + np.random.randint(lo, hi + 1)))
            rows.append({"query": q, "primary_viz": viz, "ambiguous": ambig,
                         **pert, **encode_query(q)})
    return pd.DataFrame(rows)


def train_recommendation_model(pool_A):
    le = LabelEncoder().fit(sorted(VIZ_TYPES))
    pool_A["label"] = le.transform(pool_A["primary_viz"])
    X = pool_A[FEATURE_COLS].values.astype(float)
    y = pool_A["label"].values
    Xdf = pd.DataFrame(X, columns=FEATURE_COLS)

    candidates = {
        "Logistic Regression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, random_state=RANDOM_STATE),
    }

    best_name, best_model, best_f1 = None, None, -1.0
    print("\n[Recommendation] Training candidates on full Pool A:")
    for name, m in candidates.items():
        m.fit(Xdf, y)
        preds = m.predict(Xdf)
        f1 = f1_score(y, preds, average="macro", zero_division=0)
        print(f"  {name:30s}  train Macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, m

    print(f"\n  → Selected: {best_name}  (train Macro-F1={best_f1:.4f})")
    return best_model, le, best_name


def train_forecasting_model(df):
    monthly = (df.groupby(["Year", "Month"])
                 .agg(Revenue=("Revenue", "sum"))
                 .reset_index()
                 .sort_values(["Year", "Month"]))
    ts = monthly[["Revenue"]].copy()
    for lag in [1, 2, 3, 6]:
        ts[f"lag{lag}"] = ts["Revenue"].shift(lag)
    ts["roll3_mean"] = ts["Revenue"].shift(1).rolling(3).mean()
    ts["roll3_std"]  = ts["Revenue"].shift(1).rolling(3).std()
    ts["month_num"]  = monthly["Month"].values
    ts["year_num"]   = monthly["Year"].values
    ts = ts.dropna().reset_index(drop=True)

    FC_FEATS = [c for c in ts.columns if c != "Revenue"]
    X_ts = ts[FC_FEATS].values
    y_ts = ts["Revenue"].values

    tscv = TimeSeriesSplit(n_splits=N_TS_SPLITS, test_size=TS_TEST_SIZE)
    fc_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }
    best_fc_name, best_fc_rmse = None, float("inf")
    print("\n[Forecasting] 5-fold TimeSeriesSplit CV:")
    for name, m in fc_models.items():
        rmses = []
        for tr, te in tscv.split(X_ts):
            Xtr = pd.DataFrame(X_ts[tr], columns=FC_FEATS)
            Xte = pd.DataFrame(X_ts[te], columns=FC_FEATS)
            m.fit(Xtr, y_ts[tr])
            rmses.append(np.sqrt(mean_squared_error(y_ts[te], m.predict(Xte))))
        mean_rmse = float(np.mean(rmses))
        print(f"  {name:30s}  RMSE={mean_rmse:,.0f}")
        if mean_rmse < best_fc_rmse:
            best_fc_rmse, best_fc_name = mean_rmse, name

    best_fc = fc_models[best_fc_name]
    best_fc.fit(pd.DataFrame(X_ts[:-TS_TEST_SIZE], columns=FC_FEATS), y_ts[:-TS_TEST_SIZE])
    print(f"\n  → Selected: {best_fc_name}  (RMSE={best_fc_rmse:,.0f})")
    return best_fc, FC_FEATS, monthly


def train_segmentation_model(rfm_scaled):
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    km.fit(rfm_scaled)
    return km


def train_anomaly_model(X_anom):
    iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE, n_jobs=1)
    iso.fit(X_anom)
    return iso


def run(data_path=None, demo=False):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Recommendation model (always trained, no dataset needed) ──────────
    print("=" * 55)
    print("SmartViz-ML — Training Pipeline")
    print("=" * 55)
    print("\n[1/4] Building training pool …")
    pool_A = build_training_pool()
    print(f"  Pool A: {len(pool_A)} examples  |  classes: {pool_A['primary_viz'].value_counts().to_dict()}")

    rec_model, le_viz, rec_name = train_recommendation_model(pool_A)

    # ── 2. Retail analytics models (require dataset) ─────────────────────────
    fc_model = fc_feats = rfm = km = scaler_rfm = iso = monthly_data = None

    if not demo and data_path and os.path.exists(data_path):
        print(f"\n[2/4] Loading dataset: {data_path}")
        df_2009 = pd.read_excel(data_path, sheet_name="Year 2009-2010", engine="openpyxl")
        df_2010 = pd.read_excel(data_path, sheet_name="Year 2010-2011", engine="openpyxl")
        df = pd.concat([df_2009, df_2010], ignore_index=True)
        df = df.dropna(subset=["Customer ID"])
        df["Customer ID"] = df["Customer ID"].astype(int)
        df = df[~df["Invoice"].astype(str).str.startswith("C")]
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].drop_duplicates()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Year"]  = df["InvoiceDate"].dt.year
        df["Month"] = df["InvoiceDate"].dt.month
        df["Revenue"] = df["Quantity"] * df["Price"]
        print(f"  Clean rows: {len(df):,}")

        print("\n[3/4] Training forecasting model …")
        fc_model, fc_feats, monthly_data = train_forecasting_model(df)

        print("\n[4/4] Training segmentation & anomaly models …")
        REF_DATE = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        rfm = (df.groupby("Customer ID")
                 .agg(Recency=("InvoiceDate", lambda x: (REF_DATE - x.max()).days),
                      Frequency=("Invoice", "nunique"),
                      Monetary=("Revenue", "sum"))
                 .reset_index())
        scaler_rfm = StandardScaler()
        rfm_scaled = scaler_rfm.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
        km = train_segmentation_model(rfm_scaled)

        seg_map = {}
        for rank, (cid, _) in enumerate(
            rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False).items()
            if "Cluster" in rfm.columns
            else enumerate([])
        ):
            seg_map[cid] = SEGMENT_NAMES[rank]
        rfm["Cluster"] = km.predict(rfm_scaled)
        rfm["Segment"] = rfm["Cluster"].map(
            {cid: SEGMENT_NAMES[r] for r, (cid, _) in enumerate(
                rfm.groupby("Cluster")["Monetary"].mean()
                   .sort_values(ascending=False).items())}
        )

        X_anom = StandardScaler().fit_transform(df[["Quantity", "Price", "Revenue"]])
        iso = train_anomaly_model(X_anom)
        print("  Segmentation and anomaly models trained.")
    else:
        if demo:
            print("\n[2-4/4] Demo mode — skipping retail analytics models.")
        else:
            print(f"\n[2-4/4] Dataset not found at '{data_path}' — skipping retail analytics.")
            print("         Run:  python train.py --data online_retail_II.xlsx")

    # ── Save all models ───────────────────────────────────────────────────────
    print("\n[Saving] Writing model artifacts …")
    joblib.dump(rec_model, f"{MODEL_DIR}/smartviz_model.joblib")
    joblib.dump(le_viz,    f"{MODEL_DIR}/smartviz_le.joblib")

    meta = {
        "rec_model_name": rec_name,
        "FEATURE_COLS":   FEATURE_COLS,
        "VIZ_TYPES":      VIZ_TYPES,
        "SEGMENT_NAMES":  SEGMENT_NAMES,
        "has_retail":     fc_model is not None,
    }

    if fc_model is not None:
        joblib.dump(fc_model,   f"{MODEL_DIR}/fc_model.joblib")
        joblib.dump(scaler_rfm, f"{MODEL_DIR}/rfm_scaler.joblib")
        joblib.dump(km,         f"{MODEL_DIR}/km_model.joblib")
        joblib.dump(iso,        f"{MODEL_DIR}/iso_model.joblib")
        meta["fc_feats"] = fc_feats
        if monthly_data is not None:
            monthly_data.to_csv(f"{MODEL_DIR}/monthly.csv", index=False)
        if rfm is not None:
            rfm.to_csv(f"{MODEL_DIR}/rfm.csv", index=False)

    with open(f"{MODEL_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nArtifacts saved:")
    for fn in sorted(os.listdir(MODEL_DIR)):
        sz = os.path.getsize(f"{MODEL_DIR}/{fn}") / 1024
        print(f"  {MODEL_DIR}/{fn:<35s} {sz:6.1f} KB")

    print("\nTraining complete.")
    return rec_model, le_viz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartViz-ML Training Script")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to online_retail_II.xlsx")
    parser.add_argument("--demo", action="store_true",
                        help="Train recommendation model only (no dataset needed)")
    args = parser.parse_args()
    run(data_path=args.data, demo=args.demo)
