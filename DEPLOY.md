# Deployment Guide — Streamlit Community Cloud

## Step 1 — Push to GitHub

```bash
# Create a new repo on github.com (e.g. smartviz-ml)
git init
git add .
git commit -m "SmartViz-ML v8 — initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/smartviz-ml.git
git push -u origin main
```

## Step 2 — Add model files

The model files (`.joblib`) are too large for regular Git.
Two options:

**Option A — Git LFS (recommended)**
```bash
git lfs install
git lfs track "models/*.joblib"
git add .gitattributes models/
git commit -m "Add trained models via LFS"
git push
```

**Option B — Embed a lightweight model**
Run the notebook cell below after §22 to generate a smaller RF with 50 trees:
```python
import joblib
from sklearn.ensemble import RandomForestClassifier
rf_small = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
rf_small.fit(X_train_df, y_train_b)
joblib.dump(rf_small, 'models/smartviz_rf.joblib', compress=3)
# compress=3 reduces file size ~60%
```

## Step 3 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **New app**
3. Select your GitHub repo, branch `main`, file `app.py`
4. Click **Deploy**

## Step 4 — Configure secrets (for Google Sheets)

In Streamlit Cloud → your app → **Settings** → **Secrets**, paste:

```toml
[gsheets]
spreadsheet_id = "YOUR_SHEET_ID"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key = "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
client_email = "smartviz@....iam.gserviceaccount.com"
# ... (see .streamlit/secrets.toml.example for full template)
```

Without secrets configured, human evaluation responses save to a local CSV
(ephemeral on Streamlit Cloud — use the Download button after each session).

## Step 5 — Verify

Visit your app URL, check:
- [ ] Home page loads with model status
- [ ] Demo dataset loads on Data Upload
- [ ] Recommender returns results (RF model or fallback)
- [ ] Human Evaluation form completes and downloads CSV
- [ ] No CORS warnings in logs

## Without model files

The app runs fully in **fallback mode** (rule-based recommender).
To add the trained RF later, upload the `.joblib` files via Git LFS
and redeploy — no code changes needed.
