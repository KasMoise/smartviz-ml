"""
storage.py — Cloud-compatible response storage.

Priority:
  1. Google Sheets (if credentials configured in st.secrets)
  2. Local CSV fallback (local dev / Streamlit Cloud ephemeral storage)

Setup for Google Sheets (recommended for Streamlit Cloud):
  - Create a Google Sheet named "SmartViz_HumanEval"
  - Share it with the service account email in your credentials JSON
  - In Streamlit Cloud → App settings → Secrets, add:

    [gsheets]
    spreadsheet_id = "YOUR_SHEET_ID"

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN RSA PRIVATE KEY-----\n..."
    client_email = "...@....iam.gserviceaccount.com"
    client_id = "..."
    token_uri = "https://oauth2.googleapis.com/token"
"""
import os
import json
import pandas as pd
import streamlit as st

RESP_FILE    = "human_eval_responses.csv"
SHEET_NAME   = "SmartViz_HumanEval"
COLS = [
    "participant_id","background","experience","query_id","domain",
    "business_question","ground_truth_viz","recommended_viz","confidence",
    "rank1_viz","rank2_viz","rank3_viz",
    "Relevance","Readability","Usefulness","Interpretability",
    "overall_satisfaction","comments","timestamp",
]


def _gsheets_client():
    """Return an authorised gspread client using st.secrets, or None."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception:
        return None


def save_response(row: dict) -> str:
    """
    Append one evaluation row.
    Returns 'sheets' if saved to Google Sheets, 'csv' if saved locally.
    """
    try:
        client = _gsheets_client()
        if client and "gsheets" in st.secrets:
            sheet_id = st.secrets["gsheets"]["spreadsheet_id"]
            sh       = client.open_by_key(sheet_id)
            try:
                ws = sh.worksheet(SHEET_NAME)
            except Exception:
                ws = sh.add_worksheet(title=SHEET_NAME, rows=2000, cols=len(COLS))
                ws.append_row(COLS)
            ws.append_row([str(row.get(c, "")) for c in COLS])
            return "sheets"
    except Exception:
        pass

    # Local CSV fallback
    df_new = pd.DataFrame([row])
    if os.path.exists(RESP_FILE):
        existing = pd.read_csv(RESP_FILE)
        df_all   = pd.concat([existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(RESP_FILE, index=False)
    return "csv"


def load_responses() -> pd.DataFrame:
    """Load all collected responses (Sheets first, then local CSV)."""
    try:
        client = _gsheets_client()
        if client and "gsheets" in st.secrets:
            sheet_id = st.secrets["gsheets"]["spreadsheet_id"]
            sh = client.open_by_key(sheet_id)
            ws = sh.worksheet(SHEET_NAME)
            data = ws.get_all_records()
            if data:
                return pd.DataFrame(data)
    except Exception:
        pass

    if os.path.exists(RESP_FILE):
        return pd.read_csv(RESP_FILE)
    return pd.DataFrame(columns=COLS)


def responses_as_csv() -> bytes:
    """Return all responses as CSV bytes for download."""
    df = load_responses()
    return df.to_csv(index=False).encode("utf-8")
