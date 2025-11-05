import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import base64
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import spacy
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from bs4 import BeautifulSoup
import joblib 
import re
import html2text

# ============================================================
# PART 1: GMAIL AUTH + FETCH
# ============================================================

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def get_gmail_service():
    """Authenticate and return Gmail API service"""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())
    service = build("gmail", "v1", credentials=creds)
    return service


def fetch_latest_emails(limit=100):
    """
    Fetch and clean the latest Gmail messages (non-promotional).
    Removes repetitive newsletter content, unsubscribe text, and footer junk.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=limit,
        q="-category:promotions -category:social"  # Exclude Gmail Promotions/Social
    ).execute()

    messages = results.get("messages", [])
    emails = []

    if not messages:
        st.warning("No emails found.")
        return pd.DataFrame(columns=["subject", "body", "intent"])

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()

        headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(No Subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(Unknown Sender)")

        html_body = ""
        plain_body = ""

        # --- Decode message parts ---
        def extract_parts(payload):
            nonlocal html_body, plain_body
            if "parts" in payload:
                for part in payload["parts"]:
                    extract_parts(part)
            else:
                mime_type = payload.get("mimeType", "")
                data = payload["body"].get("data")
                if data:
                    decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    if "html" in mime_type:
                        html_body += decoded
                    elif "plain" in mime_type:
                        plain_body += decoded

        extract_parts(msg_data["payload"])

        # --- Clean and extract main content ---
        body_text = ""
        if html_body:
            soup = BeautifulSoup(html_body, "html.parser")

            # Remove irrelevant tags
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
                tag.decompose()

            # Find the main content block
            main_candidates = soup.find_all(["div", "table", "section", "article"], recursive=True)
            largest = max(main_candidates, key=lambda x: len(x.get_text(strip=True)), default=soup)
            body_text = largest.get_text(separator="\n", strip=True)

        elif plain_body:
            body_text = plain_body.strip()
        else:
            body_text = "(No Content)"

        # --- Clean whitespace and remove unsubscribe text ---
        body_text = re.sub(r'\s*\n\s*', '\n', body_text)
        body_text = re.sub(r'[ \t]+', ' ', body_text)
        body_text = re.sub(
            r"(unsubscribe|manage preferences|update categories|view in browser|sponsor(ed)?|©|\bAll rights reserved\b).*",
            "",
            body_text,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()

        # --- Skip empty or clearly promotional emails ---
        if any(x in sender.lower() for x in ["bookbub", "quora", "dribbble", "newsletter"]):
            continue
        if len(body_text.split()) < 20:
            continue

        emails.append({
            "subject": subject,
            "body": body_text,
            "intent": "UNKNOWN"
        })

    return pd.DataFrame(emails)

# ============================================================
# PART 2: CSV LOAD (EXACT BEHAVIOUR AS ORIGINAL CODE)
# ============================================================

@st.cache_data
def get_mock_data():
    csv_file_path = "emails.csv"
    if not os.path.exists(csv_file_path):
        st.error(f"CSV file '{csv_file_path}' not found.")
        return pd.DataFrame(columns=["subject", "body", "intent"])
    try:
        df = pd.read_csv(csv_file_path)
        required = ["body", "intent"]
        if not all(col in df.columns for col in required):
            st.error(f"CSV must contain: {required}")
            st.error(f"Found: {list(df.columns)}")
            return pd.DataFrame(columns=["subject", "body", "intent"])
        if "subject" not in df.columns:
            df["subject"] = "Email Subject"
        df.dropna(subset=required, inplace=True)
        for col in required + ["subject"]:
            df[col] = df[col].astype(str)
        # st.success(f"Loaded {len(df)} emails from CSV ✅")
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame(columns=["subject", "body", "intent"])


# ============================================================
# PART 3: ML TRIAGE + SPACY + GEMINI
# ============================================================

@st.cache_resource
def train_triage_model(df):
    text_col, label_col = "body", "intent"
    if df.empty:
        st.warning("Empty dataset.")
        return None
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(multi_class="ovr", solver="liblinear", max_iter=1000)),
    ])
    model.fit(df[text_col], df[label_col])
    st.info("✅ Triage model trained successfully (on CSV data).")
    return model

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Please run: python -m spacy download en_core_web_sm")
        return None


def generate_ai_reply(email_body):
    """Call Gemini API silently"""
    api_key = "AIzaSyDq-FJAEZdgYx2sL0110hu-vhFJO6BORPA"
    model_name = "gemini-2.0-flash"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    system_prompt = (
        "You are a professional email assistant. Write a polite, concise reply "
        "without greetings or signatures."
    )
    prompt = f"{system_prompt}\n\nEmail:\n{email_body}\n\nReply:"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()
            return text or "AI Error: Empty response"
        return f"AI Error: {r.status_code}"
    except Exception as e:
        return f"AI Error: {e}"


# ============================================================
# PART 4: STREAMLIT DASHBOARD
# ============================================================

st.set_page_config(layout="wide")
st.title("📧 AI-Powered Gmail Assistant")
st.caption("Two-Stage: Email Triage + Gemini Reply")

# Model ko train karne ke liye CSV data hamesha load karein
csv_data_for_training = get_mock_data()

# Model ko sirf CSV data par train karein aur cache karein
triage_model = train_triage_model(csv_data_for_training)
nlp = load_spacy_model()

source = st.radio("Select Email Source", ["📩 Fetch from Gmail", "📁 Load from CSV"], horizontal=True)

if source == "📩 Fetch from Gmail":
    email_fetch_limit = st.number_input(
        "How many recent emails to fetch?", 
        min_value=5, 
        max_value=50, 
        value=10, 
        step=5
    )
    if st.button("Connect to Gmail and Fetch Emails"):
        with st.spinner(f"Fetching {email_fetch_limit} emails..."):
            df = fetch_latest_emails(limit=email_fetch_limit)
            st.session_state["emails_df"] = df
            if not df.empty:
                st.success(f"Fetched {len(df)} emails from Gmail.")
else:
    # "Load from CSV" select hone par, display ke liye CSV data set karein
    st.session_state["emails_df"] = csv_data_for_training
    st.success(f"Loaded {len(csv_data_for_training)} emails from CSV ✅")


if "emails_df" not in st.session_state or st.session_state["emails_df"].empty:
    st.warning("No emails available. Please fetch from Gmail or load from CSV.")
    st.stop()

df = st.session_state["emails_df"]

# ============================================================
# Stage 1: TRIAGE
# ============================================================

st.divider()
st.header("Stage 1: Email Classification (Triage)")

# <-- FIX YAHAN HAI: Variable ko global scope mein define kiya
actionable_intents = ["FILE_REQUEST", "MEETING_REQUEST", "REFERRAL"]

if df.empty or triage_model is None:
    st.warning("Cannot proceed — missing data or model.")
else:
    email_count = st.slider("How many emails to analyze?", 1, len(df), min(10, len(df)))

    if st.button("🚀 Analyze Emails", type="primary", use_container_width=True):
        batch = df.head(email_count)
        
        intents = list(triage_model.classes_)
        counters = {label: 0 for label in intents}
        counters["UNKNOWN"] = 0
        
        actionable = []
        # <-- FIX: Line yahan se hata di gayi hai
        
        bar = st.progress(0, text="Analyzing...")

        for i, (_, row) in enumerate(batch.iterrows()):
            try:
                intent = triage_model.predict([row["body"]])[0]
            except Exception:
                intent = "UNKNOWN"
            
            counters[intent] = counters.get(intent, 0) + 1
            if intent in actionable_intents:
                actionable.append({
                    "subject": row.get("subject", "No Subject"),
                    "body": row["body"],
                    "intent": intent
                })
            bar.progress((i + 1) / email_count)

        st.session_state["triage_done"] = True
        st.session_state["triage_counters"] = counters
        st.session_state["actionable_emails"] = actionable
        st.session_state["generated_replies"] = {}
        bar.empty()

# ============================================================
# Stage 2: REPLY GENERATION
# ============================================================

if st.session_state.get("triage_done"):
    counters = st.session_state["triage_counters"]
    actionable = st.session_state["actionable_emails"]

    st.divider()
    st.subheader("📊 Triage Summary")
    
    metric_cols = st.columns(len(counters))
    idx = 0
    for intent, count in counters.items():
        if count > 0 or intent in actionable_intents: 
            metric_cols[idx].metric(intent.replace("_", " ").title(), count)
            idx += 1

    st.divider()
    st.header(f"Stage 2: Actionable Replies ({len(actionable)} found)")

    if not actionable:
        st.success("No actionable emails found!")
    else:
        # --- YEH HAI AAPKA NAYA FORMATTING FIX ---
        
        for i, email in enumerate(actionable):
            with st.expander(f"**({email['intent']})** - {email['subject']}", expanded=False):
                
                # <pre> tag ka upyog karein - yeh formatting (newlines) ko preserve karta hai
                # style='white-space: pre-wrap;' text ko box ke andar wrap karta hai
                st.markdown(
                    f"<pre style='white-space: pre-wrap; word-wrap: break-word; font-family: sans-serif; font-size: 14px;'>{email['body']}</pre>", 
                    unsafe_allow_html=True
                )
                
                st.markdown("---") # Separator

                # Reply generation logic
                key = f"reply_{i}"
                if key in st.session_state["generated_replies"]:
                    st.text_area("Suggested Reply:", value=st.session_state["generated_replies"][key], height=120, key=f"txt_{i}")
                else:
                    if st.button("Generate AI Reply", key=f"btn_{i}"):
                        with st.spinner("Generating reply..."):
                            reply = generate_ai_reply(email["body"])
                            st.session_state["generated_replies"][key] = reply
                            st.rerun()
