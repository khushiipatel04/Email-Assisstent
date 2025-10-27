import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import spacy
import os

# ============================================================
# PART 1: MOCK DATABASE & ML/AI MODELS
# ============================================================

@st.cache_data
def get_mock_data():
    csv_file_path = 'emails.csv'
    if not os.path.exists(csv_file_path):
        st.error(f"Error: The data file '{csv_file_path}' was not found.")
        return pd.DataFrame(columns=['subject', 'body', 'intent'])

    try:
        df = pd.read_csv(csv_file_path)
        required_columns = ['body', 'intent']

        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain: {required_columns}")
            st.error(f"Found: {list(df.columns)}")
            return pd.DataFrame(columns=['subject', 'body', 'intent'])

        if 'subject' not in df.columns:
            df['subject'] = "Email Subject"

        df.dropna(subset=required_columns, inplace=True)
        for col in required_columns + ['subject']:
            df[col] = df[col].astype(str)

        st.success(f"Loaded {len(df)} emails from {csv_file_path}")
        return df

    except Exception as e:
        st.error(f"Error loading {csv_file_path}: {e}")
        return pd.DataFrame(columns=['subject', 'body', 'intent'])


@st.cache_resource
def train_triage_model(df):
    text_column = 'body'
    label_column = 'intent'

    if df.empty:
        st.warning("Cannot train model: empty data.")
        return None

    df_clean = df.dropna(subset=[text_column, label_column])
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000))
    ])

    try:
        X_train = df_clean[text_column]
        y_train = df_clean[label_column]
        model_pipeline.fit(X_train, y_train)
        st.info("Triage model trained successfully.")
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

    return model_pipeline


@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Run this once in terminal: python -m spacy download en_core_web_sm")
        return None


# ============================================================
# PART 2: GEMINI API HANDLER
# ============================================================

def list_available_models(api_key):
    """Helper to list available Gemini models for your API key."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        st.write("🧠 Available Gemini models:", models)
        return models
    except Exception as e:
        st.error(f"Error fetching model list: {e}")
        return []


def generate_ai_reply(email_body):
    """
    Calls the 'Stage 2' Generative AI (Gemini) to draft a reply.
    Uses Gemini 2.0 Flash API.
    Silent version: no progress or debug messages in the Streamlit UI.
    """

    api_key = "AIzaSyDq-FJAEZdgYx2sL0110hu-vhFJO6BORPA"  # your valid API key
    model_name = "gemini-2.0-flash"  # ✅ stable and available model
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # --- Build the prompt ---
    system_prompt = (
        "You are a helpful and professional business assistant. "
        "Write a polite, concise, and professional email reply. "
        "If the email is a request (like for a file or meeting), politely agree to it. "
        "If it's an important notice (like a referral), thank them. "
        "Write only the reply text — no greetings or signatures."
    )

    full_prompt = f"{system_prompt}\n\nEmail:\n{email_body}\n\nReply:"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": full_prompt}]}]
    }

    headers = {"Content-Type": "application/json"}

    # --- Make the API call silently ---
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            reply_text = (
                result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()
            if not reply_text:
                reply_text = "AI Error: Empty response from Gemini."
            return reply_text
        else:
            print(f"Gemini API error {response.status_code}: {response.text}")
            return "AI Error: Could not generate reply."
    except Exception as e:
        print(f"Gemini API Exception: {e}")
        return "AI Error: Something went wrong."


# ============================================================
# PART 3: LOAD DATA & MODELS
# ============================================================

if 'generated_replies' not in st.session_state:
    st.session_state['generated_replies'] = {}

df = get_mock_data()
triage_model = train_triage_model(df) if not df.empty else None
nlp = load_spacy_model() if not df.empty else None

# ============================================================
# PART 4: STREAMLIT UI
# ============================================================

st.set_page_config(layout="wide")
st.title("🤖 Two-Stage AI Email Assistant")
st.markdown("Uses **Scikit-learn** for triage & **Gemini AI** for replies.")
st.divider()

# Stage 1 — Triage Dashboard
st.header("Stage 1: Triage Dashboard")

if df.empty or triage_model is None:
    st.warning("Cannot proceed: Data unavailable or triage model failed.")
else:
    if 'email_count_slider' not in st.session_state:
        st.session_state.email_count_slider = min(10, len(df))

    email_count = st.slider("How many emails to analyze?", 1, len(df), key='email_count_slider')

    if st.button("🚀 Analyze Email Batch", type="primary", use_container_width=True):
        email_batch = df.head(email_count)
        try:
            intent_labels = list(triage_model.classes_)
        except AttributeError:
            intent_labels = df['intent'].unique()

        counters = {label: 0 for label in intent_labels}
        actionable_data = []
        actionable_intents = ["FILE_REQUEST", "MEETING_REQUEST", "REFERRAL"]
        bar = st.progress(0, text="Analyzing...")

        for i, (index, row) in enumerate(email_batch.iterrows()):
            email_body = str(row['body'])
            try:
                intent = triage_model.predict([email_body])[0]
            except Exception:
                continue

            counters[intent] = counters.get(intent, 0) + 1

            if intent in actionable_intents:
                actionable_data.append({
                    'index': index,
                    'subject': row.get('subject', 'No Subject'),
                    'body': email_body,
                    'intent': intent
                })

            bar.progress((i + 1) / email_count)

        bar.empty()
        st.session_state['triage_counters'] = counters
        st.session_state['actionable_emails'] = actionable_data
        st.session_state['triage_done'] = True
        st.session_state['generated_replies'] = {}

# Stage 2 — Replies
if st.session_state.get('triage_done'):
    counters = st.session_state['triage_counters']
    actionable_emails = st.session_state['actionable_emails']

    st.subheader("📊 Triage Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("File Requests", counters.get("FILE_REQUEST", 0))
    col2.metric("Meeting Requests", counters.get("MEETING_REQUEST", 0))
    col3.metric("Referrals", counters.get("REFERRAL", 0))

    st.divider()
    st.header("Stage 2: Action & Reply Generator")

    if not actionable_emails:
        st.success("No actionable emails found!")
    else:
        for idx, email in enumerate(actionable_emails):
            subject = email['subject']
            body = email['body']
            key_suffix = f"{idx}_{email['index']}"
            reply_key = f"reply_{key_suffix}"

            with st.container(border=True):
                st.subheader(f"Subject: {subject}")
                st.caption(f"Original: {body}")

                if reply_key in st.session_state['generated_replies']:
                    st.text_area(
                        "Suggested Reply:",
                        value=st.session_state['generated_replies'][reply_key],
                        height=100
                    )
                else:
                    if st.button("Generate AI Reply", key=f"btn_{key_suffix}"):
                        with st.spinner("Calling Gemini AI..."):
                            reply_text = generate_ai_reply(body)
                            st.session_state['generated_replies'][reply_key] = reply_text
                            st.rerun()
