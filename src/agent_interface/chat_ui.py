"""
Streamlit web interface for phishing email detection.

This UI acts as a thin client to the Phisher2025 FastAPI backend.

Features:
- Allows pasting an email OR uploading a .txt file.
- Calls the backend API to get a model prediction.
- Runs a lightweight social-engineering heuristic analysis.
- Displays the combined results to the user.
"""

import logging
import re
from pathlib import Path
import sys
import streamlit as st
import requests
import os

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------- Configuration ----------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_URL}/api/v1/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"

# ---------- Logging setup ----------
logger = logging.getLogger("phisher_ui")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def inject_dark_mode(enabled: bool):
    """Inject minimal dark-mode CSS into the page when enabled."""
    if not enabled:
        return
    dark_css = """
    <style>
    .stApp { background-color: #0e1117; color: #d6deeb; }
    .stButton>button { background-color:#1f6feb; color: white; }
    .stMarkdown, .stText, .stNumberInput { color: #d6deeb; }
    .stDataFrame table { color: #d6deeb; }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)


# ----------------------------- API Client Functions -----------------------------

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_api_health() -> dict:
    """Check the health of the backend API."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API health check failed: {e}")
        return {"status": "error", "model_loaded": False}


def get_prediction(text: str) -> dict | None:
    """Get a phishing prediction from the API."""
    try:
        payload = {"text": text}
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        logger.error(f"API prediction request failed: {e}")
        return None


# ----------------------------- Utility functions -----------------------------

def extract_urls(text: str):
    # Simple regex to capture URLs
    url_pattern = r"(https?://\S+|www\\.\S+|\b\S+\.(com|ru|tk|cn|xyz|info|tk|ml)\b)"
    return re.findall(url_pattern, text, flags=re.IGNORECASE)


def has_suspicious_tld(urls):
    suspicious = []
    for u in urls:
        # u may be a tuple because of regex groups
        url = u[0] if isinstance(u, tuple) else u
        if re.search(r"\.tk\b|\.ru\b|\.cn\b|\.ml\b", url, flags=re.IGNORECASE):
            suspicious.append(url)
    return suspicious


def social_engineering_analysis(text: str) -> dict:
    """
    Lightweight heuristics to flag common social-engineering cues.
    Returns a dict with boolean flags and small explanation strings.
    """
    text_lower = text.lower()
    urgency_words = ["urgent", "immediately", "asap", "now", "action required", "last chance", "warning", "suspend"]
    urgency_found = [w for w in urgency_words if w in text_lower]
    cred_words = ["password", "verify", "login", "credentials", "confirm", "ssn", "cvv", "account number", "reset password"]
    creds_found = [w for w in cred_words if w in text_lower]
    generic_salutations = ["dear customer", "dear valued customer", "dear user", "valued customer", "dear member"]
    generic_found = [w for w in generic_salutations if w in text_lower]
    attachment_words = ["attachment", "attached", "invoice", "pdf", ".doc", ".xls", "download"]
    attachments_found = [w for w in attachment_words if w in text_lower]
    urls = extract_urls(text)
    suspicious_urls = has_suspicious_tld(urls)
    impersonation_cues = []
    if re.search(r"from: .*@.*", text, flags=re.IGNORECASE):
        impersonation_cues.append("Contains 'From:' header (possible raw email paste)")

    score = 0
    score += min(len(urgency_found), 3) * 0.2
    score += min(len(creds_found), 3) * 0.25
    score += min(len(attachments_found), 2) * 0.15
    score += min(len(suspicious_urls), 3) * 0.25
    score = min(score, 0.99)

    return {
        "urgency_found": urgency_found,
        "creds_found": creds_found,
        "generic_salutation": generic_found,
        "attachments_mentioned": attachments_found,
        "urls": [u[0] if isinstance(u, tuple) else u for u in urls],
        "suspicious_urls": suspicious_urls,
        "impersonation_cues": impersonation_cues,
        "heuristic_score": score,
    }


# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(page_title="Phisher2025 - Phishing Detector", page_icon="🚨", layout="wide")

st.title("🚨 Phisher2025 - Phishing Email Detector")
st.markdown("Use the box below to paste an email or upload a .txt file. The backend API will classify the email and the UI will provide simple social-engineering cues.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    dark_mode = st.checkbox("Dark mode", value=True)
    if dark_mode:
        inject_dark_mode(True)
    
    st.markdown("---")
    st.header("API Status")
    health = get_api_health()
    if health.get("status") == "ok":
        st.success("API is connected")
        if health.get("model_loaded"):
            st.info("Model is loaded and ready.")
        else:
            st.warning("API is running, but the model is not loaded.")
    else:
        st.error("API connection failed.")
        st.caption(f"Could not connect to {API_URL}. Ensure the backend is running.")


# Input: upload or paste
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Upload email (.txt) or paste below", type=["txt"], help="Upload a raw .txt email or paste the body below")
    pasted = st.text_area("Or paste email text here", height=250)

with col2:
    st.markdown("### Quick Actions")
    if st.button("Use example phishing"):
        pasted = "Urgent: Verify your PayPal account immediately. Click here: bit.ly/verify2024"
    if st.button("Use example legitimate"):
        pasted = "Hello, your order from Amazon has shipped and will arrive soon."

# Decide which text to analyze
email_text = ""
if uploaded_file is not None:
    try:
        email_text = uploaded_file.getvalue().decode('utf-8')
    except Exception:
        email_text = uploaded_file.getvalue().decode('latin-1')
elif pasted and len(pasted.strip()) > 0:
    email_text = pasted


if not email_text:
    st.info("Paste an email or upload a .txt file to analyze.")
else:
    st.markdown("---")
    st.subheader("Analysis Results")

    # Run model prediction via API
    prediction_result = None
    with st.spinner("Calling prediction API..."):
        prediction_result = get_prediction(email_text)

    # Heuristic social-engineering analysis
    heur = social_engineering_analysis(email_text)

    # Display results side-by-side
    left, right = st.columns(2)

    with left:
        st.markdown("### Model Prediction (from API)")
        if prediction_result is None:
            st.warning("Model prediction failed or API is unavailable.")
        else:
            model_label = prediction_result.get("label")
            model_score = prediction_result.get("raw_score")
            
            st.write(f"**Label:** {model_label}")
            st.write(f"**Raw score:** {model_score:.3f}")
            conf = model_score if model_label == 'PHISHING' else 1.0 - model_score
            st.write(f"**Confidence:** {conf:.1%}")
            st.progress(min(max(model_score, 0.0), 1.0))

    with right:
        st.markdown("### Social-engineering Heuristics (UI-based)")
        st.write(f"**Heuristic risk (0-1):** {heur['heuristic_score']:.3f}")
        if heur['urgency_found']:
            st.warning(f"Urgency cues: {heur['urgency_found']}")
        else:
            st.write("No urgency keywords detected.")

        if heur['creds_found']:
            st.error(f"Credential-related words found: {heur['creds_found']}")
        else:
            st.write("No direct credential request words detected.")

        if heur['generic_salutation']:
            st.info(f"Generic salutation(s): {heur['generic_salutation']}")

        if heur['attachments_mentioned']:
            st.warning(f"Attachment words: {heur['attachments_mentioned']}")

        if heur['urls']:
            st.write("Detected URLs:")
            for u in heur['urls']:
                st.code(u)
            if heur['suspicious_urls']:
                st.error(f"Suspicious TLDs or domains: {heur['suspicious_urls']}")

        if heur['impersonation_cues']:
            for c in heur['impersonation_cues']:
                st.write(c)

    st.markdown("---")
    st.subheader("Recommendations & Next Steps")

    final_label = prediction_result.get("label") if prediction_result else 'LEGITIMATE'
    if final_label == 'PHISHING' or heur['heuristic_score'] > 0.4:
        st.warning("⚠️ This email contains multiple phishing indicators. Do NOT click links or provide credentials.")
        st.markdown("- Report the email to your security team or provider.\n- Do not follow links.\n- Inspect sender domain carefully.")
    else:
        st.success("✅ The email appears low-risk based on quick checks. Still exercise caution.")
        st.markdown("- If in doubt, verify with the sender through a different channel.\n- Avoid entering credentials from email links.")

    # Show full email (collapsible)
    with st.expander("Show full email text"):
        st.code(email_text)


# Footer
st.markdown("---")
st.caption("Phisher2025 • Demo UI • No email data is stored")
