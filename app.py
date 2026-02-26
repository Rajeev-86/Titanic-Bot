"""
Streamlit frontend for Titanic Dataset Chat Agent.
Communicates with the FastAPI backend to answer questions
and display visualizations.
"""

import streamlit as st
import requests
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Dataset ChatBot",
    page_icon="🚢",
    layout="wide",
)

# Support both Streamlit Cloud secrets and env vars
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .user-msg {
        background-color: #e8f0fe;
        border-left: 4px solid #4285f4;
    }
    .bot-msg {
        background-color: #f1f3f4;
        border-left: 4px solid #34a853;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🚢 Titanic Dataset ChatBot")
st.caption("Ask anything about the Titanic passengers — get answers and charts!")
st.markdown('</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Example Questions")
    examples = [
        "What percentage of passengers were male?",
        "Show me a histogram of passenger ages",
        "What was the average ticket fare?",
        "How many passengers embarked from each port?",
        "What was the survival rate by passenger class?",
        "Show a bar chart of survival by gender",
        "How many children (age < 12) survived?",
        "Show the distribution of fare prices",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    st.markdown(
        "**Tech Stack:** FastAPI · LangChain · Gemini · Streamlit"
    )

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart"):
            st.image(msg["chart"], use_container_width=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
if prompt := (prefill or st.chat_input("Ask a question about the Titanic dataset...")):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing the dataset..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": prompt},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                answer_text = data["text"]
                chart_url = data.get("chart_url")

                st.markdown(answer_text)

                chart_full_url = None
                if chart_url:
                    chart_full_url = f"{BACKEND_URL}{chart_url}"
                    st.image(chart_full_url, use_container_width=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "chart": chart_full_url,
                })

            except requests.exceptions.ConnectionError:
                err = "⚠️ Cannot connect to the backend. Make sure the FastAPI server is running on port 8000."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"⚠️ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
