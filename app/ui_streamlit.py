# app/ui_streamlit.py
import os, requests, json, streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="LLM-Powered Tech & Finance News", layout="wide")
st.title("📰 LLM-Powered Tech & Finance News")

with st.sidebar:
    st.header("Settings")
    api_override = st.text_input("API Base URL", value=API, help="e.g., http://localhost:8000")
    API = api_override or API
    st.caption("Tip: ensure the FastAPI server is running.")

tab1, tab2 = st.tabs(["Summarize", "Recommend"])

def _post(path: str, payload: dict, timeout: int = 90):
    url = f"{API}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# --- Summarize ---
with tab1:
    st.subheader("Summarize")
    mode = st.radio("Mode", options=["topic", "article"], horizontal=True)
    colA, colB = st.columns(2)
    query = colA.text_input("Topic query", placeholder="e.g., NVDA earnings and AI demand outlook")
    article_url = colB.text_input("Article URL (article mode)")
    tickers = st.text_input("Tickers (comma-separated)", placeholder="NVDA, MSFT")
    sources = st.text_input("Sources (comma-separated)", placeholder="WSJ, FT")
    k = st.slider("k (final #chunks)", min_value=3, max_value=10, value=8, step=1)
    run_sum = st.button("Run Summarization")

    if run_sum:
        payload = {
            "mode": mode,
            "query": query if mode == "topic" else None,
            "article_url": article_url if mode == "article" else None,
            "k": k,
            "tickers": [t.strip().upper() for t in tickers.split(",") if t.strip()] or None,
            "sources": [s.strip() for s in sources.split(",") if s.strip()] or None,
        }
        try:
            res = _post("/summarize", payload)
            bullets = res.get("bullets", [])
            if not bullets:
                st.info("No bullets returned.")
            else:
                st.markdown("### Summary")
                for b in bullets:
                    url = b.get("url")
                    text = b.get("text","")
                    if url:
                        st.markdown(f"- {text}  \n  🔗 {url}")
                    else:
                        st.markdown(f"- {text}")
        except requests.HTTPError as e:
            st.error(f"API error: {e.response.text}")
        except Exception as e:
            st.error(str(e))

# --- Recommend ---
with tab2:
    st.subheader("Recommend")
    col1, col2, col3 = st.columns(3)
    tk = col1.text_input("Tickers", placeholder="NVDA, MSFT")
    sc = col2.text_input("Sources", placeholder="WSJ, FT")
    sec = col3.text_input("Sections", placeholder="Tech, Markets")
    days = st.slider("Days back", min_value=7, max_value=365, value=60, step=1)
    interest = st.text_input("Interest text (optional)", placeholder="AI hardware, enterprise cloud deals")
    krec = st.slider("Top-N", min_value=3, max_value=10, value=5, step=1)
    run_rec = st.button("Get Recommendations")

    if run_rec:
        payload = {
            "user": {
                "tickers": [t.strip().upper() for t in tk.split(",") if t.strip()] or None,
                "sources": [s.strip() for s in sc.split(",") if s.strip()] or None,
                "sections": [c.strip() for c in sec.split(",") if c.strip()] or None,
                "days_back": days
            },
            "interest_text": interest or None,
            "k": krec
        }
        try:
            res = _post("/recommend", payload)
            items = res.get("items", [])
            if not items:
                st.info("No recommendations.")
            else:
                for it in items:
                    st.markdown(f"**{it['title']}**  \n"
                                f"{it['source']}, {it['published_at'][:10]}  \n"
                                f"{it['reason']}  \n"
                                f"🔗 {it['url']}")
                    st.markdown("---")
        except requests.HTTPError as e:
            st.error(f"API error: {e.response.text}")
        except Exception as e:
            st.error(str(e))
