import os
import requests
import traceback
import streamlit as st

# -----------------------------
# Config & helpers
# -----------------------------
DEFAULT_API = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG-Powered Tech & Finance News", layout="wide")
st.title("📰 RAG-Powered Tech & Finance News")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    api_override = st.text_input("API Base URL", value=DEFAULT_API, help="e.g., http://localhost:8000")
    API = api_override or DEFAULT_API

    col_s1, col_s2 = st.columns([1,1])
    ping = col_s1.button("Ping API")
    refresh_catalog = col_s2.button("Refresh options")
    st.caption("Tip: ensure the FastAPI server is running.")

# HTTP helpers

def _post(path: str, payload: dict, timeout: int = 90):
    url = f"{API}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(path: str, params: dict | None = None, timeout: int = 30):
    url = f"{API}{path}"
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def load_catalog(api_base: str) -> dict:
    """Fetch available tickers/sources/sections from the server.
    Expected FastAPI endpoint: GET /meta/catalog -> { tickers: [], sources: [], sections: [] }
    The cache key depends on the API URL.
    """
    try:
        data = requests.get(f"{api_base}/meta/catalog", timeout=20).json()
        # Ensure keys exist
        data.setdefault("tickers", [])
        data.setdefault("sources", [])
        data.setdefault("sections", [])
        # Normalize
        data["tickers"] = sorted({t.upper() for t in data["tickers"] if isinstance(t, str) and t.strip()})
        data["sources"] = sorted({s.strip() for s in data["sources"] if isinstance(s, str) and s.strip()})
        data["sections"] = sorted({c.strip() for c in data["sections"] if isinstance(c, str) and c.strip()})
        return data
    except Exception:
        # Fallback options (if API not ready). You can remove/expand these.
        return {
            "tickers": ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"],
            "sources": ["WSJ", "FT", "Bloomberg", "Reuters", "TechCrunch", "Wired"],
            "sections": ["Tech", "Markets", "Startups", "AI", "Policy"],
        }


# Small utilities

def _csv_to_list(s: str, upper: bool = False):
    if not s:
        return []
    vals = [x.strip() for x in s.split(",") if x and x.strip()]
    if upper:
        vals = [x.upper() for x in vals]
    return vals


def _merge_unique(*lists):
    seen = set()
    out = []
    for lst in lists:
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


# Ping API (optional)
if ping:
    try:
        res = _get("/health")
        st.sidebar.success(f"✅ API OK: {res}")
    except Exception as e:
        st.sidebar.error(f"❌ API not reachable: {e}")

# Catalog (option lists)
if refresh_catalog:
    load_catalog.clear()
CATALOG = load_catalog(API)

# Tabs
tab_help, tab1, tab2, tab3 = st.tabs(["❓ How to use", "📝 Summarize", "⭐ Recommend", "🤖 Ask"])

# -----------------------------
# ❓ How to use
# -----------------------------
with tab_help:
    st.subheader("Welcome 👋")
    st.markdown(
        """
        This app helps you **summarize** and **recommend** tech & finance news using a RAG (Retrieval-Augmented Generation) pipeline.

        **Quick Start**
        1. Go to **📝 Summarize** to create concise, cited bullets from a topic or a specific article URL.
        2. Go to **⭐ Recommend** to get a curated list of recent articles tailored by tickers, sources, and sections.
        3. Use the **Settings** sidebar to point the app at your FastAPI server.
        """
    )

    st.markdown("---")
    st.markdown("### What the controls mean")
    st.markdown(
        """
        **Summarize**
        - **Mode**: `topic` retrieves relevant articles based on your query; `article` summarizes the single URL.
        - **Topic query**: Natural language topic (e.g., *NVDA earnings and AI demand outlook*).
        - **Article URL**: Full URL if using `article` mode.
        - **Tickers / Sources**: Filter retrieval by companies (tickers) or publishers.
        - **Days back**: Only consider articles from the last N days.
        - **k**: Final number of chunks (snippets) fed into the summarizer after re-ranking.

        **Recommend**
        - **Tickers / Sources / Sections**: Guide the recommender to your interests.
        - **Days back**: Only consider articles published within this many days.
        - **Interest text**: Optional free-text preferences (e.g., *AI hardware, enterprise cloud deals*).
        - **Top-N**: Number of recommended articles to return.
        """
    )

    st.markdown("---")
    st.markdown("### Tips")
    st.markdown(
        """
        - If you don't see options for tickers/sources/sections, click **Refresh options** in the sidebar.
        - You can always add **extras** manually; they will be merged with your selections.
        - For best results in `topic` mode, include a time context (e.g., *last quarter*, *since July*).
        - If the API is slow, check your server logs and ensure the vector DB is warm. Caching helps!  
        """
    )

# -----------------------------
# 📝 Summarize
# -----------------------------
with tab1:
    st.subheader("Summarize")
    mode = st.radio("Mode", options=["topic", "article"], horizontal=True)

    colA, colB = st.columns(2)
    query = colA.text_input("Topic query", placeholder="e.g., NVDA earnings and AI demand outlook")
    article_url = colB.text_input("Article URL (article mode)")

    st.markdown("**Filters**")
    fc1, fc2, fc3 = st.columns([1,1,1])

    # Multiselects with server-provided options + free-form extras
    sel_tickers = fc1.multiselect(
    "Tickers (suggested)", options=CATALOG.get("tickers", []),
    help="Type to search and select.", key="sum_tickers"
    )
    extra_tickers = fc1.text_input("Extra tickers (comma-separated)", key="sum_extra_tickers")

    sel_sources = fc2.multiselect(
        "Sources (suggested)", options=CATALOG.get("sources", []),
        help="Type to search and select.", key="sum_sources"
    )
    extra_sources = fc2.text_input("Extra sources (comma-separated)", key="sum_extra_sources")

    # Days back filter
    days_back = fc3.slider(
        "Days back",
        min_value=7,
        max_value=365,
        value=60,
        step=1,
        key="sum_days_back",
    )

    # Core knobs
    k = st.slider("k (final #chunks)", min_value=3, max_value=10, value=8, step=1)

    run_sum = st.button("Run Summarization", type="primary")

    if run_sum:
        try:
            tickers = _merge_unique([t.upper() for t in sel_tickers], _csv_to_list(extra_tickers, upper=True)) or None
            sources = _merge_unique(sel_sources, _csv_to_list(extra_sources)) or None

            payload = {
                "mode": mode,
                "query": query if mode == "topic" else None,
                "article_url": article_url if mode == "article" else None,
                "k": int(k),
                "days_back": int(days_back),
                "tickers": tickers,
                "sources": sources,
            }

            res = _post("/summarize", payload)
            bullets = res.get("bullets", [])

            if not bullets:
                st.info("No bullets returned. Try adjusting filters or query.")
            else:
                st.markdown("### Summary")
                for b in bullets:
                    url = b.get("url")
                    text = b.get("text", "")
                    citation = b.get("citation", "")
                    if url:
                        st.markdown(f"- {text}  \n  🔗 {url}")
                    elif citation:
                        st.markdown(f"- {text} {citation}")
                    else:
                        st.markdown(f"- {text}")
        except requests.HTTPError as e:
            st.error(f"API error: {getattr(e, 'response', None) and e.response.text}")
        except Exception:
            st.error("An unexpected error occurred. See details below.")
            st.exception(traceback.format_exc())

# -----------------------------
# ⭐ Recommend
# -----------------------------
with tab2:
    st.subheader("Recommend")

    c1, c2, c3 = st.columns(3)
    sel_tk = c1.multiselect("Tickers (suggested)", options=CATALOG.get("tickers", []), key="rec_tickers")
    extra_tk = c1.text_input("Extra tickers (comma-separated)", key="rec_extra_tickers")

    sel_sc = c2.multiselect("Sources (suggested)", options=CATALOG.get("sources", []), key="rec_sources")
    extra_sc = c2.text_input("Extra sources (comma-separated)", key="rec_extra_sources")

    sel_sec = c3.multiselect("Sections (suggested)", options=CATALOG.get("sections", []), key="rec_sections")
    extra_sec = c3.text_input("Extra sections (comma-separated)", key="rec_extra_sections")


    days = st.slider(
        "Days back",
        min_value=7,
        max_value=365,
        value=60,
        step=1,
        key="rec_days_back",
    )
    interest = st.text_input("Interest text (optional)", placeholder="AI hardware, enterprise cloud deals")
    krec = st.slider("Top-N", min_value=3, max_value=10, value=5, step=1)

    run_rec = st.button("Get Recommendations", type="primary")

    if run_rec:
        try:
            tickers = _merge_unique([t.upper() for t in sel_tk], _csv_to_list(extra_tk, upper=True)) or None
            sources = _merge_unique(sel_sc, _csv_to_list(extra_sc)) or None
            sections = _merge_unique(sel_sec, _csv_to_list(extra_sec)) or None

            payload = {
                "user": {
                    "tickers": tickers,
                    "sources": sources,
                    "sections": sections,
                    "days_back": int(days),
                },
                "interest_text": interest or None,
                "k": int(krec),
            }

            res = _post("/recommend", payload)
            items = res.get("items", [])
            if not items:
                st.info("No recommendations.")
            else:
                for it in items:
                    title = it.get("title", "(no title)")
                    source = it.get("source", "?")
                    published = (it.get("published_at") or "")[:10]
                    reason = it.get("reason", "")
                    url = it.get("url", "")

                    st.markdown(
                        f"**{title}**  \n"
                        f"{source}, {published}  \n"
                        f"{reason}  \n"
                        f"🔗 {url}"
                    )
                    st.markdown("---")
        except requests.HTTPError as e:
            st.error(f"API error: {getattr(e, 'response', None) and e.response.text}")
        except Exception:
            st.error("An unexpected error occurred. See details below.")
            st.exception(traceback.format_exc())

# -----------------------------
# 🤖 Ask
# -----------------------------
with tab3:
    st.subheader("Ask a question about the news")
    question = st.text_area(
        "Question",
        placeholder="What has Apple announced about generative AI this quarter?",
        help="Ask about trends, companies, or themes. The assistant will cite supporting events.",
    )

    col_t1, col_t2 = st.columns(2)
    temporal_hint = col_t1.text_input(
        "Natural-language time hint",
        placeholder="since July",
        help="Optional phrase like 'since July' or 'last quarter'.",
    )
    days_back = col_t2.slider(
        "Fallback days back",
        min_value=7,
        max_value=365,
        value=90,
        step=1,
        help="Used when no explicit dates are detected.",
    )

    col_d1, col_d2 = st.columns(2)
    start_override = col_d1.text_input(
        "Start date override (YYYY-MM-DD)",
        help="Optional explicit start date.",
    )
    end_override = col_d2.text_input(
        "End date override (YYYY-MM-DD)",
        help="Optional explicit end date.",
    )

    st.markdown("**Filters**")
    fc1, fc2 = st.columns(2)
    ask_tickers = fc1.multiselect(
        "Tickers",
        options=CATALOG.get("tickers", []),
        help="Filter by tickers (optional).",
    )
    ask_extra_tickers = fc1.text_input("Extra tickers (comma-separated)")
    ask_sources = fc2.multiselect(
        "Sources",
        options=CATALOG.get("sources", []),
        help="Filter by publisher (optional).",
    )
    ask_extra_sources = fc2.text_input("Extra sources (comma-separated)")

    col_opt1, col_opt2 = st.columns(2)
    max_events = col_opt1.slider(
        "Max timeline events",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
    )
    context_k = col_opt2.slider(
        "Context chunks",
        min_value=6,
        max_value=40,
        value=18,
        step=2,
        help="Number of retrieved chunks to analyze before selecting events.",
    )

    run_qa = st.button("Ask the agent", type="primary")

    if run_qa:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                tickers = _merge_unique([t.upper() for t in ask_tickers], _csv_to_list(ask_extra_tickers, upper=True)) or None
                sources = _merge_unique(ask_sources, _csv_to_list(ask_extra_sources)) or None

                temporal_filter = {}
                if temporal_hint:
                    temporal_filter["natural_language"] = temporal_hint
                if start_override:
                    temporal_filter["start_date"] = start_override
                if end_override:
                    temporal_filter["end_date"] = end_override
                temporal_filter["days_back"] = int(days_back)
                if not any(k in temporal_filter for k in ("natural_language", "start_date", "end_date")):
                    temporal_filter = {"days_back": int(days_back)}

                payload = {
                    "question": question,
                    "days_back": int(days_back),
                    "tickers": tickers,
                    "sources": sources,
                    "max_events": int(max_events),
                    "context_k": int(context_k),
                    "temporal_filter": temporal_filter,
                }

                res = _post("/qa", payload, timeout=120)
                answer = res.get("answer", "")
                timeline = res.get("timeline", [])
                citations = res.get("citations", [])

                if answer:
                    st.markdown("### Answer")
                    st.markdown(answer)
                else:
                    st.info("No narrative answer returned.")

                if citations:
                    st.markdown("**Citations:** " + ", ".join(citations))

                if timeline:
                    st.markdown("### Timeline")
                    for item in timeline:
                        date = item.get("date") or "Unknown"
                        summary = item.get("summary", "")
                        citation = item.get("citation") or ""
                        source = item.get("source") or ""
                        url = item.get("url")
                        parts = [f"**{date}**", summary]
                        if citation:
                            parts.append(citation)
                        if source:
                            parts.append(f"_{source}_")
                        st.markdown(" - " + " — ".join([p for p in parts if p]))
                        if url:
                            st.markdown(f"   🔗 {url}")
                else:
                    st.info("No supporting events were found.")
            except requests.HTTPError as e:
                st.error(f"API error: {getattr(e, 'response', None) and e.response.text}")
            except Exception:
                st.error("An unexpected error occurred. See details below.")
                st.exception(traceback.format_exc())
