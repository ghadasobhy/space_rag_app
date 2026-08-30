"""
app.py
======
Streamlit frontend for the Multi-Source Real-Time NASA Knowledge Assistant.

Tabs:
    1. RAG Chat Assistant       — conversational Q&A grounded in NASA data.
    2. Data Feed Inspector      — raw fetched records from each source.
    3. Vector DB & Attribution  — index stats, chunk counts, source metrics.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import streamlit as st

from config import get_settings
from ingest import run_ingestion_pipeline
from nasa_connectors import fetch_all_sources
from rag_chain import build_rag_chain

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="NASA Knowledge Assistant",
    page_icon="🚀",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
def _init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "raw_records": {},       # Dict[str, List[Dict]] — last fetch, per source
        "vector_store": None,    # FAISS instance
        "rag_chain": None,       # NasaRAGChain instance
        "chunk_stats": {},       # Dict[str, int] — chunks per source
        "num_documents": 0,
        "num_chunks": 0,
        "chat_history": [],      # List[Dict[str, str]] — {"role", "content", "sources"}
        "last_refresh_ts": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# --------------------------------------------------------------------------- #
# Sidebar — API key management & data refresh controls
# --------------------------------------------------------------------------- #
def render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("🔑 Configuration")

    settings = get_settings()

    # Fields are intentionally left blank — keys are loaded from Streamlit
    # Secrets / environment variables automatically via config.get_settings().
    # Entering a value here overrides the env key for this session only.
    nasa_key_input = st.sidebar.text_input(
        "NASA API Key",
        value="",
        placeholder="Leave blank to use secret / env var",
        type="password",
        help="Get a free key at https://api.nasa.gov. Leave blank to use the "
        "key already set in Streamlit Secrets or your .env file.",
    )
    groq_key_input = st.sidebar.text_input(
        "Groq API Key (free)",
        value="",
        placeholder="Leave blank to use secret / env var",
        type="password",
        help="Free LLM via Groq — get a key at https://console.groq.com. "
        "Leave blank to use the key set in Streamlit Secrets.",
    )
    openai_key_input = st.sidebar.text_input(
        "OpenAI API Key (optional fallback)",
        value="",
        placeholder="Leave blank to use secret / env var",
        type="password",
        help="Optional — only used if no Groq key is available. "
        "Leave blank to use the key set in Streamlit Secrets.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("📡 Data Source Parameters")
    apod_days = st.sidebar.slider("APOD — days back", 1, 10, 5)
    mars_sol = st.sidebar.number_input("Mars Rover — sol (day)", min_value=0, value=1000, step=10)
    neows_days = st.sidebar.slider("NeoWs — days ahead", 1, 7, 3)
    news_items = st.sidebar.slider("NASA News — max items", 3, 20, 10)
    earthdata_keyword = st.sidebar.text_input("EarthData — keyword", value="climate")

    st.sidebar.divider()
    refresh_clicked = st.sidebar.button("🔄 Fetch & Index NASA Data", type="primary")

    st.sidebar.divider()
    st.sidebar.subheader("💬 Chat Memory")
    chain = st.session_state.get("rag_chain")
    mem_count = len(chain.memory.messages) if chain else 0
    st.sidebar.caption(f"Stored messages: {mem_count}")
    clear_memory_clicked = st.sidebar.button(
        "🗑️ Clear Chat Memory",
        disabled=(chain is None or mem_count == 0),
        help="Clears the conversation history so the next question starts fresh.",
    )
    if clear_memory_clicked and chain is not None:
        chain.clear_memory()
        st.session_state["chat_history"] = []
        st.sidebar.success("Memory cleared.")

    st.sidebar.divider()
    st.sidebar.caption(
        "Built with LangChain, FAISS, HuggingFace embeddings, and Streamlit."
    )

    resolved_groq = groq_key_input.strip() if groq_key_input.strip() else settings.groq_api_key
    resolved_openai = openai_key_input.strip() if openai_key_input.strip() else settings.openai_api_key

    # Pick the active LLM key: prefer Groq, fall back to OpenAI.
    if resolved_groq:
        active_llm_key = resolved_groq
        active_llm_provider = "groq"
    else:
        active_llm_key = resolved_openai
        active_llm_provider = "openai"

    return {
        "nasa_api_key": nasa_key_input.strip() if nasa_key_input.strip() else settings.nasa_api_key,
        "openai_api_key": resolved_openai,
        "groq_api_key": resolved_groq,
        "llm_api_key": active_llm_key,
        "llm_provider": active_llm_provider,
        "apod_days": apod_days,
        "mars_sol": mars_sol,
        "neows_days": neows_days,
        "news_items": news_items,
        "earthdata_keyword": earthdata_keyword,
        "refresh_clicked": refresh_clicked,
    }


# --------------------------------------------------------------------------- #
# Data refresh pipeline (fetch -> ingest -> index)
# --------------------------------------------------------------------------- #
def run_refresh(config: Dict[str, Any]) -> None:
    with st.spinner("Fetching from NASA sources..."):
        try:
            raw_records = fetch_all_sources(
                api_key=config["nasa_api_key"],
                apod_days=config["apod_days"],
                mars_sol=config["mars_sol"],
                neows_days=config["neows_days"],
                earthdata_keyword=config["earthdata_keyword"],
                news_items=config["news_items"],
            )
        except Exception as exc:
            st.error(f"Failed to fetch NASA data: {exc}")
            return

    total_records = sum(len(v) for v in raw_records.values())
    if total_records == 0:
        st.warning(
            "No records were retrieved from any source. Check your NASA API "
            "key and network connectivity, then try again."
        )
        st.session_state["raw_records"] = raw_records
        return

    with st.spinner(f"Ingesting {total_records} records and building FAISS index..."):
        try:
            pipeline_result = run_ingestion_pipeline(raw_records, persist=False)
        except Exception as exc:
            st.error(f"Ingestion/indexing failed: {exc}")
            return

    st.session_state["raw_records"] = raw_records
    st.session_state["vector_store"] = pipeline_result["vector_store"]
    st.session_state["chunk_stats"] = pipeline_result["stats"]
    st.session_state["num_documents"] = len(pipeline_result["documents"])
    st.session_state["num_chunks"] = len(pipeline_result["chunks"])
    st.session_state["last_refresh_ts"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Rebuild the RAG chain against the fresh index if an LLM key is set.
    if config["llm_api_key"]:
        try:
            st.session_state["rag_chain"] = build_rag_chain(
                vector_store=pipeline_result["vector_store"],
                api_key=config["llm_api_key"],
                provider=config["llm_provider"],
            )
        except Exception as exc:
            st.warning(f"Index built, but RAG chain setup failed: {exc}")
            st.session_state["rag_chain"] = None
    else:
        st.session_state["rag_chain"] = None

    st.success(
        f"Indexed {st.session_state['num_chunks']} chunks from "
        f"{st.session_state['num_documents']} documents across "
        f"{len(raw_records)} sources."
    )


# --------------------------------------------------------------------------- #
# Tab 1: RAG Chat Assistant
# --------------------------------------------------------------------------- #
def render_chat_tab(config: Dict[str, Any]) -> None:
    st.header("💬 NASA RAG Chat Assistant")
    st.caption(
        "Answers are grounded strictly in retrieved NASA data and cite the "
        "exact source of every claim."
    )

    if st.session_state["vector_store"] is None:
        st.info(
            "👈 Use the sidebar to fetch and index NASA data before chatting."
        )
        return

    if not config["llm_api_key"]:
        st.warning(
            "Enter a Groq API key (free) in the sidebar to enable answer generation. "
            "Get one at https://console.groq.com — takes 1 minute."
        )
        return

    if st.session_state["rag_chain"] is None:
        try:
            st.session_state["rag_chain"] = build_rag_chain(
                vector_store=st.session_state["vector_store"],
                api_key=config["llm_api_key"],
                provider=config["llm_provider"],
            )
        except Exception as exc:
            st.error(f"Could not initialize the RAG chain: {exc}")
            return

    # Render chat history
    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("📎 Sources used"):
                    for src in turn["sources"]:
                        st.markdown(f"- {src}")

    user_question = st.chat_input(
        "Ask about astronomy pictures, Mars photos, asteroids, climate data, or NASA news..."
    )
    if not user_question:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating a grounded answer..."):
            try:
                result = st.session_state["rag_chain"].invoke(user_question)
            except Exception as exc:
                result = {
                    "answer": f"An error occurred while answering: {exc}",
                    "source_documents": [],
                    "sources_used": [],
                }
        st.markdown(result["answer"])
        if result["sources_used"]:
            with st.expander("📎 Sources used"):
                for src in result["sources_used"]:
                    st.markdown(f"- {src}")
                st.divider()
                st.caption("Retrieved chunks (MMR, k=4, fetch_k=10):")
                for doc in result["source_documents"]:
                    meta = doc.metadata
                    st.markdown(
                        f"**{meta.get('title', 'Untitled')}** "
                        f"— *{meta.get('source', 'unknown')}* "
                        f"({meta.get('date', 'unknown date')})"
                    )
                    st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                    if meta.get("url"):
                        st.markdown(f"[View source]({meta['url']})")
                    st.markdown("---")

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources_used"],
        }
    )


# --------------------------------------------------------------------------- #
# Tab 2: Real-Time Data Feed Inspector
# --------------------------------------------------------------------------- #
def render_data_feed_tab() -> None:
    st.header("📡 Real-Time Data Feed Inspector")
    st.caption("Raw records as fetched directly from each NASA source, before chunking/embedding.")

    raw_records: Dict[str, List[Dict[str, Any]]] = st.session_state["raw_records"]
    if not raw_records:
        st.info("👈 Fetch data from the sidebar to populate this view.")
        return

    for source_name, records in raw_records.items():
        with st.expander(f"{source_name} — {len(records)} record(s)", expanded=False):
            if not records:
                st.warning("No records returned from this source on the last fetch.")
                continue
            for record in records:
                st.markdown(f"**{record.get('title', 'Untitled')}**")
                cols = st.columns([1, 1, 1])
                cols[0].caption(f"Category: {record.get('category', 'n/a')}")
                cols[1].caption(f"Date: {record.get('date', 'n/a')}")
                cols[2].caption(f"Media: {record.get('media_type', 'n/a')}")
                text_preview = record.get("text", "")
                st.write(text_preview[:400] + ("..." if len(text_preview) > 400 else ""))
                if record.get("url"):
                    st.markdown(f"[Open source ↗]({record['url']})")
                st.divider()


# --------------------------------------------------------------------------- #
# Tab 3: Vector DB Status & Source Attribution
# --------------------------------------------------------------------------- #
def render_vector_db_tab() -> None:
    st.header("🗂️ Vector DB Status & Source Attribution")

    if st.session_state["vector_store"] is None:
        st.info("👈 Fetch and index NASA data from the sidebar to see index metrics.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents ingested", st.session_state["num_documents"])
    col2.metric("Chunks indexed", st.session_state["num_chunks"])
    col3.metric("Active sources", len(st.session_state["chunk_stats"]))

    st.caption(f"Last refreshed: {st.session_state['last_refresh_ts'] or 'never'}")

    st.subheader("Chunks per source")
    stats = st.session_state["chunk_stats"]
    if stats:
        st.bar_chart(stats)
        for source, count in sorted(stats.items(), key=lambda kv: kv[1], reverse=True):
            st.markdown(f"- **{source}**: {count} chunk(s)")
    else:
        st.warning("No chunk statistics available yet.")

    st.subheader("Retrieval configuration")
    st.markdown(
        "- **Search type:** MMR (Maximal Marginal Relevance)\n"
        "- **k (results returned):** 4\n"
        "- **fetch_k (candidates considered):** 10\n"
        "- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`\n"
        "- **Chunking:** RecursiveCharacterTextSplitter "
        "(chunk_size=1000, chunk_overlap=150)"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🚀 Multi-Source Real-Time NASA Knowledge Assistant")
    st.caption(
        "RAG over APOD, Mars Rover Photos, NeoWs, data.nasa.gov, EarthData, "
        "and live NASA News — powered by LangChain + FAISS."
    )

    config = render_sidebar()

    if config["refresh_clicked"]:
        run_refresh(config)

    tab_chat, tab_feed, tab_vector_db = st.tabs(
        ["💬 Chat Assistant", "📡 Data Feed Inspector", "🗂️ Vector DB Status"]
    )

    with tab_chat:
        render_chat_tab(config)
    with tab_feed:
        render_data_feed_tab()
    with tab_vector_db:
        render_vector_db_tab()


if __name__ == "__main__":
    main()
