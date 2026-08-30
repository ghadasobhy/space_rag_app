# 🚀 Multi-Source Real-Time NASA Knowledge Assistant (RAG)

A production-ready Retrieval-Augmented Generation system that answers questions
grounded strictly in live NASA data pulled from four sources:

- **api.nasa.gov** — APOD, Mars Rover Photos, NeoWs (near-Earth asteroids)
- **data.nasa.gov** — open dataset metadata
- **earthdata.nasa.gov** — Earth science / climate collection metadata (via CMR)
- **nasa.gov/news** — live news via RSS (feedparser) with an HTML scrape (BeautifulSoup) fallback

## Architecture

```
nasa_connectors.py  →  ingest.py            →  rag_chain.py        →  app.py
(fetch raw records)    (normalize → split →     (MMR retriever +       (Streamlit
                         embed → FAISS index)     strict prompt +        3-tab UI)
                                                   LLM answer)
```

| Layer | File | Key details |
|---|---|---|
| Data Sources | `nasa_connectors.py` | `requests` + retry/backoff, `feedparser`, `BeautifulSoup` |
| Ingestion | `ingest.py` | `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=150) |
| Vector Store | `ingest.py` | HuggingFace `all-MiniLM-L6-v2` embeddings + FAISS |
| Retrieval | `rag_chain.py` | MMR retriever, `k=4`, `fetch_k=10` |
| Generation | `rag_chain.py` | Strict grounded prompt, mandatory `[Source: ...]` citations |
| UI | `app.py` | Streamlit — Chat / Data Feed Inspector / Vector DB Status tabs |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## API Keys

You need two keys, both entered in the Streamlit sidebar (or set as environment variables):

- `NASA_API_KEY` — free at https://api.nasa.gov (defaults to the shared, rate-limited `DEMO_KEY` if omitted)
- `OPENAI_API_KEY` — required to generate answers (retrieval/indexing works without it)

```bash
export NASA_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
```

## Run

```bash
streamlit run app.py
```

1. Open the sidebar, confirm/enter your API keys and fetch parameters.
2. Click **"🔄 Fetch & Index NASA Data"** — this pulls from all 4 sources, chunks, embeds, and builds a FAISS index (progress shown live).
3. Use the **Chat Assistant** tab to ask questions — every answer cites its exact NASA source.
4. Use **Data Feed Inspector** to see the raw records behind the index.
5. Use **Vector DB Status** to see chunk counts per source and retrieval configuration.

## Notes on robustness

- Every network call in `nasa_connectors.py` is wrapped in `try/except` with retry + backoff for timeouts, connection errors, and HTTP 429 rate limiting. A single failing source returns `[]` rather than crashing the whole fetch.
- NASA news prefers the RSS feed; if that returns no entries, it automatically falls back to an HTML scrape.
- The FAISS index can optionally be persisted to disk (`ingest.persist_faiss_index` / `load_faiss_index`) to avoid re-embedding on every app restart — `run_ingestion_pipeline(..., persist=True)`.
- The LLM defaults to `gpt-4o-mini` at `temperature=0.0` for maximal faithfulness to retrieved context; change `DEFAULT_LLM_MODEL` in `config.py` if you'd prefer a different OpenAI model.

## Extending

- Swap the LLM provider by replacing `build_llm` in `rag_chain.py` (e.g. Anthropic, local Ollama).
- Add more NASA endpoints (e.g. EPIC, DONKI space weather) by adding a new `fetch_*` function to `nasa_connectors.py` and wiring it into `fetch_all_sources`.
- Persist the FAISS index and load it on startup to skip re-fetching/re-embedding every session.
