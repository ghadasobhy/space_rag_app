"""
ingest.py
=========
Data Ingestion & Preprocessing Layer.

Responsibilities:
    1. Normalize the heterogeneous raw records returned by `nasa_connectors`
       (JSON API responses, RSS items, HTML-scraped snippets) into standard
       `langchain_core.documents.Document` objects with rich metadata
       (source, category, timestamp, url).
    2. Split long documents into overlapping chunks using
       `RecursiveCharacterTextSplitter` (chunk_size=1000, chunk_overlap=150).
    3. Embed chunks with a local HuggingFace sentence-transformers model.
    4. Build (or update) a FAISS vector store, with optional on-disk
       persistence so the index survives across Streamlit reruns/restarts.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Step 1: Normalization
# --------------------------------------------------------------------------- #
def records_to_documents(
    records_by_source: Dict[str, List[Dict[str, Any]]]
) -> List[Document]:
    """
    Convert the `{source_name: [raw_record, ...]}` mapping produced by
    `nasa_connectors.fetch_all_sources` into a flat list of LangChain
    `Document` objects.

    Each raw record is expected to (at minimum) contain a "text" field.
    Missing optional fields are defaulted so downstream code never has to
    guard against KeyError.

    Parameters
    ----------
    records_by_source : Dict[str, List[Dict[str, Any]]]
        Output of `nasa_connectors.fetch_all_sources`.

    Returns
    -------
    List[Document]
        Flattened, normalized documents ready for splitting.
    """
    documents: List[Document] = []
    ingestion_timestamp = datetime.now(timezone.utc).isoformat()

    for source_name, records in records_by_source.items():
        for record in records:
            text = (record.get("text") or "").strip()
            title = (record.get("title") or "").strip()
            if not text and not title:
                # Nothing meaningful to embed — skip empty records.
                continue

            # Prefix the title into the page content so the embedding model
            # captures the headline context even when chunked mid-way.
            page_content = f"{title}\n\n{text}".strip() if title else text

            metadata = {
                "source": record.get("source", source_name),
                "category": record.get("category", "uncategorized"),
                "title": title or "Untitled",
                "url": record.get("url", ""),
                "date": record.get("date", ""),
                "media_type": record.get("media_type", "text"),
                "ingested_at": ingestion_timestamp,
            }
            documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info("Normalized %d raw records into %d Documents", 
                sum(len(v) for v in records_by_source.values()), len(documents))
    return documents


# --------------------------------------------------------------------------- #
# Step 2: Splitting
# --------------------------------------------------------------------------- #
def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks for more precise retrieval.

    Uses `RecursiveCharacterTextSplitter`, which tries progressively finer
    separators ("\\n\\n", "\\n", " ", "") so chunks break at natural text
    boundaries wherever possible while still respecting `chunk_size`.

    Parameters
    ----------
    documents : List[Document]
        Normalized documents from `records_to_documents`.
    chunk_size : int
        Maximum characters per chunk (default 1000 per spec).
    chunk_overlap : int
        Overlapping characters between consecutive chunks (default 150).

    Returns
    -------
    List[Document]
        Chunked documents; metadata is preserved and propagated to every
        chunk derived from the same source document.
    """
    if not documents:
        logger.warning("split_documents called with an empty document list")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


# --------------------------------------------------------------------------- #
# Step 3 & 4: Embeddings + FAISS vector store
# --------------------------------------------------------------------------- #
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> FastEmbedEmbeddings:
    """
    Instantiate a local ONNX-based embedding model via `fastembed`.

    Runs fully locally with no API calls, no torch, and no GPU required.
    The model is downloaded once and cached; subsequent runs use the cache.

    Returns
    -------
    FastEmbedEmbeddings
        Configured embedding model wrapper.
    """
    return FastEmbedEmbeddings(model_name=model_name)


def build_faiss_index(
    chunks: List[Document],
    embedding_model: Optional[FastEmbedEmbeddings] = None,
) -> FAISS:
    """
    Build a fresh in-memory FAISS vector store from document chunks.

    Parameters
    ----------
    chunks : List[Document]
        Output of `split_documents`.
    embedding_model : Optional[HuggingFaceEmbeddings]
        Pass an existing model instance to avoid reloading weights; a new
        one is created if omitted.

    Returns
    -------
    FAISS
        The populated vector store.

    Raises
    ------
    ValueError
        If `chunks` is empty — FAISS cannot build an index with no vectors.
    """
    if not chunks:
        raise ValueError("Cannot build a FAISS index from an empty chunk list.")

    embeddings = embedding_model or get_embedding_model()
    logger.info("Embedding %d chunks with %s", len(chunks), EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def persist_faiss_index(vector_store: FAISS, directory: str = FAISS_INDEX_DIR) -> None:
    """Save the FAISS index + docstore to disk for reuse across sessions."""
    os.makedirs(directory, exist_ok=True)
    vector_store.save_local(directory)
    logger.info("Persisted FAISS index to '%s'", directory)


def load_faiss_index(
    directory: str = FAISS_INDEX_DIR,
    embedding_model: Optional[FastEmbedEmbeddings] = None,
) -> Optional[FAISS]:
    """
    Load a previously persisted FAISS index from disk, if one exists.

    Returns
    -------
    Optional[FAISS]
        The loaded vector store, or None if no index is found on disk or
        loading fails for any reason (corrupted files, version mismatch).
    """
    if not os.path.isdir(directory) or not os.listdir(directory):
        logger.info("No persisted FAISS index found at '%s'", directory)
        return None

    embeddings = embedding_model or get_embedding_model()
    try:
        vector_store = FAISS.load_local(
            directory,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded persisted FAISS index from '%s'", directory)
        return vector_store
    except Exception as exc:
        logger.error("Failed to load FAISS index from '%s': %s", directory, exc)
        return None


def add_documents_to_index(
    vector_store: FAISS,
    new_chunks: List[Document],
) -> FAISS:
    """
    Incrementally add new chunks to an existing FAISS store (e.g. after a
    fresh data refresh) instead of rebuilding from scratch.

    Returns
    -------
    FAISS
        The same vector store instance, mutated in place, returned for
        convenience/chaining.
    """
    if not new_chunks:
        logger.info("add_documents_to_index called with no new chunks — no-op")
        return vector_store
    vector_store.add_documents(new_chunks)
    logger.info("Added %d new chunks to existing FAISS index", len(new_chunks))
    return vector_store


# --------------------------------------------------------------------------- #
# End-to-end convenience pipeline
# --------------------------------------------------------------------------- #
def run_ingestion_pipeline(
    records_by_source: Dict[str, List[Dict[str, Any]]],
    persist: bool = False,
    persist_directory: str = FAISS_INDEX_DIR,
) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline end-to-end: normalize -> split -> embed
    -> index (-> optionally persist).

    Parameters
    ----------
    records_by_source : Dict[str, List[Dict[str, Any]]]
        Raw records keyed by source, as returned by
        `nasa_connectors.fetch_all_sources`.
    persist : bool
        If True, save the resulting FAISS index to `persist_directory`.
    persist_directory : str
        Target directory for persistence.

    Returns
    -------
    Dict[str, Any]
        {
          "vector_store": FAISS instance,
          "documents": List[Document]  (pre-split, for the Data Feed Inspector),
          "chunks": List[Document]     (post-split, actually indexed),
          "stats": {source: chunk_count, ...}
        }
    """
    documents = records_to_documents(records_by_source)
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were produced from the provided records — "
            "all sources may have returned empty data."
        )

    vector_store = build_faiss_index(chunks)

    if persist:
        persist_faiss_index(vector_store, directory=persist_directory)

    stats: Dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        stats[source] = stats.get(source, 0) + 1

    return {
        "vector_store": vector_store,
        "documents": documents,
        "chunks": chunks,
        "stats": stats,
    }
