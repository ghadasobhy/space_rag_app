"""
config.py
=========
Centralized configuration for the NASA Knowledge Assistant.

Holds default constants (chunk sizes, model names, endpoint URLs, timeouts)
and a small helper for resolving the NASA API key from either an environment
variable or a value passed in at runtime (e.g. from the Streamlit sidebar).

Keeping this in one place means every other module imports settings from
here instead of hard-coding "magic values" all over the codebase.

.env support
------------
If a `.env` file exists in the project root it is loaded automatically on
first import via `python-dotenv`. This lets you store secrets locally without
touching environment variables or the Streamlit sidebar. The `.env` file is
listed in `.gitignore` so it is never committed.

Example `.env`:
    NASA_API_KEY=your_nasa_key_here
    OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

from dotenv import load_dotenv

# Load .env from the project root (silent if the file doesn't exist).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


# --------------------------------------------------------------------------- #
# NASA API endpoints
# --------------------------------------------------------------------------- #
NASA_APOD_URL: Final[str] = "https://api.nasa.gov/planetary/apod"
NASA_MARS_PHOTOS_URL: Final[str] = (
    "https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos"
)
NASA_NEOWS_URL: Final[str] = "https://api.nasa.gov/neo/rest/v1/feed"
DATA_NASA_GOV_URL: Final[str] = "https://data.nasa.gov/resource/y77d-th95.json"
# EarthData's CMR (Common Metadata Repository) search API is the public,
# key-free way to query collection/granule metadata.
EARTHDATA_CMR_URL: Final[str] = "https://cmr.earthdata.nasa.gov/search/collections.json"
NASA_NEWS_RSS_URL: Final[str] = "https://www.nasa.gov/feed/"
NASA_NEWS_HTML_URL: Final[str] = "https://www.nasa.gov/news/recent/"

DEMO_API_KEY: Final[str] = "DEMO_KEY"  # NASA's public rate-limited fallback key

# --------------------------------------------------------------------------- #
# Networking
# --------------------------------------------------------------------------- #
REQUEST_TIMEOUT_SECONDS: Final[int] = 15
MAX_RETRIES: Final[int] = 2

# --------------------------------------------------------------------------- #
# Ingestion / chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE: Final[int] = 1000
CHUNK_OVERLAP: Final[int] = 150

# --------------------------------------------------------------------------- #
# Embeddings / vector store
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-small-en-v1.5"
FAISS_INDEX_DIR: Final[str] = "faiss_index"

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
RETRIEVER_SEARCH_TYPE: Final[str] = "mmr"
RETRIEVER_K: Final[int] = 4
RETRIEVER_FETCH_K: Final[int] = 10
RETRIEVER_LAMBDA_MULT: Final[float] = 0.5  # diversity/relevance trade-off for MMR

# --------------------------------------------------------------------------- #
# LLM — Groq (free) as default, OpenAI as fallback
# --------------------------------------------------------------------------- #
DEFAULT_LLM_PROVIDER: Final[str] = "groq"   # "groq" or "openai"
DEFAULT_LLM_MODEL: Final[str] = "llama-3.1-8b-instant"   # Groq model (replaces deprecated llama3-8b-8192)
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.0


@dataclass
class Settings:
    """
    Runtime-resolved settings bundle.

    Values fall back to environment variables when not explicitly supplied,
    which lets the Streamlit UI override keys per-session without requiring
    a restart or a .env edit.
    """

    nasa_api_key: str = field(
        default_factory=lambda: os.getenv("NASA_API_KEY", DEMO_API_KEY)
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )

    def with_overrides(
        self,
        nasa_api_key: str | None = None,
        openai_api_key: str | None = None,
        groq_api_key: str | None = None,
    ) -> "Settings":
        """Return a new Settings object with any provided values overridden."""
        return Settings(
            nasa_api_key=nasa_api_key.strip() if nasa_api_key else self.nasa_api_key,
            openai_api_key=(
                openai_api_key.strip() if openai_api_key else self.openai_api_key
            ),
            groq_api_key=groq_api_key.strip() if groq_api_key else self.groq_api_key,
        )


def get_settings() -> Settings:
    """Factory returning a fresh Settings instance resolved from the environment."""
    return Settings()
