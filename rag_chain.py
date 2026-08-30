"""
rag_chain.py
============
RAG Pipeline & Retrieval Layer.

Wires together:
    - An MMR (Maximal Marginal Relevance) retriever over the FAISS store,
      configured with k=4 / fetch_k=10 to balance relevance and diversity
      and reduce redundant chunks from the same source.
    - A strict, domain-specific prompt template that forces the LLM to
      answer ONLY from retrieved NASA context and to cite the source of
      every claim (e.g. "[Source: APOD]", "[Source: NASA News]").
    - Chat memory: the last `memory_window` turns of conversation are
      injected into every prompt so the LLM can answer follow-up questions
      that reference earlier exchanges.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    RETRIEVER_LAMBDA_MULT,
    RETRIEVER_SEARCH_TYPE,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Strict, domain-specific system prompt
# --------------------------------------------------------------------------- #
NASA_SYSTEM_PROMPT = """\
You are the NASA Knowledge Assistant, a factual research aide grounded \
strictly in retrieved NASA data.

RULES YOU MUST FOLLOW:
1. Answer ONLY using information contained in the "Context" section below. \
Do not use outside knowledge, prior training data, or speculation of any kind.
2. If the Context does not contain enough information to answer the \
question, respond exactly with: \
"I don't have enough information from the retrieved NASA sources to answer that." \
Do not attempt to fill gaps with assumptions.
3. Every factual statement in your answer MUST be followed by an inline \
citation naming its exact source, using the format [Source: <source name>], \
e.g. [Source: APOD], [Source: Mars Rover Photos], [Source: NeoWs], \
[Source: data.nasa.gov], [Source: EarthData], [Source: NASA News]. \
Use the "source" field shown for each context chunk below — never invent one.
4. If different chunks disagree or come from different sources, cite each \
claim to its own source separately rather than merging them.
5. Be concise and precise. Prefer bullet points for multi-fact answers.
6. Never fabricate URLs, dates, or numbers that are not present in the Context.
7. You may refer to the conversation history below to understand follow-up \
questions, but your factual answers must still come exclusively from the Context.

Context:
{context}

Answer (with inline source citations):\
"""


def build_prompt_template() -> ChatPromptTemplate:
    """Construct the strict grounded-QA prompt as a ChatPromptTemplate."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", NASA_SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
def build_mmr_retriever(
    vector_store: FAISS,
    k: int = RETRIEVER_K,
    fetch_k: int = RETRIEVER_FETCH_K,
    lambda_mult: float = RETRIEVER_LAMBDA_MULT,
) -> VectorStoreRetriever:
    """
    Build an MMR retriever over the given FAISS vector store.

    MMR (Maximal Marginal Relevance) re-ranks the initial `fetch_k` candidate
    matches to select `k` results that are both relevant to the query AND
    mutually diverse — this prevents the same near-duplicate APOD paragraph,
    for example, from taking up all four context slots.

    Parameters
    ----------
    vector_store : FAISS
        The populated vector store to search.
    k : int
        Number of chunks to ultimately return (default 4 per spec).
    fetch_k : int
        Number of candidates initially fetched before MMR re-ranking
        (default 10 per spec).
    lambda_mult : float
        0 = max diversity, 1 = max relevance. 0.5 is a balanced default.

    Returns
    -------
    VectorStoreRetriever
        Configured retriever ready to plug into the RAG chain.
    """
    return vector_store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


# --------------------------------------------------------------------------- #
# LLM factory — supports Groq (free) and OpenAI
# --------------------------------------------------------------------------- #
def build_llm(
    api_key: str,
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    provider: str = DEFAULT_LLM_PROVIDER,
) -> Union[ChatGroq, ChatOpenAI]:
    """
    Instantiate the chat LLM used for answer generation.

    Supports Groq (free, fast LLaMA models) and OpenAI.
    Temperature defaults to 0.0 for maximal faithfulness to retrieved context.

    Parameters
    ----------
    api_key : str
        Groq API key (if provider='groq') or OpenAI API key (if provider='openai').
    provider : str
        'groq' or 'openai'.
    """
    if not api_key:
        raise ValueError(
            f"A {'Groq' if provider == 'groq' else 'OpenAI'} API key is required. "
            "Set it in the Streamlit sidebar or the corresponding env var."
        )
    if provider == "groq":
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )


# --------------------------------------------------------------------------- #
# Manual "stuff" chain (avoids extra langchain-classic dependency surface)
# --------------------------------------------------------------------------- #
def _format_context(documents: List[Document]) -> str:
    """
    Render retrieved chunks into a single context string, each chunk
    explicitly labeled with its source/category/date so the LLM has no
    ambiguity about which fact came from where.
    """
    formatted_blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        header = (
            f"[Chunk {i}] source={meta.get('source', 'unknown')} | "
            f"category={meta.get('category', 'unknown')} | "
            f"title={meta.get('title', 'Untitled')} | "
            f"date={meta.get('date', 'unknown')}"
        )
        formatted_blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_blocks)


# --------------------------------------------------------------------------- #
# Chat memory helper
# --------------------------------------------------------------------------- #
MEMORY_WINDOW = 6  # max number of messages (human + AI) kept in context


def _history_to_messages(
    history: ChatMessageHistory, window: int = MEMORY_WINDOW
) -> List[Any]:
    """
    Return the last `window` messages from `history` as LangChain message
    objects ready to be inserted between the system prompt and the human turn.
    Only the most recent `window` messages are included to keep the context
    window bounded.
    """
    return history.messages[-window:] if history.messages else []


class NasaRAGChain:
    """
    Thin orchestration wrapper around retriever + prompt + LLM.

    Exposed as a class (rather than a bare LCEL pipeline) so the Streamlit
    app can easily call `.invoke()` and separately inspect
    `.last_source_documents` for the Source Attribution panel, without
    re-running retrieval.

    Chat memory
    -----------
    A `ChatMessageHistory` instance is kept per chain. Every successful
    question/answer pair is appended so the LLM can resolve follow-up
    references like "tell me more about the second asteroid" without losing
    context. Call `.clear_memory()` to reset between sessions.
    """

    def __init__(
        self,
        vector_store: FAISS,
        api_key: str,
        llm_model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        provider: str = DEFAULT_LLM_PROVIDER,
        k: int = RETRIEVER_K,
        fetch_k: int = RETRIEVER_FETCH_K,
        memory_window: int = MEMORY_WINDOW,
    ) -> None:
        self.retriever = build_mmr_retriever(vector_store, k=k, fetch_k=fetch_k)
        self.llm = build_llm(api_key, model_name=llm_model, temperature=temperature, provider=provider)
        self.prompt = build_prompt_template()
        self.memory = ChatMessageHistory()
        self.memory_window = memory_window
        self.last_source_documents: List[Document] = []

    def clear_memory(self) -> None:
        """Reset the conversation history (e.g. when the user starts a new chat)."""
        self.memory.clear()
        logger.info("Chat memory cleared.")

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Run retrieval + grounded generation for a single question.

        The last `memory_window` messages from previous turns are injected
        between the system prompt and the current human message so the LLM
        can resolve follow-up questions.

        Returns
        -------
        Dict[str, Any]
            {
              "answer": str,
              "source_documents": List[Document],
              "sources_used": List[str]  (deduplicated source names)
            }
        """
        if not question or not question.strip():
            return {
                "answer": "Please enter a question about NASA data.",
                "source_documents": [],
                "sources_used": [],
            }

        try:
            retrieved_docs = self.retriever.invoke(question)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return {
                "answer": f"Retrieval error: {exc}",
                "source_documents": [],
                "sources_used": [],
            }

        self.last_source_documents = retrieved_docs

        if not retrieved_docs:
            return {
                "answer": (
                    "I don't have enough information from the retrieved "
                    "NASA sources to answer that."
                ),
                "source_documents": [],
                "sources_used": [],
            }

        context_str = _format_context(retrieved_docs)

        # Build message list: [system] + [history window] + [human]
        system_and_context = self.prompt.format_messages(
            context=context_str, input=question
        )
        history_msgs = _history_to_messages(self.memory, self.memory_window)
        # system_and_context = [SystemMessage, HumanMessage]
        # Insert history between system and the current human message.
        messages = (
            system_and_context[:1]          # SystemMessage with context
            + history_msgs                  # prior conversation turns
            + system_and_context[1:]        # current HumanMessage
        )

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return {
                "answer": f"Generation error: {exc}",
                "source_documents": retrieved_docs,
                "sources_used": sorted(
                    {d.metadata.get("source", "unknown") for d in retrieved_docs}
                ),
            }

        # Persist this turn in memory
        self.memory.add_message(HumanMessage(content=question))
        self.memory.add_message(AIMessage(content=answer_text))

        sources_used = sorted(
            {d.metadata.get("source", "unknown") for d in retrieved_docs}
        )
        return {
            "answer": answer_text,
            "source_documents": retrieved_docs,
            "sources_used": sources_used,
        }


def build_rag_chain(
    vector_store: FAISS,
    api_key: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    provider: str = DEFAULT_LLM_PROVIDER,
    k: int = RETRIEVER_K,
    fetch_k: int = RETRIEVER_FETCH_K,
    memory_window: int = MEMORY_WINDOW,
) -> NasaRAGChain:
    """Factory function to construct a ready-to-use `NasaRAGChain`."""
    return NasaRAGChain(
        vector_store=vector_store,
        api_key=api_key,
        llm_model=llm_model,
        temperature=temperature,
        provider=provider,
        k=k,
        fetch_k=fetch_k,
        memory_window=memory_window,
    )
