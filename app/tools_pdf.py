"""PDF retrieval tool backed by a FAISS index."""

from typing import List
from textwrap import shorten

from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from config import INDEX_DIR, EMBED_MODEL


MAX_CHARS_PER_CHUNK = 600
DEFAULT_K = 4

_VECTOR_STORE: FAISS | None = None


def _get_vector_store() -> FAISS:
    """Return the cached FAISS index, loading it from disk on the first call."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        embed = OllamaEmbeddings(model=EMBED_MODEL)
        _VECTOR_STORE = FAISS.load_local(
            str(INDEX_DIR), embed, allow_dangerous_deserialization=True
        )
    return _VECTOR_STORE


def _format_hits(docs, k: int) -> str:
    """Build a compact, model-friendly context block with page refs."""
    blocks: List[str] = []
    for i, d in enumerate(docs[:k], 1):
        page = d.metadata.get("page", "?")
        text = " ".join(d.page_content.split())
        text = shorten(text, width=MAX_CHARS_PER_CHUNK, placeholder=" …")
        blocks.append(f"[p.{page}] {text}")
    return "\n\n".join(blocks) if blocks else ""


@tool
def retrieve_context(query: str, k: int = DEFAULT_K) -> str:
    """
    Retrieve the most relevant snippets from the indexed PDF.
    Args:
        query: natural-language question
        k: number of chunks to return (default = 4)
    Returns:
        A compact context string with page references,
        or 'NO_MATCH' if nothing relevant is found.
    """
    vector_store = _get_vector_store()
    docs = vector_store.similarity_search(query, k)
    context_text = _format_hits(docs, k)
    return context_text if context_text else "NO_MATCH"
