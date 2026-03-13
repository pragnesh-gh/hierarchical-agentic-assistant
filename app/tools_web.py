"""Web search tool wrapper for Tavily with normalized output."""

from typing import Any, Dict, List

from langchain.tools import tool
from langchain_tavily import TavilySearch


DEFAULT_MAX_RESULTS = 4


def _normalize_results(raw: Any) -> str:
    """Normalize Tavily results into a compact, model-friendly string."""
    if isinstance(raw, str):
        return raw

    if isinstance(raw, dict):
        results = raw.get("results", [])
    elif isinstance(raw, list):
        results = raw
    else:
        return str(raw)

    blocks: List[str] = []
    for item in results:
        if not isinstance(item, dict):
            blocks.append(str(item))
            continue

        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip()
        summary = content[:300]

        line_parts = []
        if title:
            line_parts.append(title)
        if url:
            line_parts.append(url)
        if summary:
            line_parts.append(summary)
        blocks.append(" | ".join(line_parts))

    return "\n".join(blocks)


@tool
def tavily_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """Search the web via Tavily and return normalized results."""
    tavily = TavilySearch(max_results=max_results, topic="general")
    try:
        raw = tavily.invoke(query)
    except Exception as exc:
        return f"WEB_SEARCH_ERROR: {exc}"
    return _normalize_results(raw)
