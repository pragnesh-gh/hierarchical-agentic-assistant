"""Lightweight chat-intel helpers for topic shift and UX prompts."""

from __future__ import annotations

import re
from typing import Any, List


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
    "you",
    "your",
}

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_CONTINUITY_MARKERS = (
    "this",
    "that",
    "same",
    "it",
    "also",
    "follow up",
    "continue",
    "as discussed",
    "send this",
    "what we discussed",
)
_NEW_TOPIC_MARKERS = (
    "new topic",
    "different topic",
    "new question",
    "something else",
    "switch topics",
)


def _tokenize(text: str) -> List[str]:
    lowered = (text or "").lower()
    out: List[str] = []
    for tok in _TOKEN_RE.findall(lowered):
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _latest_human(history: List[Any]) -> str:
    for msg in reversed(history):
        if getattr(msg, "type", "") == "human":
            return str(getattr(msg, "content", ""))
    return ""


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def should_suggest_new_chat(current_text: str, history_messages: List[Any]) -> bool:
    text = re.sub(r"\s+", " ", (current_text or "")).strip()
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _NEW_TOPIC_MARKERS):
        return False
    if any(marker in lowered for marker in _CONTINUITY_MARKERS):
        return False
    if lowered.startswith("/"):
        return False

    prev = _latest_human(history_messages)
    if not prev:
        return False
    prev_lower = prev.lower().strip()
    if not prev_lower:
        return False
    if any(marker in prev_lower for marker in _NEW_TOPIC_MARKERS):
        return False

    cur_tokens = _tokenize(lowered)
    prev_tokens = _tokenize(prev_lower)
    if len(cur_tokens) < 4 or len(prev_tokens) < 4:
        return False
    overlap = _jaccard(cur_tokens, prev_tokens)
    return overlap < 0.12


def new_chat_tip() -> str:
    return "Tip: This looks like a new topic. Use /new_chat to keep chats segmented."

