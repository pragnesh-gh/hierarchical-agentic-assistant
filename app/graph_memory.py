"""Graph-oriented long-term memory with Graphiti-first auto fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config import (
    GRAPHITI_PASSWORD,
    GRAPHITI_URI,
    GRAPHITI_USER,
    GRAPH_MEMORY_BACKEND,
    GRAPH_MEMORY_ENABLED,
    GRAPH_MEMORY_MAX_CHARS,
    GRAPH_MEMORY_MAX_FACTS_PER_USER,
    GRAPH_MEMORY_MIN_SCORE,
    GRAPH_MEMORY_PATH,
    GRAPH_MEMORY_TOP_K,
)
from vocabulary import DEICTIC_RE


logger = logging.getLogger(__name__)

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

_EMAIL_RE = re.compile(r"\b[\w.\-+]+@[\w.\-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"https?://\S+")
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_CAP_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|"
    r"\d{4})\b",
    flags=re.IGNORECASE,
)
_DEICTIC_RE = DEICTIC_RE
_EMAIL_INTENT_RE = re.compile(
    r"\b(?:email|mail|send|forward|share)\b",
    flags=re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(?:summary|summarize|summarise|recap|combine|all updates|digest)\b",
    flags=re.IGNORECASE,
)
_CROSS_CHAT_RE = re.compile(
    r"\b(?:previous chat|earlier chat|another chat|other chat|past chat|from history|"
    r"from before|across chats|old chat|last chat)\b",
    flags=re.IGNORECASE,
)
_SMALLTALK_AI_RE = re.compile(
    r"\b(?:how can i help|how are you|hello|hi there|sources:\s*none)\b",
    flags=re.IGNORECASE,
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    lowered = (text or "").lower()
    out: List[str] = []
    for tok in _TOKEN_RE.findall(lowered):
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _extract_entities(text: str) -> List[str]:
    if not text:
        return []
    entities = set()
    for email in _EMAIL_RE.findall(text):
        entities.add(email.lower())
    for url in _URL_RE.findall(text):
        entities.add(url.rstrip(").,;"))
    for match in _CAP_ENTITY_RE.findall(text):
        val = " ".join(match.split()).strip()
        if val and len(val) > 2:
            entities.add(val)
    for date_val in _DATE_RE.findall(text):
        entities.add(str(date_val))
    return sorted(entities)


def _normalize_preview(text: str, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3] + "..."


def _has_url(text: str) -> bool:
    return bool(_URL_RE.search(text or ""))


def _has_date(text: str) -> bool:
    return bool(_DATE_RE.search(text or ""))


def _query_intents(query: str) -> Dict[str, bool]:
    # Lightweight intent hints used to bias retrieval scoring behavior.
    text = (query or "").strip()
    return {
        "deictic": bool(_DEICTIC_RE.search(text)),
        "email_intent": bool(_EMAIL_INTENT_RE.search(text)),
        "summary_intent": bool(_SUMMARY_RE.search(text)),
        "cross_chat": bool(_CROSS_CHAT_RE.search(text)),
    }


def _classify_turn(human: str, ai: str) -> str:
    h = (human or "").strip().lower()
    a = (ai or "").strip().lower()
    if not a:
        return "empty"
    if a.startswith("confirm send?") or "reply yes/no" in a:
        return "email_draft"
    if a.startswith("email sent"):
        return "email_sent"
    if a.startswith("error:"):
        return "error"
    if _SMALLTALK_AI_RE.search(a) and len((human or "").split()) <= 10:
        return "smalltalk"
    if _has_url(ai) or "sources:" in a or _has_date(ai):
        return "qa_fact"
    if "who should i email" in a or "what should the email say" in a:
        return "email_clarification"
    if "thank you" in a and "sources:" not in a:
        return "email_body"
    if "news" in h or "today" in h or "latest" in h:
        return "qa_fact"
    return "general"


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


class _MemoryBackend:
    name = "none"

    def available(self) -> bool:
        return True

    def retrieve(self, user_key: str, chat_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        return []

    def ingest(self, user_key: str, chat_id: str, human: str, ai: str) -> None:
        return None


class _NullBackend(_MemoryBackend):
    name = "disabled"

    def available(self) -> bool:
        return False


class _LocalGraphBackend(_MemoryBackend):
    name = "local-json-graph"

    def __init__(self, path: Path, max_facts_per_user: int = 800) -> None:
        self.path = path
        self.max_facts_per_user = max(1, int(max_facts_per_user))
        self._lock = threading.Lock()

    def _empty_store(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "facts": [],
            "entity_index": {},
        }

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty_store()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return self._empty_store()
        if not isinstance(data, dict):
            return self._empty_store()
        data.setdefault("version", 1)
        data.setdefault("facts", [])
        data.setdefault("entity_index", {})
        if not isinstance(data.get("facts"), list):
            data["facts"] = []
        if not isinstance(data.get("entity_index"), dict):
            data["entity_index"] = {}
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _trim(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        per_user: Dict[str, List[Dict[str, Any]]] = {}
        for fact in facts:
            user = str(fact.get("user_key", "")).strip()
            if not user:
                continue
            per_user.setdefault(user, []).append(fact)
        trimmed: List[Dict[str, Any]] = []
        for user_facts in per_user.values():
            user_facts = sorted(user_facts, key=lambda x: str(x.get("ts", "")))
            if len(user_facts) > self.max_facts_per_user:
                user_facts = user_facts[-self.max_facts_per_user :]
            trimmed.extend(user_facts)
        return sorted(trimmed, key=lambda x: str(x.get("ts", "")))

    def _rebuild_entity_index(self, facts: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
        entity_index: Dict[str, List[str]] = {}
        for fact in facts:
            fact_id = str(fact.get("id", "")).strip()
            if not fact_id:
                continue
            entities = fact.get("entities", [])
            if not isinstance(entities, list):
                continue
            for entity in entities:
                label = str(entity).strip()
                if not label:
                    continue
                bucket = entity_index.setdefault(label, [])
                bucket.append(fact_id)
        for label in list(entity_index.keys()):
            entity_index[label] = sorted(set(entity_index[label]))
        return entity_index

    def ingest(self, user_key: str, chat_id: str, human: str, ai: str) -> None:
        # Store one turn as a "fact" plus searchable lexical/entity features.
        if not user_key:
            return
        human_clean = (human or "").strip()
        ai_clean = (ai or "").strip()
        if not human_clean and not ai_clean:
            return
        human_tokens = sorted(set(_tokenize(human_clean)))
        ai_tokens = sorted(set(_tokenize(ai_clean)))
        text = f"User: {human_clean}\nAssistant: {ai_clean}".strip()
        fact = {
            "id": secrets.token_hex(8),
            "user_key": str(user_key),
            "chat_id": str(chat_id or ""),
            "ts": _utc_iso(),
            "human": human_clean[:1200],
            "ai": ai_clean[:2000],
            "text": text[:3200],
            "tokens": sorted(set(_tokenize(text))),
            "human_tokens": human_tokens,
            "ai_tokens": ai_tokens,
            "entities": _extract_entities(text),
            "human_entities": _extract_entities(human_clean),
            "ai_entities": _extract_entities(ai_clean),
            "turn_type": _classify_turn(human_clean, ai_clean),
            "has_sources": "sources:" in ai_clean.lower(),
            "has_url": _has_url(ai_clean),
            "has_date": _has_date(ai_clean),
        }
        with self._lock:
            store = self._load()
            facts = store.get("facts", [])
            if not isinstance(facts, list):
                facts = []
            facts.append(fact)
            facts = self._trim(facts)
            store["facts"] = facts
            store["entity_index"] = self._rebuild_entity_index(facts)
            self._save(store)

    def retrieve(self, user_key: str, chat_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        # Hybrid lexical + entity + recency scoring with chat-boundary controls.
        if not user_key:
            return []
        query_clean = (query or "").strip()
        if not query_clean:
            return []
        with self._lock:
            store = self._load()
        facts = store.get("facts", [])
        if not isinstance(facts, list):
            return []

        user_facts = [f for f in facts if str(f.get("user_key", "")) == str(user_key)]
        if not user_facts:
            return []

        q_tokens = set(_tokenize(query_clean))
        q_entities = set(_extract_entities(query_clean))
        intent = _query_intents(query_clean)
        if not intent["cross_chat"]:
            user_facts = [f for f in user_facts if str(f.get("chat_id", "")) == str(chat_id)]
            if not user_facts:
                return []
        now = datetime.now(timezone.utc)
        candidates: List[Dict[str, Any]] = []

        for fact in user_facts:
            f_tokens = set(fact.get("tokens", []) if isinstance(fact.get("tokens", []), list) else [])
            f_entities = set(
                fact.get("entities", []) if isinstance(fact.get("entities", []), list) else []
            )
            f_ai_tokens = set(
                fact.get("ai_tokens", []) if isinstance(fact.get("ai_tokens", []), list) else []
            )
            f_human_tokens = set(
                fact.get("human_tokens", []) if isinstance(fact.get("human_tokens", []), list) else []
            )
            token_overlap = len(q_tokens & f_tokens)
            entity_overlap = len(q_entities & f_entities)
            ai_overlap = len(q_tokens & f_ai_tokens)
            human_overlap = len(q_tokens & f_human_tokens)

            score = 0.0
            if q_tokens:
                score += 0.75 * (token_overlap / max(1, len(q_tokens)))
                score += 0.35 * (ai_overlap / max(1, len(q_tokens)))
                score += 0.15 * (human_overlap / max(1, len(q_tokens)))
            if q_entities:
                score += 1.6 * (entity_overlap / max(1, len(q_entities)))

            same_chat = str(fact.get("chat_id", "")) == str(chat_id)
            if same_chat:
                score += 0.25

            ts_raw = str(fact.get("ts", "")).strip()
            age_hours = 1e6
            if ts_raw:
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
                    recency_bonus = max(0.0, 0.2 - min(age_hours / 720.0, 0.2))
                    score += recency_bonus
                except Exception:
                    pass

            turn_type = str(fact.get("turn_type", "")).strip().lower()
            has_url = bool(fact.get("has_url", False))
            has_date = bool(fact.get("has_date", False))
            has_sources = bool(fact.get("has_sources", False))

            if turn_type == "qa_fact":
                score += 0.1
            if turn_type == "smalltalk":
                score -= 0.25
            if turn_type in {"email_draft", "email_body"}:
                score -= 0.1
            if has_date:
                score += 0.08
            if has_url or has_sources:
                score += 0.06

            # Deictic follow-ups ("send this information...") should prefer
            # recent same-chat factual turns, even if lexical overlap is sparse.
            if intent["deictic"]:
                if same_chat:
                    score += 0.55
                if turn_type == "qa_fact":
                    score += 0.45
                if age_hours <= 24:
                    score += 0.25
                elif age_hours <= 72:
                    score += 0.12
                if turn_type == "smalltalk":
                    score -= 0.35

            # Email follow-ups often ask to package previous facts for sending.
            if intent["email_intent"] and intent["deictic"] and turn_type == "qa_fact":
                score += 0.35

            no_anchor_match = token_overlap == 0 and entity_overlap == 0
            if no_anchor_match and not (intent["deictic"] and same_chat):
                # Keep strict relevance boundaries to avoid topic bleed.
                continue
            if score < GRAPH_MEMORY_MIN_SCORE and no_anchor_match and not (
                intent["deictic"] and same_chat
            ):
                continue

            candidates.append(
                {
                    "id": str(fact.get("id", "")),
                    "score": round(score, 5),
                    "text": str(fact.get("text", "")),
                    "ts": ts_raw,
                    "chat_id": str(fact.get("chat_id", "")),
                    "_tokens": list(f_tokens),
                    "_entities": list(f_entities),
                }
            )

        candidates.sort(key=lambda x: (x["score"], x["ts"]), reverse=True)
        if not candidates:
            return []

        # Diversity for summary-style prompts to avoid near-duplicate turns.
        target_k = max(1, int(top_k))
        if intent["summary_intent"] and len(candidates) > 1:
            selected: List[Dict[str, Any]] = []
            for cand in candidates:
                if len(selected) >= target_k:
                    break
                novelty_penalty = 0.0
                for chosen in selected:
                    tok_sim = _jaccard(cand.get("_tokens", []), chosen.get("_tokens", []))
                    ent_sim = _jaccard(cand.get("_entities", []), chosen.get("_entities", []))
                    novelty_penalty = max(novelty_penalty, 0.65 * tok_sim + 0.85 * ent_sim)
                adjusted = float(cand["score"]) - novelty_penalty
                if adjusted < GRAPH_MEMORY_MIN_SCORE and selected:
                    continue
                cand["score"] = round(adjusted, 5)
                selected.append(cand)
            if selected:
                selected.sort(key=lambda x: (x["score"], x["ts"]), reverse=True)
                out: List[Dict[str, Any]] = []
                for item in selected[:target_k]:
                    cleaned = dict(item)
                    cleaned.pop("_tokens", None)
                    cleaned.pop("_entities", None)
                    out.append(cleaned)
                return out

        out2: List[Dict[str, Any]] = []
        for item in candidates[:target_k]:
            cleaned = dict(item)
            cleaned.pop("_tokens", None)
            cleaned.pop("_entities", None)
            out2.append(cleaned)
        return out2


class _GraphitiBackend(_MemoryBackend):
    """Best-effort adapter to Graphiti. Falls back on unsupported API shapes."""

    name = "graphiti"

    def __init__(self, uri: str, user: str, password: str) -> None:
        if not uri or not user or not password:
            raise RuntimeError("Graphiti credentials missing")
        try:
            from graphiti_core import Graphiti  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(f"Graphiti import failed: {exc}") from exc
        self._graphiti = Graphiti(uri, user, password)
        self._ready = False

    def _run_async(self, coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        return asyncio.run(coro)

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        build = getattr(self._graphiti, "build_indices_and_constraints", None)
        if callable(build):
            try:
                self._run_async(build())
            except Exception:
                pass
        self._ready = True

    def ingest(self, user_key: str, chat_id: str, human: str, ai: str) -> None:
        self._ensure_ready()
        add_episode = getattr(self._graphiti, "add_episode", None)
        if not callable(add_episode):
            raise RuntimeError("Graphiti add_episode unavailable")

        content = f"User({user_key}/{chat_id}): {human}\nAssistant: {ai}".strip()
        reference_time = datetime.now(timezone.utc)

        candidate_kwargs = [
            {
                "episode_type": "message",
                "episode_body": content,
                "group_id": f"{user_key}:{chat_id}",
                "reference_time": reference_time,
                "source_description": "hierarchical-agentic-qa-chat",
            },
            {
                "episode_type": "message",
                "episode_body": content,
                "reference_time": reference_time,
                "source_description": "hierarchical-agentic-qa-chat",
            },
            {
                "episode_body": content,
                "reference_time": reference_time,
            },
        ]
        last_error: Optional[Exception] = None
        for kwargs in candidate_kwargs:
            try:
                self._run_async(add_episode(**kwargs))
                return
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Graphiti add_episode failed: {last_error}")

    def retrieve(self, user_key: str, chat_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        self._ensure_ready()
        search = getattr(self._graphiti, "search", None)
        if not callable(search):
            raise RuntimeError("Graphiti search unavailable")
        try:
            result = self._run_async(
                search(query=query, num_results=max(1, int(top_k)), group_id=f"{user_key}:{chat_id}")
            )
        except TypeError:
            result = self._run_async(search(query=query, num_results=max(1, int(top_k))))
        hits: List[Dict[str, Any]] = []
        if isinstance(result, list):
            raw_hits = result
        else:
            raw_hits = getattr(result, "nodes", None) or getattr(result, "results", None) or []
        for item in raw_hits:
            text = ""
            score = 0.0
            if isinstance(item, dict):
                text = str(
                    item.get("text")
                    or item.get("summary")
                    or item.get("content")
                    or item.get("fact")
                    or ""
                )
                try:
                    score = float(item.get("score", 0.0))
                except Exception:
                    score = 0.0
            else:
                text = str(getattr(item, "text", "") or getattr(item, "summary", "") or "")
                try:
                    score = float(getattr(item, "score", 0.0) or 0.0)
                except Exception:
                    score = 0.0
            if text.strip():
                hits.append(
                    {
                        "id": str(getattr(item, "id", "") if not isinstance(item, dict) else item.get("id", "")),
                        "score": score,
                        "text": text.strip(),
                        "ts": "",
                        "chat_id": str(chat_id),
                    }
                )
        return hits[: max(1, int(top_k))]


class GraphMemoryService:
    """Singleton-friendly memory service."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fallback = _LocalGraphBackend(GRAPH_MEMORY_PATH, GRAPH_MEMORY_MAX_FACTS_PER_USER)
        self._backend: _MemoryBackend = self._build_backend()

    def _build_backend(self) -> _MemoryBackend:
        # Auto mode prefers Graphiti and degrades to local-json on failure.
        if not GRAPH_MEMORY_ENABLED:
            logger.info("Graph memory: disabled")
            return _NullBackend()

        backend = (GRAPH_MEMORY_BACKEND or "auto").strip().lower()
        if backend in {"graphiti", "auto"}:
            try:
                result = _GraphitiBackend(GRAPHITI_URI, GRAPHITI_USER, GRAPHITI_PASSWORD)
                logger.info("Graph memory: using Graphiti backend")
                return result
            except Exception as exc:
                if backend == "graphiti":
                    logger.warning("Graph memory: Graphiti backend requested but unavailable, falling back to local JSON — %s", exc)
                    return self._fallback
                logger.warning("Graph memory: Graphiti unavailable, falling back to local JSON — %s", exc)
        if backend in {"local", "json", "auto"}:
            logger.info("Graph memory: using local JSON backend")
            return self._fallback
        logger.info("Graph memory: using local JSON backend")
        return self._fallback

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def ingest_turn(self, user_key: str, chat_id: str, human: str, ai: str) -> None:
        if not GRAPH_MEMORY_ENABLED:
            return
        human_clean = (human or "").strip()
        ai_clean = (ai or "").strip()
        if not human_clean and not ai_clean:
            return
        # Avoid poisoning long-term memory with internal traces/debug.
        if ai_clean.startswith("[Planner]") or ai_clean.startswith("[Debug]"):
            return
        with self._lock:
            try:
                self._backend.ingest(user_key, chat_id, human_clean, ai_clean)
                return
            except Exception as exc:
                # If primary backend fails at runtime, degrade gracefully.
                if self._backend is not self._fallback:
                    logger.warning("Graph memory ingest failed on %s, degrading to %s — %s", self._backend.name, self._fallback.name, exc)
                    self._backend = self._fallback
                try:
                    self._backend.ingest(user_key, chat_id, human_clean, ai_clean)
                except Exception as exc:
                    logger.error("Graph memory ingest failed on all backends — %s", exc)
                    return

    def retrieve_hits(self, user_key: str, chat_id: str, query: str, top_k: int = GRAPH_MEMORY_TOP_K) -> List[Dict[str, Any]]:
        if not GRAPH_MEMORY_ENABLED:
            return []
        query_clean = (query or "").strip()
        if not query_clean:
            return []
        with self._lock:
            try:
                return self._backend.retrieve(user_key, chat_id, query_clean, max(1, int(top_k)))
            except Exception as exc:
                if self._backend is not self._fallback:
                    logger.warning("Graph memory retrieve failed on %s, degrading to %s — %s", self._backend.name, self._fallback.name, exc)
                    self._backend = self._fallback
                    try:
                        return self._backend.retrieve(user_key, chat_id, query_clean, max(1, int(top_k)))
                    except Exception as exc:
                        logger.error("Graph memory retrieve failed on all backends — %s", exc)
                        return []
                return []

    def retrieve_context(
        self,
        user_key: str,
        chat_id: str,
        query: str,
        top_k: int = GRAPH_MEMORY_TOP_K,
        max_chars: int = GRAPH_MEMORY_MAX_CHARS,
    ) -> str:
        hits = self.retrieve_hits(user_key, chat_id, query, top_k=top_k)
        if not hits:
            return ""
        lines = ["Long-term memory hints:"]
        remaining = max(120, int(max_chars))
        for idx, hit in enumerate(hits, start=1):
            body = _normalize_preview(str(hit.get("text", "")), max_chars=220)
            line = f"- ({idx}) {body}"
            if len(line) + 1 > remaining:
                break
            lines.append(line)
            remaining -= len(line) + 1
            if remaining < 40:
                break
        return "\n".join(lines).strip()


_SERVICE: Optional[GraphMemoryService] = None
_SERVICE_LOCK = threading.Lock()


def _get_service() -> GraphMemoryService:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = GraphMemoryService()
    return _SERVICE


def retrieve_memory_context(user_key: str, chat_id: str, query: str) -> str:
    return _get_service().retrieve_context(user_key, chat_id, query)


def ingest_turn_memory(user_key: str, chat_id: str, human: str, ai: str) -> None:
    _get_service().ingest_turn(user_key, chat_id, human, ai)


def graph_memory_backend_name() -> str:
    return _get_service().backend_name


# Public aliases for focused tests and tooling.
LocalGraphMemoryBackend = _LocalGraphBackend
