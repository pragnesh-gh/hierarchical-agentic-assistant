"""Per-user multi-chat session store with persistent memory."""

import json
import time
import secrets
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)

from config import DATA_DIR
from state import TaskState, EmailFrame, LastAnswer, DraftState, FlagState


SESSIONS_PATH = DATA_DIR / "chat_sessions.json"
MAX_TURNS = 20

_cache_local = threading.local()
_file_lock = threading.Lock()


class SessionCache:
    """Context manager for batching session I/O within a single turn."""

    def __init__(self):
        self._data = None
        self._dirty = False

    def __enter__(self):
        _cache_local.cache = self
        self._data = _load_sessions_raw()
        self._dirty = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dirty and self._data is not None:
            _save_sessions_raw(self._data)
        _cache_local.cache = None
        return False

    def get_data(self):
        return self._data

    def mark_dirty(self):
        self._dirty = True


def _default_task_state() -> TaskState:
    # Canonical per-chat task memory schema; all runtime writes normalize to this shape.
    return {
        "active_task": "",
        "email_frame": {
            "stage": "",
            "recipient": "",
            "topic": "",
            "body": "",
            "pending_confirmation": False,
        },
        "last_answer": {
            "text": "",
            "sources": [],
            "accepted": True,
        },
        "rejected_answers": [],
        "last_contact": "",
        "preferences": {},
    }


def _normalize_task_state(task_state: Any) -> TaskState:
    base = _default_task_state()
    if not isinstance(task_state, dict):
        return base

    active_task = str(task_state.get("active_task", "")).strip()
    if active_task:
        base["active_task"] = active_task

    email_frame = task_state.get("email_frame", {})
    if isinstance(email_frame, dict):
        merged_email = dict(base["email_frame"])
        for key in ("stage", "recipient", "topic", "body"):
            value = email_frame.get(key)
            if value is not None:
                merged_email[key] = str(value).strip()
        merged_email["pending_confirmation"] = bool(
            email_frame.get("pending_confirmation", merged_email["pending_confirmation"])
        )
        base["email_frame"] = merged_email

    last_answer = task_state.get("last_answer", {})
    if isinstance(last_answer, dict):
        merged_last_answer = dict(base["last_answer"])
        merged_last_answer["text"] = str(last_answer.get("text", "")).strip()
        raw_sources = last_answer.get("sources", [])
        if isinstance(raw_sources, list):
            merged_last_answer["sources"] = [str(s).strip() for s in raw_sources if str(s).strip()]
        merged_last_answer["accepted"] = bool(last_answer.get("accepted", True))
        base["last_answer"] = merged_last_answer

    rejected_answers = task_state.get("rejected_answers", [])
    if isinstance(rejected_answers, list):
        cleaned_rejected: List[Dict[str, str]] = []
        for item in rejected_answers[-10:]:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                reason = str(item.get("reason", "")).strip()
            else:
                text = str(item).strip()
                reason = ""
            if text:
                cleaned_rejected.append({"text": text, "reason": reason})
        base["rejected_answers"] = cleaned_rejected

    last_contact = str(task_state.get("last_contact", "")).strip()
    if last_contact:
        base["last_contact"] = last_contact

    preferences = task_state.get("preferences", {})
    if isinstance(preferences, dict):
        base["preferences"] = preferences
    return base


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _title_from_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return "New Chat"
    max_chars = 56
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _chat_title_from_messages(messages: List[BaseMessage], fallback: str = "") -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            title = _title_from_text(str(getattr(msg, "content", "")))
            if title:
                return title
    return _title_from_text(fallback)


def _ensure_chat_meta_defaults(chat: Dict[str, Any]) -> None:
    if not isinstance(chat, dict):
        return
    preview = str(chat.get("preview", "") or "").strip()
    title = str(chat.get("title", "") or "").strip()
    if not title:
        chat["title"] = _title_from_text(preview) or "New Chat"
    chat.setdefault("created_at", _now_iso())
    chat.setdefault("last_active", _now_iso())
    chat.setdefault("archived", False)
    chat.setdefault("pinned", False)


def _load_sessions_raw() -> Dict[str, Any]:
    with _file_lock:
        if not SESSIONS_PATH.exists():
            return {"users": {}}
        with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return {"users": {}}
        if not isinstance(data, dict):
            return {"users": {}}
        if "users" not in data or not isinstance(data["users"], dict):
            data["users"] = {}
        return data


def _save_sessions_raw(data: Dict[str, Any]) -> None:
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock:
        with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _load_sessions() -> Dict[str, Any]:
    cache = getattr(_cache_local, 'cache', None)
    if cache is not None:
        return cache.get_data()
    return _load_sessions_raw()


def _save_sessions(data: Dict[str, Any]) -> None:
    cache = getattr(_cache_local, 'cache', None)
    if cache is not None:
        cache.mark_dirty()
        return
    _save_sessions_raw(data)


def _new_chat_id() -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = secrets.token_hex(2)
    return f"{date}-{suffix}"


def _get_user(data: Dict[str, Any], user_key: str) -> Dict[str, Any]:
    users = data.setdefault("users", {})
    user = users.get(user_key)
    if not isinstance(user, dict):
        user = {"active_chat_id": None, "chats": [], "states": {}, "prefs": {}}
        users[user_key] = user
    user.setdefault("chats", [])
    user.setdefault("states", {})
    user.setdefault("prefs", {})
    chats = user.get("chats", [])
    if isinstance(chats, list):
        for chat in chats:
            if isinstance(chat, dict):
                _ensure_chat_meta_defaults(chat)
    return user


def _ensure_chat(user: Dict[str, Any]) -> str:
    active = user.get("active_chat_id")
    if active and active in user.get("states", {}):
        return active
    new_id = _new_chat_id()
    user["active_chat_id"] = new_id
    user["states"][new_id] = {
        "messages": [],
        "draft": {},
        "flags": {},
        "task_state": _default_task_state(),
    }
    user["chats"].append(
        {
            "id": new_id,
            "created_at": _now_iso(),
            "last_active": _now_iso(),
            "title": "New Chat",
            "preview": "",
            "archived": False,
            "pinned": False,
        }
    )
    return new_id


def _update_chat_meta(user: Dict[str, Any], chat_id: str, preview: Optional[str] = None) -> None:
    for chat in user.get("chats", []):
        if chat.get("id") == chat_id:
            _ensure_chat_meta_defaults(chat)
            chat["last_active"] = _now_iso()
            if preview is not None:
                chat["preview"] = preview
                if not str(chat.get("title", "")).strip() or str(chat.get("title", "")).strip() == "New Chat":
                    chat["title"] = _title_from_text(preview) or "New Chat"
            return


def _trim_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    # Bound in-memory history by human turns to keep prompt cost stable.
    human_indexes = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indexes) <= MAX_TURNS:
        return messages
    cutoff_idx = human_indexes[-MAX_TURNS]
    return messages[cutoff_idx:]


def _filter_memory_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    # Persist only user/assistant conversational content, not internal traces.
    filtered: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage):
            content = str(getattr(msg, "content", ""))
            if content.startswith("[Planner]") or content.startswith("[Debug]"):
                continue
        filtered.append(msg)
    return filtered


def _serialize_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    return [message_to_dict(m) for m in messages]


def _deserialize_messages(items: List[Dict[str, Any]]) -> List[BaseMessage]:
    return messages_from_dict(items)


def get_active_chat(user_key: str) -> str:
    data = _load_sessions()
    user = _get_user(data, user_key)
    chat_id = _ensure_chat(user)
    _save_sessions(data)
    return chat_id


def list_chats(user_key: str) -> List[Dict[str, str]]:
    data = _load_sessions()
    user = _get_user(data, user_key)
    chats = user.get("chats", [])
    if not isinstance(chats, list):
        return []
    rows = [
        {
            "id": str(c.get("id", "")),
            "title": str(c.get("title", "")),
            "created_at": str(c.get("created_at", "")),
            "last_active": str(c.get("last_active", "")),
            "preview": str(c.get("preview", "")),
        }
        for c in chats
    ]
    rows.sort(key=lambda x: x.get("last_active", ""), reverse=True)
    return rows


def new_chat(user_key: str) -> str:
    data = _load_sessions()
    user = _get_user(data, user_key)
    new_id = _new_chat_id()
    user["active_chat_id"] = new_id
    user["states"][new_id] = {"messages": [], "draft": {}, "flags": {}}
    user["states"][new_id]["task_state"] = _default_task_state()
    user["chats"].append(
        {
            "id": new_id,
            "created_at": _now_iso(),
            "last_active": _now_iso(),
            "title": "New Chat",
            "preview": "",
            "archived": False,
            "pinned": False,
        }
    )
    _save_sessions(data)
    return new_id


def switch_chat(user_key: str, chat_id: str) -> bool:
    data = _load_sessions()
    user = _get_user(data, user_key)
    if chat_id in user.get("states", {}):
        user["active_chat_id"] = chat_id
        _save_sessions(data)
        return True
    return False


def load_messages(user_key: str, chat_id: str) -> List[BaseMessage]:
    data = _load_sessions()
    user = _get_user(data, user_key)
    state = user.get("states", {}).get(chat_id, {})
    items = state.get("messages", []) if isinstance(state, dict) else []
    if not isinstance(items, list):
        return []
    return _deserialize_messages(items)


def save_messages(user_key: str, chat_id: str, messages: List[BaseMessage], preview: str) -> None:
    data = _load_sessions()
    user = _get_user(data, user_key)
    state = user.get("states", {}).setdefault(chat_id, {})
    filtered = _filter_memory_messages(messages)
    trimmed = _trim_messages(filtered)
    state["messages"] = _serialize_messages(trimmed)
    _update_chat_meta(user, chat_id, preview=preview)
    for chat in user.get("chats", []):
        if not isinstance(chat, dict):
            continue
        if str(chat.get("id", "")) != chat_id:
            continue
        title = str(chat.get("title", "")).strip()
        if not title or title == "New Chat":
            chat["title"] = _chat_title_from_messages(trimmed, fallback=preview)
        break
    _save_sessions(data)


def get_draft(user_key: str, chat_id: str) -> DraftState:
    data = _load_sessions()
    user = _get_user(data, user_key)
    state = user.get("states", {}).get(chat_id, {})
    draft = state.get("draft", {}) if isinstance(state, dict) else {}
    return draft if isinstance(draft, dict) else {}


def set_draft(user_key: str, chat_id: str, draft: DraftState) -> None:
    data = _load_sessions()
    user = _get_user(data, user_key)
    state = user.get("states", {}).setdefault(chat_id, {})
    state["draft"] = draft
    _update_chat_meta(user, chat_id)
    _save_sessions(data)


def clear_draft(user_key: str, chat_id: str) -> None:
    data = _load_sessions()
    user = _get_user(data, user_key)
    state = user.get("states", {}).setdefault(chat_id, {})
    state["draft"] = {}
    _update_chat_meta(user, chat_id)
    _save_sessions(data)


def get_task_state(user_key: str, chat_id: str) -> TaskState:
    data = _load_sessions()
    user = _get_user(data, user_key)
    state = user.get("states", {}).get(chat_id, {})
    task_state = state.get("task_state", {}) if isinstance(state, dict) else {}
    normalized = _normalize_task_state(task_state)
    if isinstance(state, dict) and state.get("task_state") != normalized:
        state["task_state"] = normalized
        _update_chat_meta(user, chat_id)
        _save_sessions(data)
    return normalized


def set_task_state(user_key: str, chat_id: str, task_state: TaskState) -> None:
    data = _load_sessions()
    user = _get_user(data, user_key)
    state = user.get("states", {}).setdefault(chat_id, {})
    state["task_state"] = _normalize_task_state(task_state)
    _update_chat_meta(user, chat_id)
    _save_sessions(data)


def normalize_task_state(task_state: Any) -> TaskState:
    """Public helper used by runtime nodes before persisting task state."""
    return _normalize_task_state(task_state)


def get_flags(user_key: str, chat_id: str) -> FlagState:
    data = _load_sessions()
    user = _get_user(data, user_key)
    state = user.get("states", {}).get(chat_id, {})
    flags = state.get("flags", {}) if isinstance(state, dict) else {}
    return flags if isinstance(flags, dict) else {}


def set_flags(user_key: str, chat_id: str, updates: FlagState) -> None:
    data = _load_sessions()
    user = _get_user(data, user_key)
    state = user.get("states", {}).setdefault(chat_id, {})
    flags = state.get("flags", {})
    if not isinstance(flags, dict):
        flags = {}
    for key, value in updates.items():
        if value is None:
            flags.pop(key, None)
        else:
            flags[key] = value
    state["flags"] = flags
    _update_chat_meta(user, chat_id)
    _save_sessions(data)


def get_email_prefs(user_key: str) -> Dict[str, Any]:
    data = _load_sessions()
    user = _get_user(data, user_key)
    prefs = user.get("prefs", {})
    email_prefs = prefs.get("email", {}) if isinstance(prefs, dict) else {}
    return email_prefs if isinstance(email_prefs, dict) else {}


def set_email_prefs(user_key: str, updates: Dict[str, Any]) -> None:
    data = _load_sessions()
    user = _get_user(data, user_key)
    prefs = user.setdefault("prefs", {})
    email_prefs = prefs.get("email", {})
    if not isinstance(email_prefs, dict):
        email_prefs = {}
    for key, value in updates.items():
        email_prefs[key] = value
    prefs["email"] = email_prefs
    _save_sessions(data)


def rename_chat(user_key: str, chat_id: str, new_title: str) -> bool:
    title = _title_from_text(new_title)
    if not title:
        return False
    data = _load_sessions()
    user = _get_user(data, user_key)
    for chat in user.get("chats", []):
        if not isinstance(chat, dict):
            continue
        if str(chat.get("id", "")) != str(chat_id):
            continue
        chat["title"] = title
        chat["last_active"] = _now_iso()
        _save_sessions(data)
        return True
    return False


def search_chats(user_key: str, query: str, limit: int = 20) -> List[Dict[str, str]]:
    normalized = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if not normalized:
        return []
    rows = list_chats(user_key)
    out: List[Dict[str, str]] = []
    for row in rows:
        haystack = " ".join(
            [
                str(row.get("title", "")).lower(),
                str(row.get("preview", "")).lower(),
                str(row.get("id", "")).lower(),
            ]
        )
        if normalized in haystack:
            out.append(row)
            if len(out) >= max(1, int(limit)):
                break
    return out


def resolve_chat_selector(user_key: str, selector: str) -> str:
    raw = str(selector or "").strip()
    if not raw:
        return ""
    rows = list_chats(user_key)
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(rows):
            return str(rows[idx - 1].get("id", ""))
    for row in rows:
        if str(row.get("id", "")) == raw:
            return raw
    return ""
