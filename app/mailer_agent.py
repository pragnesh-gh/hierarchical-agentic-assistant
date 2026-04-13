"""Mailer agent that drafts and sends allowlisted emails."""

import re
import json
from typing import Any, Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from contacts import resolve_contact, load_contacts
from tools_email import send_email_to_address, send_email_to_contact
from chat_sessions import (
    get_draft,
    set_draft,
    clear_draft,
    get_email_prefs,
    set_email_prefs,
    normalize_task_state,
    set_flags,
)
from config import ASYNC_TIMEOUT_EMAIL_SEC
from identity import load_identity_text
from intent_utils import is_no_email_only, parse_confirmation_intent
from state import DraftState, TaskState
from vocabulary import SUMMARY_MARKERS


DEFAULT_TONE = "formal conversational"
DEFAULT_SIGNATURE_NAME = "Pragnesh Kumar"
AI_FOOTER = "\n\n---\nThis is an AI automated email sent by hierarchical_qa_bot."
ASSISTANT_IDENTITY_LINE = "I am Arjun, Pragnesh's AI assistant."
EMAIL_IN_TEXT_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


# ---------------------------------------------------------------------------
# JSON parsing helpers  (must appear before create_mailer)
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    # Some reasoning models leak hidden-thought blocks before JSON.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _decode_json_string(raw: str) -> str:
    if raw is None:
        return ""
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        decoded = raw.replace("\\n", "\n").replace("\\t", "\t")
        decoded = decoded.replace('\\"', '"').replace("\\\\", "\\")
        return decoded


def _extract_json_string_value(text: str, start: int) -> Tuple[str, int, bool]:
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != '"':
        return "", i, False
    i += 1
    buf: List[str] = []
    escaped = False
    while i < len(text):
        ch = text[i]
        if escaped:
            buf.append(ch)
            escaped = False
        else:
            if ch == "\\":
                escaped = True
                buf.append(ch)
            elif ch == '"':
                return "".join(buf), i + 1, True
            else:
                buf.append(ch)
        i += 1
    return "".join(buf), i, False


def _extract_json_field(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:', text, re.IGNORECASE)
    if not match:
        return ""
    value, _, ok = _extract_json_string_value(text, match.end())
    if not ok:
        return ""
    return _decode_json_string(value).strip()


def _parse_json_fields(text: str) -> Tuple[str, str, str]:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "", "", "missing JSON payload"
    json_text = cleaned[start : end + 1]
    try:
        data = json.loads(json_text)
    except Exception:
        return "", "", "invalid JSON payload"
    subject = str(data.get("subject", "")).strip()
    body = str(data.get("body", "")).strip()
    if not subject:
        return "", "", "missing subject"
    if not body:
        return "", "", "missing body"
    return subject, body, ""


def _parse_lenient_json_fields(text: str) -> Tuple[str, str, str]:
    cleaned = _strip_code_fences(text)
    subject = _extract_json_field(cleaned, "subject")
    body = _extract_json_field(cleaned, "body")
    if not subject:
        return "", "", "missing subject"
    if not body:
        return "", "", "missing body"
    return subject, body, ""


def _fallback_extract_fields(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    subject_match = re.search(r"^\s*subject\s*[:\-]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    body_match = re.search(r"^\s*body\s*[:\-]\s*", text, re.IGNORECASE | re.MULTILINE)
    subject = subject_match.group(1).strip() if subject_match else ""
    body = ""
    if body_match:
        body_start = body_match.end()
        body = text[body_start:].strip()
    return subject, body


def _extract_fields(text: str) -> Tuple[str, str, str]:
    subject, body, error = _parse_json_fields(text)
    if error in ("missing JSON payload", "invalid JSON payload"):
        subject_lenient, body_lenient, err_lenient = _parse_lenient_json_fields(text)
        if not err_lenient:
            return subject_lenient, body_lenient, ""
        fallback_subject, fallback_body = _fallback_extract_fields(text)
        if fallback_subject and fallback_body:
            return fallback_subject, fallback_body, ""
    return subject, body, error


# ---------------------------------------------------------------------------
# Intent / extraction helpers
# ---------------------------------------------------------------------------

def _extract_tone(text: str, default_tone: str) -> Tuple[str, str, bool]:
    tone = default_tone
    match = re.search(r"\btone\s*[:=]?\s*([^\n]+)$", text, re.IGNORECASE)
    if match:
        tone = match.group(1).strip()
        text = text[: match.start()].strip()
        return tone, text, True
    return tone, text, False


def _strip_body_prefix(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = cleaned.lstrip(" ,;:.-?!")
    cleaned = re.sub(
        r"^(and\s+)?(the\s+)?(email\s+)?((him|her|them)\s+that|should|that|to|about|regarding|with|saying|says)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _fallback_topic_from_email_request(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"\b(?:send|compose|draft|shoot|drop|forward|share)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:an?\s+)?(?:mail|email)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bto\s+[^\n,.!?]+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    cleaned = _strip_body_prefix(cleaned)
    return cleaned.strip()


def _normalize_contact_name(name: str) -> str:
    if not name:
        return ""
    cleaned = name.strip()
    cleaned = re.sub(r"^(my|our|the)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(listed\s+contact|contact)\s+", "", cleaned, flags=re.IGNORECASE)
    relation = r"brother|sister|dad|father|mom|mother|parent|wife|husband|friend"
    cleaned = re.sub(rf"^(?:{relation})\s+(?=\S)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"'s$", "", cleaned)
    cleaned = cleaned.strip(" ,.;\n\t")
    return cleaned.strip()


def _trim_recipient_tail(name: str) -> str:
    """Trim accidental instruction tail captured with recipient name."""
    if not name:
        return ""
    cleaned = re.sub(r"\s+", " ", name).strip()
    marker = re.search(
        r"\b(remind(?:ing)?|tell(?:ing)?|ask(?:ing)?|inform(?:ing)?|notify(?:ing)?|saying|says)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if marker:
        cleaned = cleaned[: marker.start()].strip(" ,.;:-")
    # Trim trailing "via email/mail" that gets captured with recipient name
    cleaned = re.sub(r"\s+(?:via|through|by|over)\s+(?:email|mail|e-mail)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_direct_email(text: str) -> str:
    if not text:
        return ""
    match = EMAIL_IN_TEXT_RE.search(text)
    if not match:
        return ""
    return match.group(0).strip()


def _match_email_intent(text: str) -> Tuple[str, str]:
    connector = (
        r"(?:,|;|:|\.|\?|!|\n|\s-\s|\s+and\s+|\s+that\s+|\s+which\s+|\s+with\s+|"
        r"\s+about\s+|\s+regarding\s+|\s+remind(?:ing)?\s+|\s+tell(?:ing)?\s+|"
        r"\s+ask(?:ing)?\s+|\s+inform(?:ing)?\s+|\s+notify(?:ing)?\s+)"
    )
    patterns = [
        r"\b(?:send\s+)?(?:a\s+)?(?:demo\s+)?(?:mail|email)\s+about\s+(?P<body>.+?)\s+to\s+(?P<name>[^,\.\n!?]+)",
        rf"\b(?:compose|draft|shoot|drop)\s+an?\s+email\s+to\s+(?P<name>[^\.\n!?]+?)(?:{connector}(?P<body>.+))?$",
        rf"\b(?:send\s+)?(?:a\s+)?(?:demo\s+)?(?:mail|email)\s+to\s+(?P<name>[^\.\n!?]+?)(?:{connector}(?P<body>.+))?$",
        rf"\bsend\s+(?P<name>[^\.\n!?]+?)\s+an?\s+email(?:{connector}(?P<body>.+))?$",
        rf"\b(?:send|forward|share)\s+(?:this|it|that|this information|this info|the same information|same information|same info|same answer|that information|that answer)\s+(?:to|with)\s+(?P<name>[^\.\n!?]+?)(?:{connector}(?P<body>.+))?$",
        rf"\b(?:mail|email)\s+(?P<name>[^\s\.\n!?,]+(?:\s+[^\s\.\n!?,]+)?)\s+(?:about|regarding)\s+(?P<body>.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        name = (match.group("name") or "").strip()
        name = _trim_recipient_tail(name)
        body = (match.groupdict().get("body") or "").strip()
        if not body:
            tail = text[match.end() :].strip()
            body = _strip_body_prefix(tail)
        body = _strip_body_prefix(body)
        return _normalize_contact_name(name), body
    return "", ""


def _wants_summary_content(text: str) -> bool:
    normalized = (text or "").lower()
    return any(marker in normalized for marker in SUMMARY_MARKERS)


def _instruction_requests_research(text: str) -> bool:
    normalized = (text or "").lower()
    if not normalized:
        return False
    hints = (
        "research",
        "internet",
        "web",
        "online",
        "search",
        "look up",
        "find",
        "latest",
        "today",
        "winner",
        "news",
    )
    return any(hint in normalized for hint in hints)


def _tool_context_for_edit(messages: List[Any], max_chars: int = 1400) -> str:
    snippets: List[str] = []
    for msg in messages:
        if getattr(msg, "type", "") != "tool":
            continue
        name = str(getattr(msg, "name", "") or "tool")
        content = _strip_error_lines(str(getattr(msg, "content", "")))
        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            continue
        snippets.append(f"- [{name}] {content[:380]}")
        if len(snippets) >= 4:
            break
    if not snippets:
        return "none"
    return "\n".join(snippets)[:max_chars]


_REQUEST_STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "and", "or", "in", "on", "at", "with",
    "is", "are", "be", "this", "that", "it", "him", "her", "them", "you", "your",
    "please", "mail", "email", "send", "about", "regarding", "remind", "reminder",
    "update", "body", "subject", "tone",
}


def _request_keywords(text: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) >= 4 and t not in _REQUEST_STOPWORDS}


def _body_covers_request(body_content: str, request_text: str) -> bool:
    req = _request_keywords(request_text)
    if not req:
        return True
    body = _request_keywords(_strip_sources_from_text(body_content or "", set()))
    overlap = req.intersection(body)
    required = 1 if len(req) <= 3 else 2
    return len(overlap) >= required


def _is_generic_subject(subject: str) -> bool:
    normalized = (subject or "").strip().lower()
    if not normalized:
        return True
    generic = {
        "introduction",
        "re: request",
        "pragnesh's ai assistant",
        "ai assistant",
    }
    return normalized in generic


def _fallback_subject_from_request(request_text: str) -> str:
    cleaned = _strip_body_prefix(request_text or "").strip(" .,:;!?")
    normalized = cleaned.lower()
    if "interview" in normalized and "tomorrow" in normalized and ("10am" in normalized or "10 am" in normalized):
        return "Interview Reminder: Tomorrow at 10 AM"
    if "interview" in normalized and "tomorrow" in normalized:
        return "Interview Reminder: Tomorrow"
    if "movie" in normalized and "ticket" in normalized:
        return "Reminder: Book Movie Tickets"
    if "movie" in normalized:
        return "Movie Reminder"
    if "remind" in normalized:
        return "Reminder"
    words = [w for w in cleaned.split() if w]
    if not words:
        return "Reminder"
    return " ".join(words[:8]).strip().capitalize()


def _apply_request_fallback(subject: str, body_content: str, request_text: str) -> Tuple[str, str]:
    """Keep generated drafts anchored to explicit user request details."""
    request = (request_text or "").strip()
    out_subject = (subject or "").strip()
    out_body = (body_content or "").strip()
    if not request:
        return out_subject, out_body

    if not _body_covers_request(out_body, request):
        out_body = request
    if _is_generic_subject(out_subject):
        out_subject = _fallback_subject_from_request(request)
    return out_subject, out_body


def _availability_request_line(user_text: str) -> str:
    normalized = (user_text or "").lower()
    if "free this week" in normalized:
        return "Could you let me know when you are free this week to discuss this?"
    if "when are you free" in normalized or "when he is free" in normalized or "availability" in normalized:
        return "Could you let me know when you are free to discuss this?"
    return ""


def _clean_fact_line(text: str) -> str:
    cleaned = _strip_sources_from_text(text or "", set())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(
        r"^(?:i am arjun[, ]*pragnesh'?s ai assistant\.?\s*)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" -")


def _extract_memory_facts(memory_context: str) -> List[str]:
    if not memory_context:
        return []
    facts: List[str] = []
    for line in memory_context.splitlines():
        ln = line.strip()
        if not ln.startswith("-"):
            continue
        payload = re.sub(r"^-\s*\(\d+\)\s*", "", ln).strip()
        if "Assistant:" in payload:
            payload = payload.split("Assistant:", 1)[1].strip()
        payload = _clean_fact_line(payload)
        if payload and len(payload) > 20:
            facts.append(payload)
    return facts


def _recent_ai_facts(messages: List[Any], limit: int = 3) -> List[str]:
    facts: List[str] = []
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "ai":
            continue
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue
        lowered = content.lower()
        if lowered.startswith("[planner]") or lowered.startswith("[debug]"):
            continue
        if "confirm send?" in lowered:
            continue
        if lowered.startswith("email sent"):
            continue
        if "sources: none" in lowered and "http" not in lowered:
            continue
        core = _clean_fact_line(content.split("\n\nSources:", 1)[0].strip())
        if core and len(core) > 20:
            facts.append(core)
            if len(facts) >= limit:
                break
    return list(reversed(facts))


def _dedupe_facts(facts: List[str], max_facts: int = 4) -> List[str]:
    out: List[str] = []
    seen = set()
    for fact in facts:
        norm = re.sub(r"\W+", " ", fact.lower()).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(fact)
        if len(out) >= max_facts:
            break
    return out


def _fact_subject(facts: List[str]) -> str:
    joined = " ".join(facts).lower()
    has_openai = "openai" in joined
    has_tesla = "tesla" in joined
    if has_openai and has_tesla:
        return "Tesla and OpenAI Updates"
    if has_openai:
        return "OpenAI Update"
    if has_tesla:
        return "Tesla Update"
    return "Summary of Recent Updates"


def _compose_fact_transfer_content(facts: List[str], user_text: str) -> str:
    lines: List[str] = []
    if facts:
        lines.append("Here is a quick summary of the key updates we discussed:")
        for fact in facts:
            lines.append(f"- {fact}")
    availability = _availability_request_line(user_text)
    if availability:
        if lines:
            lines.append("")
        lines.append(availability)
    return "\n".join(lines).strip()


def _collect_fact_bundle(messages: List[Any], task_state: TaskState, memory_context: str, user_text: str) -> List[str]:
    # Merge candidate facts from short-term chat + task snapshot + long-term memory.
    # This lets "send this information" resolve to concrete content.
    candidates: List[str] = []
    recent = _recent_ai_facts(messages, limit=4)
    candidates.extend(recent)
    last_answer = _clean_fact_line(_task_last_answer_text(task_state))
    if last_answer:
        candidates.append(last_answer)
    mem = _extract_memory_facts(memory_context)
    candidates.extend(mem)
    max_facts = 4 if _wants_summary_content(user_text) else 2
    return _dedupe_facts(candidates, max_facts=max_facts)


def _contacts_for_prompt() -> str:
    contacts = []
    for contact in load_contacts():
        name = str(contact.get("name", "")).strip()
        aliases = contact.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        aliases = [str(a).strip() for a in aliases if str(a).strip()]
        if name or aliases:
            contacts.append({"name": name, "aliases": aliases})
    return json.dumps(contacts, ensure_ascii=True)


def _parse_intent_json(text: str) -> Tuple[str, str, str]:
    if not text:
        return "", "", ""
    text = _strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "", "", ""
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return "", "", ""
    to_name = str(
        data.get("to_name")
        or data.get("to")
        or data.get("recipient")
        or data.get("name")
        or ""
    ).strip()
    body = str(data.get("body") or "").strip()
    tone = str(data.get("tone") or "").strip()
    return to_name, body, tone


def _find_contact_in_text(text: str) -> Tuple[str, List[str]]:
    normalized = (text or "").lower()
    if not normalized:
        return "", []
    matches = []
    for contact in load_contacts():
        name = str(contact.get("name", "")).strip()
        aliases = contact.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        candidates = [name] + [str(a).strip() for a in aliases]
        candidates = [c for c in candidates if c]
        for candidate in candidates:
            pattern = rf"(?<!\w){re.escape(candidate.lower())}(?!\w)"
            if re.search(pattern, normalized):
                matches.append(contact)
                break
    if len(matches) == 1:
        return str(matches[0].get("name", "")).strip(), []
    if len(matches) > 1:
        names = [str(m.get("name", "")) for m in matches]
        return "", names
    return "", []


def _latest_user_text(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            return str(getattr(msg, "content", ""))
    return ""


def _latest_non_debug_ai_text(messages: List[Any]) -> str:
    """Return the latest usable AI reply for follow-up email requests."""
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "ai":
            continue
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue
        if content.startswith("[Planner]") or content.startswith("[Debug]"):
            continue
        if "confirm send?" in content.lower():
            continue
        return content
    return ""


def _task_last_answer_text(task_state: TaskState) -> str:
    if not isinstance(task_state, dict):
        return ""
    last_answer = task_state.get("last_answer", {})
    if not isinstance(last_answer, dict):
        return ""
    if not bool(last_answer.get("accepted", True)):
        return ""
    return str(last_answer.get("text", "")).strip()


def _wants_same_information(text: str) -> bool:
    normalized = (text or "").lower()
    patterns = [
        "same information",
        "same info",
        "same answer",
        "this information",
        "this info",
        "send this",
        "send it",
        "forward this",
        "forward it",
        "share this",
        "share it",
        "that information",
        "that answer",
    ]
    return any(pat in normalized for pat in patterns)


def _resolve_email_prefs(user_key: str) -> Tuple[str, str]:
    prefs = get_email_prefs(user_key)
    tone = str(prefs.get("tone") or DEFAULT_TONE)
    signature_name = str(prefs.get("signature_name") or DEFAULT_SIGNATURE_NAME)
    if prefs.get("tone") is None or prefs.get("signature_name") is None:
        set_email_prefs(user_key, {"tone": tone, "signature_name": signature_name})
    return tone, signature_name


def _greeting_line(recipient: str) -> str:
    return f"Hello {recipient}!"


def _preview_body(body: str) -> str:
    return f"{body}{AI_FOOTER}"


def _debug_payload(label: str, text: str, limit: int = 1200) -> str:
    snippet = text if text is not None else ""
    snippet = snippet.strip()
    if len(snippet) > limit:
        snippet = f"{snippet[:limit]}..."
    return f"[Debug] {label}: {snippet}"


def _resolve_sources_policy(text: str, current: bool) -> bool:
    normalized = text.lower()
    include_patterns = [
        "include sources", "add sources", "include citations", "add citations",
        "include links", "add links", "with sources", "with citations", "with links",
    ]
    exclude_patterns = [
        "no sources", "remove sources", "without sources",
        "no citations", "remove citations", "without citations",
    ]
    if any(pat in normalized for pat in include_patterns):
        return True
    if any(pat in normalized for pat in exclude_patterns):
        return False
    return current


def _strip_sources_request(text: str) -> str:
    if not text:
        return text
    cleaned = text
    patterns = [
        r"include sources", r"add sources", r"include citations", r"add citations",
        r"include links", r"add links", r"with sources", r"with citations", r"with links",
        r"no sources", r"remove sources", r"without sources",
        r"no citations", r"remove citations", r"without citations",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,.;\n\t")


def _strip_error_lines(text: str) -> str:
    if not text:
        return ""
    lines: List[str] = []
    for line in text.splitlines():
        if "WEB_SEARCH_ERROR" in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_allowed_urls(text: str) -> set:
    return set(re.findall(r"https?://\S+", text or ""))


def _strip_sources_from_text(text: str, allowed_urls: set) -> str:
    if not text:
        return ""
    def _replace_url(match: re.Match) -> str:
        url = match.group(0)
        return url if url in allowed_urls else ""

    cleaned = re.sub(r"https?://\S+", _replace_url, text)
    cleaned = re.sub(r"\[p\.\d+\]", "", cleaned)
    lines: List[str] = []
    for line in cleaned.splitlines():
        if "WEB_SEARCH_ERROR" in line:
            continue
        if re.match(r"^\s*Sources?\s*:", line, re.IGNORECASE):
            continue
        stripped = line.strip()
        if stripped == "-":
            continue
        lines.append(line.rstrip())
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
    return cleaned_text


def _sanitize_context_text(text: str, include_sources: bool) -> str:
    cleaned = _strip_error_lines(text)
    if include_sources:
        return cleaned
    return _strip_sources_from_text(cleaned, set())


def _memory_primary_fact(memory_context: str) -> str:
    if not memory_context:
        return ""
    lines = [ln.strip() for ln in memory_context.splitlines() if ln.strip()]
    bullet = ""
    for ln in lines:
        if ln.startswith("-"):
            bullet = ln
            break
    if not bullet:
        return ""
    text = re.sub(r"^-\s*\(\d+\)\s*", "", bullet).strip()
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[1].strip()
    text = _strip_sources_from_text(text, set()).strip()
    return text[:900]


def _build_context(messages, include_sources: bool, memory_context: str = "") -> str:
    blocks = []
    for m in messages:
        m_type = getattr(m, "type", "")
        content = str(getattr(m, "content", ""))
        if m_type == "tool" and content:
            cleaned = _sanitize_context_text(content, include_sources)
            if cleaned:
                blocks.append(cleaned)
        elif m_type == "ai" and content:
            lowered = content.lower()
            # Skip debug/planner/confirmation messages, include actual answers
            if lowered.startswith("[planner]") or lowered.startswith("[debug]"):
                continue
            if "confirm send?" in lowered:
                continue
            if lowered.startswith("email sent"):
                continue
            if content.startswith("FINAL"):
                cleaned = _sanitize_context_text(content, include_sources)
                if cleaned:
                    blocks.append(cleaned)
            else:
                # Include recent AI answers as context for email drafting
                cleaned = _sanitize_context_text(content, include_sources)
                if cleaned and len(cleaned) > 20:
                    blocks.append(cleaned)
    memory_text = _sanitize_context_text(memory_context, include_sources).strip()
    if memory_text:
        blocks.append(memory_text)
    # Keep context bounded to avoid overwhelming the small model
    combined = "\n\n".join(blocks) if blocks else ""
    return combined[-3000:] if len(combined) > 3000 else combined


def _extract_sources(messages) -> Tuple[List[str], List[str], str]:
    pdf_citations = set()
    web_urls = set()
    web_tool_called = False
    web_error = ""
    web_content_nonempty = False
    for m in messages:
        if getattr(m, "type", "") != "tool":
            continue
        name = getattr(m, "name", "")
        content = str(getattr(m, "content", ""))
        if name == "retrieve_context":
            for match in re.findall(r"\[p\.(\d+)\]", content):
                pdf_citations.add(f"[p.{match}]")
        if name == "tavily_search":
            web_tool_called = True
            if content.strip():
                web_content_nonempty = True
            if "WEB_SEARCH_ERROR" in content:
                _, _, err_text = content.partition("WEB_SEARCH_ERROR:")
                web_error = err_text.strip()
            for match in re.findall(r"https?://\S+", content):
                web_urls.add(match.rstrip(").,;"))
    web_failure_reason = ""
    if web_tool_called and not web_urls:
        if web_error:
            web_failure_reason = f"web search failed: {web_error}"
        elif web_content_nonempty:
            web_failure_reason = "web search returned no usable URLs"
        else:
            web_failure_reason = "web search returned no results"
    return sorted(pdf_citations), sorted(web_urls), web_failure_reason


def _sources_lines(pdf_citations: List[str], web_urls: List[str]) -> List[str]:
    lines: List[str] = []
    for citation in pdf_citations:
        lines.append(f"- {citation}")
    for url in web_urls:
        lines.append(f"- {url}")
    return lines


def _sources_block(lines: List[str], failure_reason: str) -> str:
    if lines:
        return "Sources (for you):\n" + "\n".join(lines)
    if failure_reason:
        return f"Sources (for you): None ({failure_reason})"
    return ""


def _strip_greeting_and_signature(text: str, signature_name: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.match(r"^(hello|hi|dear)\b", lines[0].strip(), re.IGNORECASE):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip().lower() == signature_name.lower():
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    if lines and re.match(r"^(thank you|thanks|regards|best|sincerely)[!.,]?$", lines[-1].strip(), re.IGNORECASE):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).strip()


def _normalize_email_body(
    raw_body: str,
    recipient: str,
    signature_name: str,
    include_sources: bool,
    sources_lines: List[str],
    allowed_urls: set,
) -> str:
    cleaned = _strip_sources_from_text(raw_body, allowed_urls)
    content = _strip_greeting_and_signature(cleaned, signature_name)
    identity_line = ASSISTANT_IDENTITY_LINE
    if content:
        lowered = content.lower()
        if "pragnesh's ai assistant" not in lowered:
            content = f"{identity_line}\n\n{content}"
    else:
        content = identity_line
    if include_sources and sources_lines:
        sources_text = "\n".join(sources_lines)
        if content:
            content = f"{content}\n\nSources:\n{sources_text}"
        else:
            content = f"Sources:\n{sources_text}"
    greeting = _greeting_line(recipient)
    if content:
        return f"{greeting}\n\n{content}\n\nThank you\n{signature_name}"
    return f"{greeting}\n\nThank you\n{signature_name}"


def _build_confirm_message(canonical: str, email: str, subject: str, tone: str, body: str, sources_block: str) -> str:
    confirm = (
        "Confirm send?\n"
        f"To: {canonical} <{email}>\n"
        f"Subject: {subject}\n"
        f"Tone: {tone}\n"
        f"Body:\n{_preview_body(body)}\n"
        "Reply yes/no."
    )
    if sources_block:
        confirm = f"{confirm}\n{sources_block}"
    return confirm


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _intent_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """/no_think
You extract email intent fields from user messages.

Return JSON only with keys: to_name, body, tone.
- to_name: canonical allowed contact name if explicitly mentioned or an alias is used.
- If the message is only a recipient name or alias, set body to "".
- If no recipient is mentioned, set to_name to "".
- tone: explicit tone request if present, otherwise "".
- Do NOT add or rephrase details from the user message.

Allowed contacts (name + aliases):
{contacts}

User message:
{message}
"""
    )


def _email_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """/no_think
You are a mail drafting assistant.

Assistant identity:
{identity}

Use the context to draft a concise email.
Requirements:
- Improve grammar and clarity
- Keep tone: {tone}
- Do NOT add new information
- Subject <= 8 words
- Body should NOT include a greeting or a signature
- Do NOT include sources, citations, or URLs unless explicitly requested
- Return JSON only with keys: subject, body
- Output must be a single JSON object, no Markdown, no code fences
- Use \\n for line breaks inside the body value

Context:
{context}

User request:
{request}
"""
    )


def _edit_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """/no_think
You are editing an email draft.
Assistant identity:
{identity}
Apply the instruction exactly, preserve all other content.
Tone: {tone}
Return JSON only with keys: subject, body (subject <= 8 words).
Output must be a single JSON object, no Markdown, no code fences.
Use \\n for line breaks inside the body value.
Body should NOT include a greeting or a signature.
Do NOT include sources, citations, or URLs unless explicitly requested.
If research context is provided, use it to update facts in the draft.
Do not claim you cannot browse/research; context is already provided.

Instruction:
{instruction}

Current email:
{body}

Research context:
{research_context}
"""
    )


# ---------------------------------------------------------------------------
# Sub-handlers (one per draft state)
# ---------------------------------------------------------------------------

def _handle_send_confirmation(
    draft: DraftState,
    user_text: str,
    user_key: str,
    chat_id: str,
    messages: List[Any],
    llm,
    identity_text: str,
    edit_prompt: ChatPromptTemplate,
    default_tone: str,
    signature_name: str,
    sources_lines: List[str],
    sources_block_text: str,
) -> Dict[str, Any]:
    """Handle confirm/cancel/edit responses when a draft is pending confirmation."""
    confirmation_intent = parse_confirmation_intent(user_text)

    if confirmation_intent == "confirm":
        try:
            draft_email = str(draft.get("to_email", "")).strip()
            with ThreadPoolExecutor(max_workers=1) as executor:
                if draft_email:
                    future = executor.submit(
                        send_email_to_address,
                        draft_email,
                        draft.get("subject", ""),
                        draft.get("body", ""),
                    )
                else:
                    future = executor.submit(
                        send_email_to_contact,
                        draft.get("to_name", ""),
                        draft.get("subject", ""),
                        draft.get("body", ""),
                    )
                message_id = future.result(timeout=max(1, int(ASYNC_TIMEOUT_EMAIL_SEC)))
            clear_draft(user_key, chat_id)
            return {"messages": [AIMessage(content=f"Email sent. (id: {message_id})")]}
        except FuturesTimeoutError:
            clear_draft(user_key, chat_id)
            return {"messages": [AIMessage(content="Error: email send timed out. Please retry.")]}
        except Exception as exc:
            clear_draft(user_key, chat_id)
            return {"messages": [AIMessage(content=f"Error: {exc}")]}

    if confirmation_intent == "decline":
        clear_draft(user_key, chat_id)
        return {"messages": [AIMessage(content="Cancelled.")]}

    # Treat anything else as an edit instruction
    current_tone = str(draft.get("tone") or default_tone)
    tone, instruction, tone_explicit = _extract_tone(user_text, current_tone)
    if tone_explicit:
        set_email_prefs(user_key, {"tone": tone})
    include_sources = _resolve_sources_policy(user_text, bool(draft.get("include_sources")))
    instruction = _strip_sources_request(instruction).strip()
    canonical = draft.get("canonical_name") or draft.get("to_name")
    current_body_clean = _strip_sources_from_text(
        str(draft.get("body", "")), _extract_allowed_urls(str(draft.get("body", "")))
    )
    current_body_content = _strip_greeting_and_signature(current_body_clean, signature_name)
    subject = str(draft.get("subject", "")).strip()
    body_content = current_body_content

    if instruction or tone_explicit:
        instruction_text = instruction or "Rewrite to match the specified tone."
        research_context = "none"
        if _instruction_requests_research(user_text):
            research_context = _tool_context_for_edit(messages)
        chain = edit_prompt | llm
        edited = chain.invoke(
            {
                "identity": identity_text,
                "instruction": instruction_text,
                "body": body_content,
                "tone": tone,
                "research_context": research_context,
            }
        )
        text = str(getattr(edited, "content", ""))
        subject, body_content, error = _extract_fields(text)
        if error:
            return {
                "messages": [
                    AIMessage(content=_debug_payload("mailer_llm_response", text)),
                    AIMessage(content=f"Could not update draft: {error}."),
                ]
            }
        subject, body_content = _apply_request_fallback(subject, body_content, instruction_text)

    allowed_urls = _extract_allowed_urls(user_text)
    allowed_urls.update(_extract_allowed_urls(str(draft.get("body", ""))))
    body = _normalize_email_body(
        body_content, str(canonical), signature_name, include_sources, sources_lines, allowed_urls
    )
    updated = {
        "pending": True,
        "to_name": draft.get("to_name"),
        "to_email": draft.get("to_email", ""),
        "canonical_name": canonical,
        "subject": subject,
        "body": body,
        "tone": tone,
        "include_sources": include_sources,
        "signature_name": signature_name,
    }
    set_draft(user_key, chat_id, updated)
    email = str(updated.get("to_email", "")).strip()
    if not email:
        match, _ = resolve_contact(updated.get("to_name", ""))
        email = str(match.get("email", "")) if match else ""
    confirm = (
        "Updated draft. Confirm send?\n"
        f"To: {canonical} <{email}>\n"
        f"Subject: {subject}\n"
        f"Tone: {tone}\n"
        f"Body:\n{_preview_body(body)}\n"
        "Reply yes/no (or send edits)."
    )
    if sources_block_text:
        confirm = f"{confirm}\n{sources_block_text}"
    return {"messages": [AIMessage(content=confirm)]}


def _handle_body_stage(
    draft: DraftState,
    messages: List[Any],
    user_text: str,
    user_key: str,
    chat_id: str,
    llm,
    identity_text: str,
    email_prompt: ChatPromptTemplate,
    signature_name: str,
    sources_lines: List[str],
    sources_block_text: str,
    memory_context: str,
) -> Dict[str, Any]:
    """Handle the case where we already know the recipient but need the body."""
    to_name = str(draft.get("to_name", ""))
    to_email = str(draft.get("to_email", "")).strip()
    canonical = str(draft.get("canonical_name", ""))
    default_tone = str(draft.get("tone") or DEFAULT_TONE)

    if not user_text.strip():
        return {"messages": [AIMessage(content="What should the email say?")]}

    email = to_email
    if not to_email:
        match, matches = resolve_contact(to_name)
        if matches:
            names = ", ".join(str(m.get("name", "")) for m in matches)
            return {"messages": [AIMessage(content=f"Multiple matches for '{to_name}': {names}")]}
        if not match:
            return {"messages": [AIMessage(content=f"No allowlisted contact found for '{to_name}'.")]}
        if not canonical:
            canonical = str(match.get("name", ""))
        email = str(match.get("email", "")).strip()
    elif not canonical:
        canonical = to_email

    tone, request_text, tone_explicit = _extract_tone(user_text, default_tone)
    if tone_explicit:
        set_email_prefs(user_key, {"tone": tone})
    include_sources = _resolve_sources_policy(user_text, bool(draft.get("include_sources")))
    request_text = _strip_sources_request(request_text)
    if not request_text.strip():
        set_draft(user_key, chat_id, {
            "to_name": to_name, "to_email": email, "canonical_name": canonical,
            "tone": tone, "stage": "body", "include_sources": include_sources,
        })
        return {"messages": [AIMessage(content="What should the email say?")]}

    context = _build_context(messages, include_sources, memory_context=memory_context)
    chain = email_prompt | llm
    draft_text = chain.invoke(
        {"identity": identity_text, "context": context, "request": request_text, "tone": tone}
    )
    text = str(getattr(draft_text, "content", ""))
    subject, body_content, error = _extract_fields(text)
    if error:
        return {
            "messages": [
                AIMessage(content=_debug_payload("mailer_llm_response", text)),
                AIMessage(content=f"Could not draft email: {error}."),
            ]
        }
    subject, body_content = _apply_request_fallback(subject, body_content, request_text)

    allowed_urls = _extract_allowed_urls(user_text)
    body = _normalize_email_body(
        body_content, canonical, signature_name, include_sources, sources_lines, allowed_urls
    )
    set_draft(user_key, chat_id, {
        "pending": True, "to_name": to_name, "to_email": email, "canonical_name": canonical,
        "subject": subject, "body": body, "tone": tone,
        "include_sources": include_sources, "signature_name": signature_name,
    })
    confirm = _build_confirm_message(canonical, email, subject, tone, body, sources_block_text)
    return {"messages": [AIMessage(content=confirm)]}


def _handle_recipient_stage(
    draft: DraftState,
    messages: List[Any],
    user_text: str,
    user_key: str,
    chat_id: str,
    llm,
    identity_text: str,
    email_prompt: ChatPromptTemplate,
    signature_name: str,
    sources_lines: List[str],
    sources_block_text: str,
    task_state: TaskState,
    memory_context: str,
) -> Dict[str, Any]:
    """Handle the case where we know body/topic but still need recipient."""
    to_name = ""
    to_email = _extract_direct_email(user_text)
    regex_name, regex_body = _match_email_intent(user_text)
    if regex_name:
        to_name = regex_name
    if not to_name and not to_email:
        scanned_name, _ = _find_contact_in_text(user_text)
        if scanned_name:
            to_name = scanned_name
    if not to_name and not to_email:
        return {"messages": [AIMessage(content="Who should I email?")]}

    email = to_email
    if to_email:
        canonical = to_email
    else:
        match, matches = resolve_contact(to_name)
        if matches:
            names = ", ".join(str(m.get("name", "")) for m in matches)
            return {"messages": [AIMessage(content=f"Multiple matches for '{to_name}': {names}")]}
        if not match:
            return {"messages": [AIMessage(content=f"No allowlisted contact found for '{to_name}'. Use /contacts to see available names.")]}
        canonical = str(match.get("name", ""))
        email = str(match.get("email", "")).strip()
    tone = str(draft.get("tone") or DEFAULT_TONE)
    include_sources = bool(draft.get("include_sources"))

    body_hint = str(draft.get("body_hint") or "").strip()
    if regex_body:
        body_hint = regex_body
    if not body_hint:
        body_hint = _task_last_answer_text(task_state)[:1200]

    if not body_hint:
        set_draft(
            user_key,
            chat_id,
            {
                "to_name": to_name,
                "to_email": email,
                "canonical_name": canonical,
                "tone": tone,
                "stage": "body",
                "include_sources": include_sources,
            },
        )
        return {"messages": [AIMessage(content="What should the email say?")]}

    request_text = _strip_sources_request(body_hint)
    context = _build_context(messages, include_sources, memory_context=memory_context)
    chain = email_prompt | llm
    draft_text = chain.invoke(
        {"identity": identity_text, "context": context, "request": request_text, "tone": tone}
    )
    text = str(getattr(draft_text, "content", ""))
    subject, body_content, error = _extract_fields(text)
    if error:
        return {
            "messages": [
                AIMessage(content=_debug_payload("mailer_llm_response", text)),
                AIMessage(content=f"Could not draft email: {error}."),
            ]
        }
    subject, body_content = _apply_request_fallback(subject, body_content, request_text)

    allowed_urls = _extract_allowed_urls(user_text)
    body = _normalize_email_body(
        body_content, canonical, signature_name, include_sources, sources_lines, allowed_urls
    )
    set_draft(
        user_key,
        chat_id,
        {
            "pending": True,
            "to_name": to_name,
            "to_email": email,
            "canonical_name": canonical,
            "subject": subject,
            "body": body,
            "tone": tone,
            "include_sources": include_sources,
            "signature_name": signature_name,
        },
    )
    confirm = _build_confirm_message(canonical, email, subject, tone, body, sources_block_text)
    return {"messages": [AIMessage(content=confirm)]}


def _handle_new_email(
    messages: List[Any],
    user_text: str,
    user_key: str,
    chat_id: str,
    llm,
    identity_text: str,
    email_prompt: ChatPromptTemplate,
    intent_prompt: ChatPromptTemplate,
    default_tone: str,
    signature_name: str,
    sources_lines: List[str],
    sources_block_text: str,
    task_state: TaskState,
    memory_context: str,
) -> Dict[str, Any]:
    """Handle a brand-new email request with no existing draft.

    Parses recipient/body intent, fills missing fields from context when safe,
    stores draft, then asks for explicit confirmation.
    """
    to_name = ""
    to_email = ""
    body = ""
    tone_from_intent = ""
    wants_same = _wants_same_information(user_text)
    wants_summary = _wants_summary_content(user_text)
    direct_email = _extract_direct_email(user_text)
    if direct_email:
        to_email = direct_email

    regex_name, regex_body = _match_email_intent(user_text)
    if regex_name:
        if EMAIL_IN_TEXT_RE.fullmatch(regex_name.strip()):
            to_email = regex_name.strip()
        else:
            to_name = regex_name
    if regex_body:
        body = regex_body

    if not to_name:
        scanned_name, _ = _find_contact_in_text(user_text)
        if scanned_name:
            to_name = scanned_name

    # LLM extraction is now a fallback only.
    if user_text.strip() and ((not to_name and not to_email) or not body):
        try:
            intent_chain = intent_prompt | llm
            intent_result = intent_chain.invoke({
                "message": user_text,
                "contacts": _contacts_for_prompt(),
            })
            intent_text = str(getattr(intent_result, "content", ""))
            llm_to_name, llm_body, llm_tone = _parse_intent_json(intent_text)
            llm_to_name = _normalize_contact_name(llm_to_name)
            if not to_name and not to_email and llm_to_name:
                if EMAIL_IN_TEXT_RE.fullmatch(llm_to_name):
                    to_email = llm_to_name
                else:
                    to_name = llm_to_name
            if not body and llm_body:
                body = llm_body
            if llm_tone:
                tone_from_intent = llm_tone
        except Exception:
            pass

    recipient_name = ""
    recipient_email = to_email.strip()
    if to_name and not recipient_email:
        match, matches = resolve_contact(to_name)
        if not match and not matches:
            to_name = ""
        elif match:
            recipient_name = str(match.get("name", "")).strip()
            recipient_email = str(match.get("email", "")).strip()
    elif recipient_email:
        recipient_name = recipient_email

    if not body and wants_same:
        latest_ai = _latest_non_debug_ai_text(messages)
        reused = _strip_sources_from_text(latest_ai, set())
        reused = reused.strip()
        if reused:
            # Keep drafts compact and avoid copying an entire long answer thread.
            body = reused[:1200]
    if wants_same or wants_summary:
        facts = _collect_fact_bundle(messages, task_state, memory_context, user_text)
        if facts:
            deterministic_content = _compose_fact_transfer_content(facts, user_text)
            if deterministic_content:
                body = deterministic_content
    if not body:
        task_last_answer = _task_last_answer_text(task_state)
        if task_last_answer and (to_name or recipient_email):
            body = task_last_answer[:1200]
    if not body and (wants_same or wants_summary):
        fallback_mem = _memory_primary_fact(memory_context)
        if fallback_mem:
            body = _compose_fact_transfer_content([fallback_mem], user_text)

    if tone_from_intent:
        tone = tone_from_intent
        tone_explicit = True
    else:
        tone, body, tone_explicit = _extract_tone(body, default_tone)
    if tone_explicit:
        set_email_prefs(user_key, {"tone": tone})
    include_sources = _resolve_sources_policy(user_text, False)

    if not to_name and not recipient_email:
        body_hint = _strip_sources_from_text(body, set()).strip()
        if not body_hint:
            body_hint = _fallback_topic_from_email_request(user_text)
        set_draft(
            user_key,
            chat_id,
            {
                "stage": "recipient",
                "body_hint": body_hint,
                "tone": tone,
                "include_sources": include_sources,
            },
        )
        return {"messages": [AIMessage(content="Who should I email?")]}

    if not recipient_email:
        match, matches = resolve_contact(to_name)
        if matches:
            names = ", ".join(str(m.get("name", "")) for m in matches)
            return {"messages": [AIMessage(content=f"Multiple matches for '{to_name}': {names}")]}
        if not match:
            return {"messages": [AIMessage(content=f"No allowlisted contact found for '{to_name}'. Use /contacts to see available names.")]}
        recipient_name = str(match.get("name", ""))
        recipient_email = str(match.get("email", "")).strip()
    if not recipient_name:
        recipient_name = recipient_email

    if not body:
        set_draft(user_key, chat_id, {
            "to_name": to_name, "to_email": recipient_email, "canonical_name": recipient_name,
            "tone": tone, "stage": "body", "include_sources": include_sources,
        })
        return {"messages": [AIMessage(content="What should the email say?")]}

    request_text = _strip_sources_request(body)
    if not request_text.strip():
        set_draft(user_key, chat_id, {
            "to_name": to_name, "to_email": recipient_email, "canonical_name": recipient_name,
            "tone": tone, "stage": "body", "include_sources": include_sources,
        })
        return {"messages": [AIMessage(content="What should the email say?")]}

    if wants_same or wants_summary:
        facts_for_subject = _collect_fact_bundle(messages, task_state, memory_context, user_text)
        subject = _fact_subject(facts_for_subject)
        body_content = request_text
    else:
        context = _build_context(messages, include_sources, memory_context=memory_context)
        chain = email_prompt | llm
        draft_text = chain.invoke(
            {"identity": identity_text, "context": context, "request": request_text, "tone": tone}
        )
        text = str(getattr(draft_text, "content", ""))
        subject, body_content, error = _extract_fields(text)
        if error:
            return {
                "messages": [
                    AIMessage(content=_debug_payload("mailer_llm_response", text)),
                    AIMessage(content=f"Could not draft email: {error}."),
                ]
            }
        subject, body_content = _apply_request_fallback(subject, body_content, request_text)

    allowed_urls = _extract_allowed_urls(user_text)
    body = _normalize_email_body(
        body_content, recipient_name, signature_name, include_sources, sources_lines, allowed_urls
    )
    set_draft(user_key, chat_id, {
        "pending": True, "to_name": to_name, "to_email": recipient_email, "canonical_name": recipient_name,
        "subject": subject, "body": body, "tone": tone,
        "include_sources": include_sources, "signature_name": signature_name,
    })
    confirm = _build_confirm_message(recipient_name, recipient_email, subject, tone, body, sources_block_text)
    return {"messages": [AIMessage(content=confirm)]}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_mailer(llm):
    email_prompt = _email_prompt()
    edit_prompt = _edit_prompt()
    intent_prompt = _intent_prompt()
    identity_text = load_identity_text()

    def mailer(state):
        # Mailer runs as a small state machine over persisted draft stages.
        messages = state.get("messages", [])
        user_key = state.get("user_key")
        chat_id = state.get("chat_id")
        if not user_key or not chat_id:
            return {"messages": [AIMessage(content="Missing user context for mailer.")]}

        draft = get_draft(user_key, chat_id)
        task_state = normalize_task_state(state.get("task_state", {}))
        task_state["active_task"] = "email"
        memory_context = str(state.get("memory_context", "") or "").strip()
        user_text = _latest_user_text(messages)
        default_tone, signature_name = _resolve_email_prefs(user_key)
        pdf_citations, web_urls, web_failure_reason = _extract_sources(messages)
        src_lines = _sources_lines(pdf_citations, web_urls)
        src_block = _sources_block(src_lines, web_failure_reason)

        result: Dict[str, Any]
        if is_no_email_only(user_text):
            clear_draft(user_key, chat_id)
            set_flags(user_key, chat_id, {"followup_reset": True})
            result = {"messages": [AIMessage(content="Okay, I won't email anyone. What would you like to do next?")]}
        elif draft.get("pending"):
            result = _handle_send_confirmation(
                draft, user_text, user_key, chat_id, messages,
                llm, identity_text, edit_prompt, default_tone, signature_name, src_lines, src_block,
            )
        elif draft.get("stage") == "body":
            result = _handle_body_stage(
                draft, messages, user_text, user_key, chat_id,
                llm, identity_text, email_prompt, signature_name, src_lines, src_block, memory_context,
            )
        elif draft.get("stage") == "recipient":
            result = _handle_recipient_stage(
                draft, messages, user_text, user_key, chat_id,
                llm, identity_text, email_prompt, signature_name, src_lines, src_block, task_state, memory_context,
            )
        else:
            result = _handle_new_email(
                messages, user_text, user_key, chat_id,
                llm, identity_text, email_prompt, intent_prompt, default_tone, signature_name, src_lines, src_block, task_state, memory_context,
            )

        # Keep `task_state.email_frame` synchronized with persisted draft slots.
        latest_draft = get_draft(user_key, chat_id)
        email_frame = task_state.get("email_frame", {})
        if not isinstance(email_frame, dict):
            email_frame = {}
        if latest_draft:
            email_frame["stage"] = str(latest_draft.get("stage", "confirm" if latest_draft.get("pending") else "")).strip()
            email_frame["recipient"] = str(
                latest_draft.get("canonical_name") or latest_draft.get("to_name") or ""
            ).strip()
            email_frame["topic"] = str(latest_draft.get("subject", "")).strip()
            email_frame["body"] = str(
                latest_draft.get("body_hint") or latest_draft.get("body") or ""
            ).strip()[:1200]
            email_frame["pending_confirmation"] = bool(latest_draft.get("pending"))
            if email_frame["recipient"]:
                task_state["last_contact"] = email_frame["recipient"]
        else:
            email_frame = {
                "stage": "",
                "recipient": "",
                "topic": "",
                "body": "",
                "pending_confirmation": False,
            }
        task_state["email_frame"] = email_frame
        if not latest_draft:
            task_state["active_task"] = "qa"

        result["task_state"] = task_state
        return result

    return mailer
