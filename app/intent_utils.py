"""Shared intent detection and message utility helpers."""

import re
from typing import List, TYPE_CHECKING

from vocabulary import CONVERSATIONAL_STARTS

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


# ---------------------------------------------------------------------------
# Message utilities (shared across planner, researcher, answerer)
# ---------------------------------------------------------------------------

def latest_human_text(messages: "List[BaseMessage]") -> str:
    """Return the content of the most recent HumanMessage."""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            return str(getattr(msg, "content", ""))
    return ""


def previous_human_text(messages: "List[BaseMessage]") -> str:
    """Return the content of the second-most-recent HumanMessage."""
    found_latest = False
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "human":
            continue
        if not found_latest:
            found_latest = True
            continue
        return str(getattr(msg, "content", ""))
    return ""


def _compact_text(text: str, max_chars: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3] + "..."


def compact_conversation(
    messages: "List[BaseMessage]",
    max_messages: int = 8,
    max_chars: int = 320,
) -> str:
    """Return a compact text view of recent messages for prompt injection."""
    recent = messages[-max_messages:] if max_messages > 0 else messages
    lines: List[str] = []
    for msg in recent:
        msg_type = getattr(msg, "type", "")
        content = str(getattr(msg, "content", ""))
        if msg_type not in {"human", "ai", "tool"}:
            continue
        if msg_type == "ai" and (
            content.startswith("[Planner]") or content.startswith("[Debug]")
        ):
            continue
        name = getattr(msg, "name", "") if msg_type == "tool" else ""
        prefix = f"{msg_type}:{name}" if name else msg_type
        lines.append(f"{prefix}: {_compact_text(content, max_chars)}")
    return "\n".join(lines)


# Patterns used to detect conversational / greeting messages so they are
# never silently merged with a previous research query.
_CONVERSATIONAL_STARTS = CONVERSATIONAL_STARTS


def _is_conversational_message(text: str) -> bool:
    """Return True if text looks like a greeting or simple conversational message."""
    normalized = text.lower().strip()
    stripped = normalized.rstrip("!?., ")
    if stripped in _CONVERSATIONAL_STARTS:
        return True
    if any(
        normalized.startswith(p) and len(normalized) < len(p) + 15
        for p in _CONVERSATIONAL_STARTS
    ):
        return True
    # Longer forms like: "hello! how are you doing today?"
    if re.match(r"^(hello|hi|hey|good (morning|afternoon|evening))\b", normalized):
        if "how are you" in normalized and not any(
            key in normalized for key in ("news", "score", "weather", "stock", "price")
        ):
            return True
    return False


def is_followup_request(text: str) -> bool:
    """Return True if the text looks like a short follow-up rather than a new question."""
    normalized = text.lower().strip()
    if not normalized:
        return False
    followup_phrases = [
        "use the web",
        "use web",
        "web search",
        "research",
        "sources",
        "citations",
        "links",
        "another draft",
        "new draft",
        "rewrite",
        "revise",
        "edit",
        "update",
        "same as",
        "as before",
        "again",
    ]
    if any(p in normalized for p in followup_phrases):
        return True
    return len(normalized.split()) <= 5


def effective_query(
    messages: "List[BaseMessage]",
    followup_reset: bool = False,
    email_hint: bool = False,
) -> str:
    """
    Return the query text the agents should act on.

    If the latest human message looks like a follow-up modifier (e.g. "use the
    web", "rewrite it") and the previous message is a proper question, the two
    are merged so all context is preserved.
    """
    latest = latest_human_text(messages)
    previous = previous_human_text(messages)
    if latest and previous:
        # Never merge a greeting/conversational message with a previous query;
        # doing so would make "hello!" look like a follow-up research request.
        if not _is_conversational_message(latest) and should_merge_followup(
            latest, previous, followup_reset, email_hint, is_followup_request(latest)
        ):
            return f"{previous} {latest}".strip()
    return latest


# ---------------------------------------------------------------------------
# Intent patterns
# ---------------------------------------------------------------------------

_EMAIL_PATTERNS = [
    r"\bemail\b",
    r"\bmail\b",
    r"send\s+an?\s+email",
    r"send\s+an?\s+mail",
    r"compose\s+an?\s+email",
    r"draft\s+an?\s+email",
    r"shoot\s+an?\s+email",
    r"drop\s+an?\s+email",
    r"send\s+.+\s+an?\s+email",
    # Follow-up style email requests without explicit "email":
    r"\bsend\s+(?:this|it|that|the same information|same information|same info)\s+to\b",
    r"\bforward\s+(?:this|it|that|the same information|same information|same info)\s+to\b",
    r"\bshare\s+(?:this|it|that|the same information|same information|same info)\s+with\b",
]

_NO_EMAIL_PATTERNS = [
    r"\bdo not email\b",
    r"\bdon't email\b",
    r"\bdo not mail\b",
    r"\bdon't mail\b",
    r"\bno emails?\b",
    r"\bno mail\b",
    r"\bwithout emailing\b",
    r"\bwithout email\b",
    r"\bwithout mail\b",
    r"\bnot (?:send|sending) (?:an? )?(?:email|mail)\b",
    r"\bdo not send (?:an? )?(?:email|mail)\b",
    r"\bdon't send (?:an? )?(?:email|mail)\b",
    r"\bstop (?:emailing|mailing|sending emails|sending mail)\b",
    r"\bcancel (?:the )?(?:email|mail)\b",
    r"\bskip (?:the )?(?:email|mail)\b",
    r"\bavoid (?:email|mail)\b",
    r"\bnever email\b",
    r"\bnot (?:email|mail) (?:anyone|anybody|someone|them)?\b",
    r"\bdon't want to (?:email|mail)\b",
    r"\bdo not want to (?:email|mail)\b",
]

_NO_EMAIL_FILLER = [
    r"\bplease\b",
    r"\banyone\b",
    r"\banybody\b",
    r"\bjust\b",
    r"\bfor now\b",
    r"\bthanks\b",
    r"\bthank you\b",
]

_NEW_TASK_PHRASES = [
    "new request",
    "new task",
    "start over",
    "start a new",
    "different question",
    "something else",
    "switch topics",
    "ignore that",
    "forget that",
    "never mind",
    "scratch that",
]


def detect_no_email_intent(text: str) -> bool:
    normalized = (text or "").lower()
    if not normalized:
        return False
    return any(re.search(pat, normalized) for pat in _NO_EMAIL_PATTERNS)


def detect_email_intent(text: str, email_hint: bool = False) -> bool:
    if detect_no_email_intent(text):
        return False
    if email_hint:
        return True
    normalized = (text or "").lower()
    if not normalized:
        return False
    return any(re.search(pat, normalized) for pat in _EMAIL_PATTERNS)


def is_new_task_intent(text: str) -> bool:
    normalized = (text or "").lower().strip()
    if not normalized:
        return False
    return any(phrase in normalized for phrase in _NEW_TASK_PHRASES)


def strip_no_email_intent(text: str) -> str:
    cleaned = (text or "")
    for pattern in _NO_EMAIL_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def is_no_email_only(text: str) -> bool:
    if not detect_no_email_intent(text):
        return False
    cleaned = strip_no_email_intent(text)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    for filler in _NO_EMAIL_FILLER:
        cleaned = re.sub(filler, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return len(cleaned.split()) == 0


def should_merge_followup(
    latest: str,
    previous: str,
    followup_reset: bool,
    email_hint: bool,
    is_followup: bool,
) -> bool:
    if not latest or not previous:
        return False
    if followup_reset:
        return False
    if not is_followup:
        return False
    if detect_no_email_intent(latest) or is_new_task_intent(latest):
        return False
    if not email_hint:
        prev_email = detect_email_intent(previous, email_hint=False)
        latest_email = detect_email_intent(latest, email_hint=False)
        if prev_email and not latest_email:
            return False
    return True


# ---------------------------------------------------------------------------
# Email confirmation parsing
# ---------------------------------------------------------------------------

_CONFIRM_DECLINE_PATTERNS = (
    r"\b(?:no|nope|nah)\b",
    r"\b(?:don't|do not)\s+send\b",
    r"\bnot\s+now\b",
    r"\blater\b",
    r"\bhold\s+off\b",
    r"\bcancel(?:\s+it)?\b",
    r"\bstop\b",
    r"\bskip\b",
    r"\bexit\b",
)

_CONFIRM_ACCEPT_PATTERNS = (
    r"\bgo\s+ahead(?:\s+and)?(?:\s+send)?\b",
    r"\bplease\s+send(?:\s+it|\s+this)?\b",
    r"\bsend\s+(?:it|this)(?:\s+now)?\b",
    r"\bsend\s+now\b",
    r"\blooks?\s+good(?:\s*,?\s*send(?:\s+it)?)?\b",
    r"\bsounds?\s+good(?:\s*,?\s*send(?:\s+it)?)?\b",
    r"\bapproved?\b",
    r"\bconfirm(?:ed)?\b",
)

_EDIT_CUE_PATTERNS = (
    r"\bedit\b",
    r"\bchange\b",
    r"\bupdate\b",
    r"\brewrite\b",
    r"\bshorter\b",
    r"\blonger\b",
    r"\btone\b",
    r"\bsubject\b",
    r"\bbody\b",
    r"\badd\b",
    r"\bremove\b",
)


def parse_confirmation_intent(text: str) -> str:
    """Return one of: 'confirm', 'decline', or 'unknown'."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return "unknown"

    if any(re.search(pattern, normalized) for pattern in _CONFIRM_DECLINE_PATTERNS):
        return "decline"

    has_edit_cue = any(re.search(pattern, normalized) for pattern in _EDIT_CUE_PATTERNS)
    if has_edit_cue:
        if re.search(r"\b(?:please\s+)?send\s+(?:it|this)(?:\s+now)?\b", normalized):
            return "confirm"
        return "unknown"

    if any(re.search(pattern, normalized) for pattern in _CONFIRM_ACCEPT_PATTERNS):
        return "confirm"

    if re.fullmatch(
        r"(yes|y|yeah|yep|yup|sure|ok|okay|alright|fine)(?:\s+please)?[.! ]*",
        normalized,
    ):
        return "confirm"

    return "unknown"
