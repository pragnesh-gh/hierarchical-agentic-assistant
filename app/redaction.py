"""Redaction helpers for logs and tracing payloads."""

import re
from typing import Any


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*([A-Za-z0-9_\-]{8,})"
)


def redact_text(text: str) -> str:
    if not text:
        return text
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = _SECRET_RE.sub(r"\1=[REDACTED]", redacted)
    return redacted


def redact_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return redact_text(payload)
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {key: redact_payload(value) for key, value in payload.items()}
    return payload
