"""Identity/profile loader for assistant prompts."""

from pathlib import Path
from functools import lru_cache

from config import BASE_DIR


IDENTITY_PATH = BASE_DIR / "identity.md"

DEFAULT_IDENTITY = (
    "Arjun is a calm, wise, practical assistant. "
    "Prioritize correctness, clarity, and safe confirmation before actions."
)


@lru_cache(maxsize=1)
def load_identity_text(max_chars: int = 900) -> str:
    """Load a compact identity block for prompt injection."""
    path: Path = IDENTITY_PATH
    if not path.exists():
        return DEFAULT_IDENTITY
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return DEFAULT_IDENTITY
    if not text:
        return DEFAULT_IDENTITY
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text
