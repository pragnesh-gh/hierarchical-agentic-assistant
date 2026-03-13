"""Central configuration for paths, models, and runtime settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Keep tracing off unless explicitly enabled per run.
if os.getenv("TRACE", "0") != "1":
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ.setdefault("LANGCHAIN_HIDE_INPUTS", "true")
    os.environ.setdefault("LANGCHAIN_HIDE_OUTPUTS", "true")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, str(int(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "Deep_Work.pdf"
INDEX_DIR = DATA_DIR / "faiss_index"
SECRETS_DIR = BASE_DIR / "secrets"
CONTACTS_ALLOWLIST = DATA_DIR / "contacts_allowlist.json"

MODEL_PRESET = os.getenv("MODEL_PRESET", "production").strip().lower()
_PRESET_MODELS = {
    # Best guardrails+latency blend on this hardware (March 3, 2026 benchmark).
    "production": {
        "planner": "qwen3.5:2b",
        "researcher": "qwen3.5:2b",
        "answerer": "qwen3.5:2b",
        "mailer": "qwen3.5:2b",
    },
    # Higher-capacity bias, but may be unstable on low-VRAM laptops.
    "max_accuracy": {
        "planner": "qwen3.5:4b",
        "researcher": "qwen3.5:4b",
        "answerer": "qwen3.5:4b",
        "mailer": "qwen3.5:4b",
    },
    # Fastest local fallback.
    "speed": {
        "planner": "qwen3:1.7b",
        "researcher": "qwen3:1.7b",
        "answerer": "qwen3:1.7b",
        "mailer": "qwen3:1.7b",
    },
}
_preset = _PRESET_MODELS.get(MODEL_PRESET, _PRESET_MODELS["production"])

PLANNER_MODEL = os.getenv("PLANNER_MODEL", _preset["planner"])
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", _preset["researcher"])
ANSWERER_MODEL = os.getenv("ANSWERER_MODEL", _preset["answerer"])
MAILER_MODEL = os.getenv("MAILER_MODEL", _preset["mailer"])

EMBED_MODEL = "nomic-embed-text:latest"
ALT_EMBED_MODEL = "mxbai-embed-large:latest"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Conservative defaults for 4GB VRAM + 16GB RAM laptops.
# Slightly higher context by default to support explicit task-state memory.
NUM_CTX = _env_int("NUM_CTX", 3072)
NUM_THREAD = _env_optional_int("NUM_THREAD")
NUM_PREDICT = _env_optional_int("NUM_PREDICT")
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "20m")
DISABLE_STREAMING = _env_bool("DISABLE_STREAMING", False)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.8"))
TOP_K = _env_int("TOP_K", 20)
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.1"))

# Async feature flags: keep LLM path sequential; async only on I/O-heavy edges.
ASYNC_TOOLS = _env_bool("ASYNC_TOOLS", True)
ASYNC_PERSIST = _env_bool("ASYNC_PERSIST", True)
TELEGRAM_PROGRESS = _env_bool("TELEGRAM_PROGRESS", True)
ASYNC_TIMEOUT_WEB_SEC = _env_int("ASYNC_TIMEOUT_WEB_SEC", 20)
ASYNC_TIMEOUT_PDF_SEC = _env_int("ASYNC_TIMEOUT_PDF_SEC", 15)
ASYNC_TIMEOUT_EMAIL_SEC = _env_int("ASYNC_TIMEOUT_EMAIL_SEC", 30)

# Per-role context windows (planner stays lower for responsiveness).
PLANNER_NUM_CTX = _env_int("PLANNER_NUM_CTX", min(NUM_CTX, 2048))
RESEARCHER_NUM_CTX = _env_int("RESEARCHER_NUM_CTX", NUM_CTX)
ANSWERER_NUM_CTX = _env_int("ANSWERER_NUM_CTX", NUM_CTX)
MAILER_NUM_CTX = _env_int("MAILER_NUM_CTX", NUM_CTX)

# Inline groundedness judge settings.
# Default: reuse the answerer model so Ollama serves from VRAM without a swap.
# Override via GROUNDEDNESS_MODEL env var to use a separate (larger) judge model.
GROUNDEDNESS_MODEL = os.getenv("GROUNDEDNESS_MODEL", "")
GROUNDEDNESS_NUM_CTX = _env_int("GROUNDEDNESS_NUM_CTX", 1024)
GROUNDEDNESS_TEMPERATURE = _env_float("GROUNDEDNESS_TEMPERATURE", 0.1)

PLANNER_REASONING = _env_bool("PLANNER_REASONING", False)
RESEARCHER_REASONING = _env_bool("RESEARCHER_REASONING", False)
ANSWERER_REASONING = _env_bool("ANSWERER_REASONING", False)
MAILER_REASONING = _env_bool("MAILER_REASONING", False)

# Graph long-term memory (Graphiti-first, local JSON-graph fallback).
GRAPH_MEMORY_ENABLED = _env_bool("GRAPH_MEMORY_ENABLED", True)
GRAPH_MEMORY_BACKEND = os.getenv("GRAPH_MEMORY_BACKEND", "auto").strip().lower()
GRAPH_MEMORY_TOP_K = _env_int("GRAPH_MEMORY_TOP_K", 6)
GRAPH_MEMORY_MAX_CHARS = _env_int("GRAPH_MEMORY_MAX_CHARS", 900)
GRAPH_MEMORY_MAX_FACTS_PER_USER = _env_int("GRAPH_MEMORY_MAX_FACTS_PER_USER", 800)
GRAPH_MEMORY_MIN_SCORE = _env_float("GRAPH_MEMORY_MIN_SCORE", 0.08)
GRAPH_MEMORY_PATH = DATA_DIR / os.getenv("GRAPH_MEMORY_FILE", "graph_memory.json").strip()

# Optional Graphiti backend settings.
GRAPHITI_URI = os.getenv("GRAPHITI_URI", "").strip()
GRAPHITI_USER = os.getenv("GRAPHITI_USER", "").strip()
GRAPHITI_PASSWORD = os.getenv("GRAPHITI_PASSWORD", "").strip()
