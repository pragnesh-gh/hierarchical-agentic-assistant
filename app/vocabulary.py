"""Single source of truth for intent-detection keyword lists and patterns.

Every module that needs to match summary/deictic markers, freshness hints,
web hints, or conversational openers should import from here instead of
maintaining its own copy.
"""

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# 1. Summary / deictic markers  (detect "send previous answer")
#    Union of: mailer_agent.SUMMARY_MARKERS, turn_controller._wants_same_answer
#    markers, and graph_memory._DEICTIC_RE phrases.
# ---------------------------------------------------------------------------

SUMMARY_MARKERS: Tuple[str, ...] = (
    "summary",
    "summarize",
    "summarise",
    "what we discussed",
    "discussed",
    "this information",
    "this answer",
    "same information",
    "same info",
    "same answer",
    "send this",
    "send it",
    "that information",
    "that answer",
    "share this",
    "share it",
    "forward this",
    "forward it",
)

DEICTIC_RE: re.Pattern = re.compile(
    r"\b(?:"
    + "|".join(re.escape(m) for m in SUMMARY_MARKERS if m not in ("summary", "summarize", "summarise", "what we discussed", "discussed"))
    + r")\b",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 2. Freshness / web hints  (detect "needs live data")
#    From: planner_agent._email_requires_research
# ---------------------------------------------------------------------------

FRESHNESS_HINTS: Tuple[str, ...] = (
    "weather",
    "time",
    "news",
    "current",
    "today",
    "latest",
    "score",
    "stocks",
    "price",
    "current affairs",
)

WEB_HINTS: Tuple[str, ...] = (
    "web",
    "internet",
    "online",
    "website",
    "websites",
    "search online",
    "look it up",
)

# ---------------------------------------------------------------------------
# 3. Conversational patterns  (detect greetings / smalltalk)
#    Union of: intent_utils._CONVERSATIONAL_STARTS and
#    guardrails.conversational_patterns.
# ---------------------------------------------------------------------------

CONVERSATIONAL_STARTS: Tuple[str, ...] = (
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "howdy",
    "greetings",
    "what's up",
    "whats up",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "got it",
    "sure",
    "alright",
    "how are you",
    "what can you do",
    "help me",
    "who are you",
    "what are you",
    "yes",
    "no",
    "y",
    "n",
)
