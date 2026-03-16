"""Guardrail checks for tool choice, citations, and plan validity."""

import json
import re
from typing import Any, Dict, List

from vocabulary import CONVERSATIONAL_STARTS


def _is_conversational_query(normalized: str) -> bool:
    """Detect greeting/small-talk queries that should not trigger tools."""
    stripped = normalized.rstrip("!?., ")
    conversational_patterns = CONVERSATIONAL_STARTS
    if stripped in conversational_patterns or any(
        normalized.startswith(p) and len(normalized) < len(p) + 15
        for p in conversational_patterns
    ):
        return True

    # Handle longer greeting forms like:
    # "hello! how are you doing today?"
    if re.match(r"^(hello|hi|hey|good (morning|afternoon|evening))\b", normalized):
        if "how are you" in normalized and not any(
            k in normalized for k in ("news", "score", "weather", "stock", "price")
        ):
            return True
        if "?" not in normalized and len(normalized.split()) <= 10:
            return True
    return False


def classify_query_source(query: str) -> str:
    """Classify a query as pdf/web/both/conversational/unknown using simple heuristics."""
    normalized = query.lower().strip()
    if "email" in normalized or "mail" in normalized or "send an email" in normalized:
        return "email"

    if _is_conversational_query(normalized):
        return "conversational"
    pdf_keywords = ["deep work", "cal newport", "newport"]
    web_keywords = [
        "current",
        "latest",
        "today",
        "now",
        "recent",
        "score",
        "news",
        "time in",
        "date",
        "t20",
    ]
    both_hints = ["compare", "both", "book and web", "book vs", "and web"]

    wants_pdf = any(key in normalized for key in pdf_keywords)
    wants_web = any(key in normalized for key in web_keywords)
    wants_both = any(key in normalized for key in both_hints)

    if wants_both or (wants_pdf and wants_web):
        return "both"
    if wants_pdf:
        return "pdf"
    if wants_web:
        return "web"
    return "unknown"


def extract_tool_names(messages: List[Any]) -> List[str]:
    """Extract tool names from tool calls and tool responses."""
    tools: List[str] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                if name:
                    tools.append(str(name))

        if getattr(msg, "type", "") == "tool":
            name = getattr(msg, "name", None)
            if name:
                tools.append(str(name))
    return tools


def final_answer_text(messages: List[Any]) -> str:
    """Return the latest AI message content as the final answer text."""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            return str(getattr(msg, "content", ""))
    return ""


def _tool_before_final_answer(messages: List[Any]) -> bool:
    """Check that at least one tool message appears before final AI answer text."""
    first_final_ai_idx = None
    tool_indexes: List[int] = []

    for idx, msg in enumerate(messages):
        msg_type = getattr(msg, "type", "")
        if msg_type == "tool":
            tool_indexes.append(idx)
            continue
        if msg_type != "ai":
            continue

        content = str(getattr(msg, "content", ""))
        if content.startswith("[Planner]") or content.startswith("[Debug]"):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        first_final_ai_idx = idx
        break

    if first_final_ai_idx is None:
        return bool(tool_indexes)
    return any(idx < first_final_ai_idx for idx in tool_indexes)


def guardrail_checks(question: str, plan: Any, messages: List[Any]) -> Dict[str, Any]:
    """Run simple pass/fail checks for planning and tool usage."""
    tool_names = set(extract_tool_names(messages))
    expected_source = classify_query_source(question)
    final_text = final_answer_text(messages)

    plan_parseable = isinstance(plan, list) and len(plan) > 0
    tool_called = bool(tool_names)

    if expected_source == "pdf":
        tool_choice_correct = "retrieve_context" in tool_names
    elif expected_source == "web":
        tool_choice_correct = "tavily_search" in tool_names
    elif expected_source == "both":
        tool_choice_correct = {
            "retrieve_context",
            "tavily_search",
        }.issubset(tool_names)
    elif expected_source == "email":
        tool_choice_correct = True
    elif expected_source == "conversational":
        tool_choice_correct = True
    else:
        tool_choice_correct = tool_called

    pdf_citations_ok = True
    web_sources_ok = True
    if expected_source != "email":
        if "retrieve_context" in tool_names:
            pdf_citations_ok = "[p." in final_text
        if "tavily_search" in tool_names:
            web_sources_ok = "http" in final_text or "www." in final_text

    if expected_source in {"pdf", "web", "both"}:
        tool_called_before_response = _tool_before_final_answer(messages)
    elif expected_source == "unknown":
        tool_called_before_response = tool_called
    else:
        tool_called_before_response = True

    return {
        "plan_parseable": plan_parseable,
        "tool_called_before_response": tool_called_before_response,
        "tool_choice_correct": tool_choice_correct,
        "pdf_citations_present": pdf_citations_ok,
        "web_sources_present": web_sources_ok,
        "expected_source": expected_source,
        "observed_tools": sorted(tool_names),
    }


def classify_failure(checks: Dict[str, Any]) -> str:
    """Classify a guardrail result into a single failure category.

    Priority order matches investigation order: plan issues first,
    then routing, then tool execution, then output formatting.
    """
    if not checks:
        return "no_checks"
    if not checks.get("plan_parseable"):
        return "plan_generation_failure"
    if not checks.get("tool_choice_correct"):
        return "routing_error"
    if not checks.get("tool_called_before_response"):
        return "tool_skip"
    if not checks.get("pdf_citations_present"):
        return "citation_missing"
    if not checks.get("web_sources_present"):
        return "source_missing"
    return "pass"


def extract_tool_outputs(messages: List[Any]) -> str:
    """Collect tool response text from message history for groundedness checking."""
    parts: List[str] = []
    for msg in messages:
        if getattr(msg, "type", "") != "tool":
            continue
        content = str(getattr(msg, "content", "")).strip()
        if content:
            parts.append(content[:600])
    return "\n---\n".join(parts)


def check_groundedness(answer_text: str, messages: List[Any], llm) -> Dict[str, Any]:
    """Judge whether every claim in the answer is supported by tool outputs.

    Uses a small, focused prompt within a tight context window.
    Returns {"verdict": "grounded"|"ungrounded"|"partial", "reason": "..."}.
    """
    tool_text = extract_tool_outputs(messages)
    if not tool_text.strip():
        return {"verdict": "no_tool_output", "reason": "No tool outputs to check against."}

    # Strip the Sources section from the answer before judging.
    answer_core = answer_text
    if "\n\nSources:\n" in answer_core:
        answer_core = answer_core.split("\n\nSources:\n", 1)[0]
    elif "\n\nSources: None" in answer_core:
        answer_core = answer_core.split("\n\nSources: None", 1)[0]
    answer_core = answer_core.strip()

    if not answer_core:
        return {"verdict": "no_answer", "reason": "Empty answer text."}

    prompt = f"""/no_think
You are a factual groundedness judge. Given SOURCE evidence and an ANSWER,
decide if every claim in the ANSWER is supported by the SOURCE.

SOURCE:
{tool_text[:700]}

ANSWER:
{answer_core[:500]}

Respond with exactly one JSON object, no other text:
{{"verdict": "grounded" or "ungrounded" or "partial", "reason": "one sentence"}}"""

    try:
        result = llm.invoke(prompt)
        raw = str(getattr(result, "content", str(result))).strip()
        # Strip think tags if present.
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[-1].strip()
        # Extract JSON from response.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
            verdict = str(parsed.get("verdict", "unknown")).lower().strip()
            if verdict not in {"grounded", "ungrounded", "partial"}:
                verdict = "unknown"
            return {
                "verdict": verdict,
                "reason": str(parsed.get("reason", ""))[:200],
            }
        return {"verdict": "parse_error", "reason": f"Could not extract JSON: {raw[:120]}"}
    except Exception as exc:
        return {"verdict": "error", "reason": str(exc)[:200]}
