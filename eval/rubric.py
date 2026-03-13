"""Eval rubric that applies guardrail checks to a run."""

from typing import Any, Dict, List

from app.guardrails import guardrail_checks


def evaluate_question(question: str, plan: Any, messages: List[Any]) -> Dict[str, Any]:
    """Apply pass/fail guardrail checks for a single QA run."""
    return guardrail_checks(question, plan, messages)
