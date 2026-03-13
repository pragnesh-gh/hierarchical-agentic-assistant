"""Controller that classifies the latest turn intent for routing."""

import json
import re
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from chat_sessions import normalize_task_state
from guardrails import classify_query_source
from intent_utils import (
    compact_conversation,
    detect_email_intent,
    is_new_task_intent,
    parse_confirmation_intent,
)


INTENTS = {"qa", "email", "edit_draft", "confirm_send", "retry", "reset", "smalltalk"}
RESET_SCOPES = {"none", "task_only", "chat_soft_reset"}


def _default_control() -> Dict[str, Any]:
    return {"intent": "qa", "reset_scope": "none", "use_last_answer": False, "note": ""}


def _wants_retry(text: str) -> bool:
    normalized = (text or "").lower()
    patterns = [
        "redo",
        "retry",
        "try again",
        "regenerate",
        "rewrite that",
        "rephrase that",
        "that is wrong",
        "wrong answer",
        "do that again",
        "not what i asked",
    ]
    return any(p in normalized for p in patterns)


def _wants_reset(text: str) -> str:
    normalized = (text or "").lower()
    if any(p in normalized for p in ("clear chat", "forget everything", "start fresh chat", "wipe context")):
        return "chat_soft_reset"
    if is_new_task_intent(normalized) or any(
        p in normalized
        for p in (
            "new topic",
            "new question",
            "ignore previous",
            "ignore that",
            "start over",
            "reset this",
            "reset task",
        )
    ):
        return "task_only"
    return "none"


def _wants_same_answer(text: str) -> bool:
    normalized = (text or "").lower()
    markers = (
        "same information",
        "same info",
        "same answer",
        "send this",
        "send it",
        "share this",
        "share it",
        "forward this",
        "forward it",
        "that answer",
        "that information",
    )
    return any(m in normalized for m in markers)


def _looks_like_edit(text: str) -> bool:
    normalized = (text or "").lower()
    edit_terms = (
        "edit",
        "rewrite",
        "shorter",
        "longer",
        "change",
        "update",
        "tone",
        "subject",
        "body",
        "formal",
        "casual",
        "add",
        "remove",
    )
    return any(term in normalized for term in edit_terms)


def _parse_json_payload(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _heuristic_control(latest_user: str, draft: Dict[str, Any], task_state: Dict[str, Any]) -> Dict[str, Any]:
    # Fast local classifier first: cheap, deterministic, and robust for obvious cases.
    # LLM classification later refines this, but these guards protect critical actions
    # (reset/retry/confirm_send) from prompt or parsing variance.
    normalized = (latest_user or "").strip()
    lowered = normalized.lower()
    control = _default_control()

    draft_pending = bool(draft.get("pending"))
    draft_stage = str(draft.get("stage", "")).lower().strip()
    has_draft = draft_pending or bool(draft_stage)
    has_last_answer = bool(str(task_state.get("last_answer", {}).get("text", "")).strip())

    reset_scope = _wants_reset(lowered)
    if reset_scope != "none":
        control["intent"] = "reset"
        control["reset_scope"] = reset_scope
        control["note"] = "heuristic reset"
        return control

    if _wants_retry(lowered):
        control["intent"] = "retry"
        control["note"] = "heuristic retry"
        return control

    if has_draft and detect_email_intent(lowered, email_hint=False):
        control["intent"] = "email"
        control["use_last_answer"] = has_last_answer and _wants_same_answer(lowered)
        control["note"] = "heuristic email with draft"
        return control

    if has_draft and parse_confirmation_intent(lowered) in {"confirm", "decline"}:
        control["intent"] = "confirm_send"
        control["note"] = "heuristic confirm"
        return control

    # Keep smalltalk out of costly tool/planner paths.
    source_guess = classify_query_source(lowered)
    if source_guess == "conversational":
        control["intent"] = "smalltalk"
        control["note"] = "heuristic smalltalk"
        return control

    email_intent = detect_email_intent(lowered, email_hint=False)
    if email_intent:
        control["intent"] = "email"
        control["use_last_answer"] = has_last_answer and _wants_same_answer(lowered)
        control["note"] = "heuristic email"
        return control

    if has_draft and draft_stage == "recipient":
        control["intent"] = "edit_draft"
        control["note"] = "heuristic recipient stage"
        return control

    if has_draft and (draft_stage == "body" or _looks_like_edit(lowered)):
        control["intent"] = "edit_draft"
        control["note"] = "heuristic draft edit"
        return control

    control["intent"] = "qa"
    control["use_last_answer"] = has_last_answer and _wants_same_answer(lowered)
    return control


def _controller_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """/no_think
Classify the user's latest turn for orchestration.

Return JSON only with keys:
- intent: one of ["qa","email","edit_draft","confirm_send","retry","reset","smalltalk"]
- reset_scope: one of ["none","task_only","chat_soft_reset"]
- use_last_answer: boolean
- note: short reason (max 8 words)

Rules:
- "retry/redo/try again/wrong answer/rewrite" -> intent="retry"
- "new topic/start over/ignore previous" -> intent="reset", reset_scope="task_only"
- "clear chat/forget everything" -> intent="reset", reset_scope="chat_soft_reset"
- Greeting/chit-chat -> intent="smalltalk"
- Email/send/forward/share intent -> intent="email"
- If a draft is pending and user says yes/no/cancel -> intent="confirm_send"
- If a draft exists and user edits wording/tone/subject/body -> intent="edit_draft"
- Otherwise -> intent="qa"
- Set use_last_answer=true when user implies "same/this/it/that answer/info".

Latest user text:
{latest_user}

Draft state:
{draft_state}

Task state:
{task_state}

Recent conversation:
{recent_conversation}
"""
    )


def classify_turn(
    llm: Any,
    latest_user: str,
    messages: list,
    draft: Dict[str, Any],
    task_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Return normalized control JSON for planner routing.

    Flow:
    1) Run deterministic heuristic classifier first.
    2) Ask LLM classifier as a refinement layer.
    3) Apply safety overrides so reset/retry/confirm are never weakened.
    """
    normalized_task = normalize_task_state(task_state)
    heuristic = _heuristic_control(latest_user, draft, normalized_task)
    if llm is None:
        return heuristic

    prompt = _controller_prompt()
    recent = compact_conversation(messages, max_messages=6, max_chars=220)
    try:
        chain = prompt | llm
        response = chain.invoke(
            {
                "latest_user": latest_user,
                "draft_state": json.dumps(draft or {}, ensure_ascii=True),
                "task_state": json.dumps(normalized_task, ensure_ascii=True),
                "recent_conversation": recent,
            }
        )
        text = str(getattr(response, "content", "")).strip()
        parsed = _parse_json_payload(text)
        if not parsed:
            return heuristic

        intent = str(parsed.get("intent", "")).strip().lower()
        reset_scope = str(parsed.get("reset_scope", "none")).strip().lower()
        use_last_answer = bool(parsed.get("use_last_answer", False))
        note = str(parsed.get("note", "")).strip()

        if intent not in INTENTS:
            return heuristic
        if reset_scope not in RESET_SCOPES:
            reset_scope = "none"

        out = {
            "intent": intent,
            "reset_scope": reset_scope,
            "use_last_answer": use_last_answer,
            "note": note[:80],
        }

        # Guard-rail: preserve explicit reset/retry instructions.
        heuristic_intent = str(heuristic.get("intent", "")).strip().lower()
        if heuristic_intent in {"reset", "retry"}:
            return heuristic

        # Guard-rail: keep explicit confirm safe even if model misses it.
        if heuristic_intent == "confirm_send":
            out["intent"] = "confirm_send"

        # Guard-rail: when a draft exists, do not let LLM downgrade
        # draft-edit/email heuristics to plain QA.
        draft_pending = bool(draft.get("pending"))
        draft_stage = str(draft.get("stage", "")).lower().strip()
        has_draft = draft_pending or bool(draft_stage)
        if has_draft and heuristic_intent in {"edit_draft", "email"}:
            if str(out.get("intent", "")).strip().lower() == "qa":
                return heuristic
        return out
    except Exception:
        return heuristic
