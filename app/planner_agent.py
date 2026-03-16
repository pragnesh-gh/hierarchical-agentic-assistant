"""Planner agent that emits a validated JSON plan and routes steps."""

# lab4/planner_agent.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, BaseMessage

import json
import re
from typing import Any, Dict, List

from state import AgentState, EmailFrame
from chat_sessions import clear_draft, normalize_task_state, set_flags
from guardrails import classify_query_source
from identity import load_identity_text
from intent_utils import (
    detect_email_intent,
    detect_no_email_intent,
    is_new_task_intent,
    parse_confirmation_intent,
    compact_conversation,
    latest_human_text,
    effective_query,
)
from turn_controller import classify_turn
from vocabulary import FRESHNESS_HINTS, WEB_HINTS


ALLOWED_ACTIONS = {"researcher", "answerer", "mailer"}
ALLOWED_TOOLS = {"retrieve_context", "tavily_search"}
MIN_STEPS = 2
MIN_STEPS_EMAIL = 1
MAX_STEPS = 6


def _empty_email_frame() -> EmailFrame:
    return {
        "stage": "",
        "recipient": "",
        "topic": "",
        "body": "",
        "pending_confirmation": False,
    }


def _detect_email_intent(query: str, email_hint: bool = False) -> bool:
    return detect_email_intent(query, email_hint=email_hint)


def _is_confirmation_response(text: str) -> bool:
    return parse_confirmation_intent(text) in {"confirm", "decline"}


def _has_recent_confirm_prompt(messages: List[BaseMessage]) -> bool:
    """True only when a recent assistant turn explicitly asked for send confirmation."""
    seen = 0
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "ai":
            continue
        content = str(getattr(msg, "content", "")).lower()
        if not content or content.startswith("[planner]") or content.startswith("[debug]"):
            continue
        seen += 1
        if "confirm send?" in content or "reply yes/no" in content:
            return True
        if seen >= 2:
            break
    return False


def _email_requires_research(query: str) -> bool:
    normalized = query.lower()
    if any(h in normalized for h in FRESHNESS_HINTS):
        return True
    if any(h in normalized for h in WEB_HINTS):
        return True
    # A bare year often appears in event reminders and should not force web.
    if "use the web" in normalized or "use the internet" in normalized:
        return True
    return False


def _is_draft_update_request(text: str) -> bool:
    normalized = (text or "").lower().strip()
    if not normalized:
        return False
    strong_markers = (
        "update draft",
        "update the draft",
        "edit draft",
        "edit the draft",
        "revise draft",
        "revise the draft",
        "rewrite draft",
        "rewrite the draft",
    )
    if any(marker in normalized for marker in strong_markers):
        return True
    has_edit = any(term in normalized for term in ("update", "edit", "revise", "rewrite", "change"))
    has_draft_word = "draft" in normalized
    return has_edit and has_draft_word


def _fallback_plan_for_query(query: str, email_hint: bool = False) -> List[Dict[str, Any]]:
    """Return a safe default plan when planner output is missing/invalid.

    This keeps runtime robust even if model formatting drifts.
    """
    source = classify_query_source(query)
    if source == "conversational":
        return [{"id": 0, "action": "answerer", "description": "Respond directly."}]
    if _detect_email_intent(query, email_hint):
        if _email_requires_research(query):
            return [
                {
                    "id": 0,
                    "action": "researcher",
                    "description": "Use web search for up-to-date context.",
                    "tools": ["tavily_search"],
                },
                {
                    "id": 1,
                    "action": "mailer",
                    "description": "Draft and send the email.",
                },
            ]
        return [
            {
                "id": 0,
                "action": "mailer",
                "description": "Draft and send the email.",
            }
        ]
    if source == "pdf":
        return [
            {
                "id": 0,
                "action": "researcher",
                "description": "Use PDF RAG for book context.",
                "tools": ["retrieve_context"],
            },
            {
                "id": 1,
                "action": "answerer",
                "description": "Synthesize final answer with sources.",
            },
        ]
    if source == "web":
        return [
            {
                "id": 0,
                "action": "researcher",
                "description": "Use web search for up-to-date context.",
                "tools": ["tavily_search"],
            },
            {
                "id": 1,
                "action": "answerer",
                "description": "Synthesize final answer with sources.",
            },
        ]
    return [
        {
            "id": 0,
            "action": "researcher",
            "description": "Gather book + web context.",
            "tools": ["retrieve_context", "tavily_search"],
        },
        {
            "id": 1,
            "action": "answerer",
            "description": "Synthesize final answer with sources.",
        },
    ]


def _repair_plan(steps: Any, query: str, email_hint: bool = False) -> List[Dict[str, Any]]:
    # Normalize arbitrary planner output into executable step objects.
    if not isinstance(steps, list):
        return _fallback_plan_for_query(query, email_hint)

    email_intent = _detect_email_intent(query, email_hint)
    source = classify_query_source(query)
    conversational = source == "conversational"
    if conversational:
        return [{"id": 0, "action": "answerer", "description": "Respond directly."}]
    if email_intent:
        if _email_requires_research(query):
            required_tools = {"tavily_search"}
        else:
            required_tools = set()
    else:
        if source == "pdf":
            required_tools = {"retrieve_context"}
        elif source == "web":
            required_tools = {"tavily_search"}
        else:
            required_tools = {"retrieve_context", "tavily_search"}

    repaired: List[Dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "")).lower().strip()
        if action not in ALLOWED_ACTIONS:
            continue
        if action == "mailer" and not email_intent:
            continue
        description = str(step.get("description", "")).strip()
        new_step: Dict[str, Any] = {"action": action, "description": description}

        if action == "researcher":
            tool_list: List[str] = []
            raw_tools = step.get("tools")
            if isinstance(raw_tools, list):
                for t in raw_tools:
                    tool_name = str(t).strip()
                    if tool_name in ALLOWED_TOOLS and tool_name in required_tools:
                        tool_list.append(tool_name)
            if not tool_list:
                if required_tools == {"tavily_search"}:
                    tool_list = ["tavily_search"]
                elif required_tools == {"retrieve_context"}:
                    tool_list = ["retrieve_context"]
                elif required_tools:
                    tool_list = ["retrieve_context"]
            new_step["tools"] = tool_list

        if action == "mailer":
            new_step.pop("tools", None)

        repaired.append(new_step)

    if not repaired:
        return _fallback_plan_for_query(query)

    if not email_intent and repaired[0].get("action") != "researcher":
        repaired.insert(
            0,
            {
                "action": "researcher",
                "description": "Gather context from tools.",
                "tools": ["retrieve_context"] if "retrieve_context" in required_tools else ["tavily_search"],
            },
        )

    if email_intent:
        if repaired[-1].get("action") != "mailer":
            repaired.append(
                {
                    "action": "mailer",
                    "description": "Draft and send the email.",
                }
            )
    elif repaired[-1].get("action") != "answerer":
        repaired.append(
            {
                "action": "answerer",
                "description": "Synthesize final answer with sources.",
            }
        )

    present_tools = set()
    for step in repaired:
        if step.get("action") == "researcher":
            for tool in step.get("tools", []):
                present_tools.add(str(tool))

    missing_tools = required_tools - present_tools
    if missing_tools:
        first_researcher_idx = -1
        for idx, step in enumerate(repaired):
            if step.get("action") == "researcher":
                first_researcher_idx = idx
                break

        # Prefer adding all missing tools to the first researcher step so
        # the research tool node can fan out PDF + web in parallel.
        if first_researcher_idx >= 0:
            existing = repaired[first_researcher_idx].get("tools", [])
            existing_list = existing if isinstance(existing, list) else []
            merged_tools = list(dict.fromkeys([str(t) for t in existing_list] + sorted(missing_tools)))
            repaired[first_researcher_idx]["tools"] = merged_tools
        else:
            for tool in sorted(missing_tools):
                insert_idx = -1 if not email_intent else max(len(repaired) - 1, 0)
                repaired.insert(
                    insert_idx,
                    {
                        "action": "researcher",
                        "description": "Gather additional context from tools.",
                        "tools": [tool],
                    },
                )

    min_steps = MIN_STEPS_EMAIL if email_intent else MIN_STEPS
    while len(repaired) < min_steps:
        if required_tools:
            fill_tool = next(iter(required_tools))
            repaired.insert(
                -1,
                {
                    "action": "researcher",
                    "description": "Gather additional context for completeness.",
                    "tools": [fill_tool],
                },
            )
        else:
            repaired.insert(
                0,
                {
                    "action": "mailer",
                    "description": "Draft and send the email.",
                },
            )
            break

    if len(repaired) > MAX_STEPS:
        trimmed = repaired[: MAX_STEPS]
        if email_intent:
            if trimmed[-1].get("action") != "mailer":
                trimmed[-1] = {
                    "action": "mailer",
                    "description": "Draft and send the email.",
                }
        else:
            if trimmed[-1].get("action") != "answerer":
                trimmed[-1] = {
                    "action": "answerer",
                    "description": "Synthesize final answer with sources.",
                }
        repaired = trimmed

    for idx, step in enumerate(repaired):
        step["id"] = idx

    return repaired


def _validate_plan(plan: Any, query: str, email_hint: bool = False) -> bool:
    """Validate the planner JSON against required structure and tool rules."""
    if not isinstance(plan, list):
        return False
    email_intent = _detect_email_intent(query, email_hint)
    source = classify_query_source(query)
    conversational = source == "conversational"
    if conversational:
        min_steps = 1
    elif email_intent:
        min_steps = MIN_STEPS_EMAIL
    else:
        min_steps = MIN_STEPS
    if not (min_steps <= len(plan) <= MAX_STEPS):
        return False

    for step in plan:
        if not isinstance(step, dict):
            return False
        action = str(step.get("action", "")).lower().strip()
        if action not in ALLOWED_ACTIONS:
            return False
        if not email_intent and action == "mailer":
            return False

        tools = step.get("tools")
        if action == "researcher":
            if not isinstance(tools, list) or not tools:
                return False
            tool_names = {str(t).strip() for t in tools}
            if not tool_names.issubset(ALLOWED_TOOLS):
                return False
        if action == "answerer" and tools:
            return False
        if action == "mailer" and tools:
            return False

    if email_intent:
        if str(plan[-1].get("action", "")).lower().strip() != "mailer":
            return False
    elif conversational:
        if str(plan[0].get("action", "")).lower().strip() != "answerer":
            return False
    else:
        if str(plan[0].get("action", "")).lower().strip() != "researcher":
            return False
        if str(plan[-1].get("action", "")).lower().strip() != "answerer":
            return False
    return True


def _resolve_intent_and_state(
    llm,
    latest_user_text: str,
    messages: List[BaseMessage],
    draft: Dict[str, Any],
    task_state: Dict[str, Any],
    user_key,
    chat_id,
    followup_reset: bool,
) -> Dict[str, Any]:
    """Classify intent and apply state mutations (retry, reset, email, etc.).

    Returns a dict with keys: turn_intent, reset_scope, use_last_answer,
    skip_followup, has_active_draft, task_state, draft.
    """
    control = classify_turn(llm, latest_user_text, messages, draft, task_state)
    turn_intent = str(control.get("intent", "qa")).strip().lower()
    reset_scope = str(control.get("reset_scope", "none")).strip().lower()
    use_last_answer = bool(control.get("use_last_answer", False))

    no_email_intent = detect_no_email_intent(latest_user_text)
    new_task_intent = is_new_task_intent(latest_user_text)
    skip_followup = followup_reset or no_email_intent or new_task_intent

    has_active_draft = isinstance(draft, dict) and bool(draft.get("pending") or draft.get("stage"))
    if has_active_draft and _is_draft_update_request(latest_user_text):
        turn_intent = "edit_draft"
        task_state["active_task"] = "email"

    if turn_intent in {"email", "edit_draft", "confirm_send"}:
        skip_followup = True

    if turn_intent in {"retry", "reset"}:
        skip_followup = True

    if turn_intent == "retry":
        last_answer_text = str(task_state.get("last_answer", {}).get("text", "")).strip()
        if last_answer_text:
            rejected = task_state.get("rejected_answers", [])
            if not isinstance(rejected, list):
                rejected = []
            rejected.append({"text": last_answer_text[:1800], "reason": "retry"})
            task_state["rejected_answers"] = rejected[-10:]
        task_state["last_answer"] = {"text": "", "sources": [], "accepted": False}
        task_state["active_task"] = "qa"

    if turn_intent == "reset":
        task_state["active_task"] = ""
        task_state["email_frame"] = _empty_email_frame()
        if reset_scope == "chat_soft_reset":
            task_state["last_answer"] = {"text": "", "sources": [], "accepted": True}
            task_state["rejected_answers"] = []
            task_state["last_contact"] = ""
        if user_key and chat_id:
            clear_draft(user_key, chat_id)
        draft = {}

    if turn_intent in {"email", "edit_draft", "confirm_send"}:
        task_state["active_task"] = "email"
        if use_last_answer:
            last_answer_text = str(task_state.get("last_answer", {}).get("text", "")).strip()
            if last_answer_text:
                email_frame = task_state.get("email_frame", _empty_email_frame())
                if not isinstance(email_frame, dict):
                    email_frame = _empty_email_frame()
                if not str(email_frame.get("body", "")).strip():
                    email_frame["body"] = last_answer_text[:1200]
                task_state["email_frame"] = email_frame
    elif turn_intent == "smalltalk":
        task_state["active_task"] = "smalltalk"
    else:
        task_state["active_task"] = "qa"

    return {
        "turn_intent": turn_intent,
        "reset_scope": reset_scope,
        "use_last_answer": use_last_answer,
        "skip_followup": skip_followup,
        "has_active_draft": has_active_draft,
        "task_state": task_state,
        "draft": draft,
    }


def _handle_active_draft(
    state: AgentState,
    latest_user_text: str,
    messages: List[BaseMessage],
    turn_intent: str,
    task_state: Dict[str, Any],
    has_active_draft: bool,
    step_index: int,
):
    """Handle draft fast-paths when a draft is active.

    Returns (early_return_dict, defer_draft, email_hint) where early_return_dict
    is a complete AgentState response if we should return early, or None.
    """
    if step_index >= 0 or not has_active_draft:
        email_hint = turn_intent in {"email", "edit_draft", "confirm_send"}
        return None, False, email_hint

    if turn_intent == "confirm_send" and _is_confirmation_response(latest_user_text):
        if _has_recent_confirm_prompt(messages):
            debug_msg = AIMessage(content="[Planner] Draft pending, routing to mailer.")
            return {
                "next": "mailer",
                "plan": state.get("plan", []),
                "step_index": state.get("step_index", -1),
                "messages": [debug_msg],
                "task_state": task_state,
                "turn_intent": turn_intent,
                "planner_reasoning": "[fast-path] draft_pending_confirmation",
            }, False, False
        return None, True, False

    if turn_intent in {"email", "edit_draft"}:
        if _email_requires_research(latest_user_text):
            plan = [
                {
                    "id": 0,
                    "action": "researcher",
                    "description": "Use web search for up-to-date context.",
                    "tools": ["tavily_search"],
                },
                {
                    "id": 1,
                    "action": "mailer",
                    "description": "Update and confirm the email draft.",
                },
            ]
            debug_msg = AIMessage(
                content="[Planner] Draft pending; routing via researcher then mailer."
            )
            return {
                "next": "researcher",
                "plan": plan,
                "step_index": 0,
                "messages": [debug_msg],
                "task_state": task_state,
                "turn_intent": turn_intent,
                "planner_reasoning": "[fast-path] draft_update_with_research",
            }, False, False
        debug_msg = AIMessage(content="[Planner] Draft pending, routing to mailer.")
        return {
            "next": "mailer",
            "plan": state.get("plan", []),
            "step_index": state.get("step_index", -1),
            "messages": [debug_msg],
            "task_state": task_state,
            "turn_intent": turn_intent,
            "planner_reasoning": "[fast-path] draft_pending_to_mailer",
        }, False, False

    return None, True, False


def _generate_plan(
    llm,
    prompt,
    identity_text: str,
    user_query: str,
    turn_intent: str,
    use_last_answer: bool,
    email_hint: bool,
    memory_context: str,
    messages: List[BaseMessage],
) -> tuple:
    """Generate or select a plan for the current query.

    Returns (plan, planner_reasoning_raw, state_messages).
    """
    source_guess = classify_query_source(user_query)
    control_hint = f"intent={turn_intent}; use_last_answer={use_last_answer}"

    # Fast-path high-confidence routing to reduce planner latency.
    if turn_intent == "smalltalk" or source_guess == "conversational":
        plan = [{"id": 0, "action": "answerer", "description": "Respond directly."}]
        return plan, "[fast-path] conversational/smalltalk detected", []

    if turn_intent in {"email", "edit_draft", "confirm_send"} or _detect_email_intent(user_query, email_hint):
        plan = _fallback_plan_for_query(user_query, email_hint=True)
        return plan, "[fast-path] email intent detected", []

    if source_guess in {"pdf", "web", "both"}:
        plan = _fallback_plan_for_query(user_query, email_hint=False)
        return plan, f"[fast-path] source={source_guess}", []

    # Ambiguous query: ask planner LLM.
    chain = prompt | llm | StrOutputParser()
    raw_plan = chain.invoke(
        {
            "identity": identity_text,
            "query": user_query,
            "control_hint": control_hint,
            "memory_context": memory_context,
            "messages": compact_conversation(messages),
        }
    )

    try:
        raw_text = raw_plan.strip()
        if raw_text.startswith("{") and raw_text.endswith("}"):
            plan_obj = json.loads(raw_text)
        else:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            plan_obj = json.loads(raw_text[start : end + 1])
        steps = plan_obj.get("steps", [])
        repaired = _repair_plan(steps, user_query, email_hint)
        if _validate_plan(repaired, user_query, email_hint):
            return repaired, raw_plan, []
        plan = _fallback_plan_for_query(
            user_query,
            email_hint=turn_intent in {"email", "edit_draft", "confirm_send"} or email_hint,
        )
        debug_msg = AIMessage(
            content=(
                "[Planner] Invalid JSON plan structure, using fallback. "
                f"Raw: {raw_plan}"
            )
        )
        return plan, raw_plan, [debug_msg]
    except Exception as e:
        plan = _fallback_plan_for_query(
            user_query,
            email_hint=turn_intent in {"email", "edit_draft", "confirm_send"} or email_hint,
        )
        debug_msg = AIMessage(
            content=(
                "[Planner] Failed to parse JSON plan, using fallback. "
                f"Error: {e}. Raw: {raw_plan}"
            )
        )
        return plan, raw_plan, [debug_msg]


def _route_next_step(plan: List[Dict[str, Any]], step_index: int) -> tuple:
    """Advance step pointer and return (next_action, next_index)."""
    next_index = step_index + 1
    if next_index >= len(plan):
        return "FINISH", next_index

    step = plan[next_index]
    action = str(step.get("action", "")).lower().strip()
    if action not in ("researcher", "answerer", "mailer"):
        action = "answerer" if next_index == len(plan) - 1 else "researcher"
    return action, next_index


def create_supervisor(llm):
    """Planner + supervisor: classifies intent, generates plans, routes steps."""

    identity_text = load_identity_text()

    prompt = ChatPromptTemplate.from_template(
        """/no_think
Create a minimal execution plan as JSON.

Assistant identity:
{identity}

Available actions:
- researcher (tool use only)
- answerer (final response)
- mailer (email drafting/sending)

Rules:
- Output ONLY JSON: {{"steps":[...]}}
- 1-6 total steps.
- Non-email QA: first step researcher, last step answerer, and 2-6 steps.
- Email requests: last step must be mailer.
- If email needs current info (today/latest/news/web), add researcher+tavily_search before mailer.
- Greetings/chit-chat: exactly one answerer step.
- researcher step requires a non-empty tools list.
- Allowed tools: retrieve_context, tavily_search.
- answerer/mailer steps must omit tools or use [].
- Use long-term memory only as continuity hints; never treat it as fresh web evidence.

User question:
{query}

Controller hint:
{control_hint}

Long-term memory hints:
{memory_context}

Recent conversation:
{messages}
"""
    )

    def supervisor(state: AgentState) -> AgentState:
        """Create or advance a validated JSON plan and route the next step."""
        messages = state.get("messages", [])
        latest_user_text = latest_human_text(messages)
        user_key = state.get("user_key")
        chat_id = state.get("chat_id")
        memory_context = str(state.get("memory_context", "") or "").strip() or "none"

        flags = state.get("flags", {})
        followup_reset = bool(flags.get("followup_reset")) if isinstance(flags, dict) else False
        if followup_reset and user_key and chat_id:
            set_flags(user_key, chat_id, {"followup_reset": None})

        task_state = normalize_task_state(state.get("task_state", {}))
        draft = state.get("draft", {})
        if not isinstance(draft, dict):
            draft = {}

        # 1. Classify intent and update task state accordingly.
        intent_result = _resolve_intent_and_state(
            llm, latest_user_text, messages, draft, task_state,
            user_key, chat_id, followup_reset,
        )
        turn_intent = intent_result["turn_intent"]
        reset_scope = intent_result["reset_scope"]
        use_last_answer = intent_result["use_last_answer"]
        skip_followup = intent_result["skip_followup"]
        has_active_draft = intent_result["has_active_draft"]
        task_state = intent_result["task_state"]
        draft = intent_result["draft"]

        plan: List[Dict[str, Any]] = state.get("plan", [])
        step_index: int = state.get("step_index", -1)

        # 2. Handle active draft fast-paths (confirm, edit, research+send).
        early_return, defer_draft, email_hint = _handle_active_draft(
            state, latest_user_text, messages, turn_intent, task_state,
            has_active_draft, step_index,
        )
        if early_return is not None:
            return early_return

        if defer_draft:
            skip_followup = True
        user_query = effective_query(messages, followup_reset=skip_followup, email_hint=email_hint)
        if defer_draft:
            plan = []
            step_index = -1

        # 3. Generate plan if none exists.
        if not plan:
            plan, planner_reasoning_raw, state_messages = _generate_plan(
                llm, prompt, identity_text, user_query, turn_intent,
                use_last_answer, email_hint, memory_context, messages,
            )
            step_index = -1
        else:
            planner_reasoning_raw = ""
            state_messages: List[BaseMessage] = []

        # 4. Route to the next step.
        nxt, next_index = _route_next_step(plan, step_index)

        # Debug trace message.
        debug_text_lines = [
            f"[Planner] Turn intent: {turn_intent} (reset={reset_scope}, use_last={use_last_answer})",
            f"[Planner] Plan steps ({len(plan)}):",
            *[
                f"  - #{s.get('id', i)} {s.get('action')} :: {s.get('description', '')}"
                for i, s in enumerate(plan)
            ],
            f"[Planner] Next step index: {next_index}, action: {nxt}",
        ]
        debug_msg = AIMessage(content="\n".join(debug_text_lines))
        state_messages.append(debug_msg)

        return {
            "next": nxt,
            "plan": plan,
            "step_index": next_index,
            "messages": state_messages,
            "task_state": task_state,
            "turn_intent": turn_intent,
            "planner_reasoning": planner_reasoning_raw,
        }

    return supervisor
