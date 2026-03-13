"""Telegram bot entrypoint for hierarchical-agentic-qa."""

import asyncio
import json
import os
import sys
import importlib
import tempfile
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple, Dict

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from langchain_core.messages import HumanMessage

from app.config import (
    BASE_DIR,
    PLANNER_MODEL,
    RESEARCHER_MODEL,
    ANSWERER_MODEL,
    MAILER_MODEL,
    ASYNC_PERSIST,
    TELEGRAM_PROGRESS,
)
from app.state import AgentState
from app.graph import build_app
from app.guardrails import check_groundedness, classify_failure, classify_query_source, guardrail_checks
from app.metrics import now_ms, duration_ms
from app.redaction import redact_payload
from app.chat_sessions import (
    get_active_chat,
    list_chats,
    new_chat,
    rename_chat,
    resolve_chat_selector,
    search_chats,
    switch_chat,
    load_messages,
    save_messages,
    get_draft,
    get_task_state,
    set_task_state,
    clear_draft,
    get_flags,
    set_flags,
)
from app.chat_intel import new_chat_tip, should_suggest_new_chat
from app.contacts import list_contacts
from app.graph_memory import (
    graph_memory_backend_name,
    ingest_turn_memory,
    retrieve_memory_context,
)

_WHISPER_MODEL = None
_CHAT_LOCKS: Dict[str, asyncio.Lock] = {}


def _configure_tracing() -> None:
    trace_on = os.getenv("TRACE", "0") == "1"
    if trace_on:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_HIDE_INPUTS"] = "true"
        os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true"


def _chat_lock(user_key: str, chat_id: str) -> asyncio.Lock:
    # Prevent concurrent runs for the same user+chat, which can corrupt state order.
    key = f"{user_key}:{chat_id}"
    lock = _CHAT_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _CHAT_LOCKS[key] = lock
    return lock


def _summarize_messages(msgs: List[Any]) -> List[dict]:
    summarized = []
    for m in msgs:
        m_type = getattr(m, "type", "unknown")
        entry = {"type": m_type}

        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = redact_payload(tool_calls)

        if m_type == "tool":
            entry["name"] = getattr(m, "name", None)
            content = str(getattr(m, "content", ""))[:500]
            entry["content"] = redact_payload(content)
        else:
            content = str(getattr(m, "content", ""))[:500]
            entry["content"] = redact_payload(content)

        summarized.append(entry)
    return summarized


def _extract_step_delta(prev_len: int | None, messages: List[Any]) -> Tuple[List[Any], int]:
    if prev_len is None:
        return messages, len(messages)
    if prev_len <= len(messages):
        return messages[prev_len:], len(messages)
    return messages, len(messages)


def _extract_tool_calls(messages: List[Any]) -> List[Any]:
    tool_calls = []
    for m in messages:
        calls = getattr(m, "tool_calls", None)
        if calls:
            tool_calls.extend(calls)
    return redact_payload(tool_calls) if tool_calls else []


def _extract_tool_results(messages: List[Any]) -> List[dict]:
    results = []
    for m in messages:
        if getattr(m, "type", "") == "tool":
            name = getattr(m, "name", None)
            content = str(getattr(m, "content", ""))[:500]
            results.append({"name": name, "content": redact_payload(content)})
    return results


def _infer_role(messages: List[Any], state: Dict[str, Any] | None = None) -> str:
    if any(getattr(m, "type", "") == "tool" for m in messages):
        return "tool"
    if any(str(getattr(m, "content", "")).startswith("[Planner]") for m in messages):
        return "planner"
    if any(getattr(m, "tool_calls", None) for m in messages):
        return "researcher"
    if isinstance(state, dict):
        plan = state.get("plan", [])
        step_index = state.get("step_index", -1)
        if isinstance(plan, list) and isinstance(step_index, int) and 0 <= step_index < len(plan):
            action = str(plan[step_index].get("action", "")).lower().strip()
            if action in {"answerer", "mailer"}:
                return action
    return "answerer"


def _role_to_model(role: str) -> str:
    if role == "planner":
        return PLANNER_MODEL
    if role == "researcher":
        return RESEARCHER_MODEL
    if role == "answerer":
        return ANSWERER_MODEL
    if role == "mailer":
        return MAILER_MODEL
    return "tool"


def _role_to_action(role: str) -> str:
    if role == "planner":
        return "plan"
    if role == "researcher":
        return "research"
    if role == "answerer":
        return "answer"
    if role == "mailer":
        return "mail"
    return "tool_result"


def _infer_role_from_state(messages: List[Any], state: Dict[str, Any]) -> str:
    if any(getattr(m, "type", "") == "tool" for m in messages):
        return "tool"
    if any(str(getattr(m, "content", "")).startswith("[Planner]") for m in messages):
        return "planner"
    if any(getattr(m, "tool_calls", None) for m in messages):
        return "researcher"
    plan = state.get("plan", [])
    step_index = state.get("step_index", -1)
    if isinstance(plan, list) and isinstance(step_index, int) and 0 <= step_index < len(plan):
        action = str(plan[step_index].get("action", "")).lower().strip()
        if action in {"answerer", "mailer"}:
            return action
    return "answerer"


def _append_jsonl(path: Any, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_groundedness_check(
    question: str,
    answer_text: str,
    messages: List[Any],
    groundedness_llm: Any,
) -> Dict[str, Any] | None:
    source_type = classify_query_source(question)
    if groundedness_llm is None:
        return None
    if source_type not in {"pdf", "web", "both", "unknown"}:
        return None
    return check_groundedness(answer_text, messages, groundedness_llm)


def _groundedness_log_record(run_id: str, groundedness_llm: Any, groundedness_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "step_index": -1,
        "role": "groundedness_judge",
        "model": str(getattr(groundedness_llm, "model", "unknown")),
        "action": "groundedness_check",
        "groundedness_check": groundedness_result,
        "latency_ms": None,
        "errors": None,
    }


async def _save_state_async(
    user_key: str,
    chat_id: str,
    messages: List[Any],
    preview: str,
    task_state: Dict[str, Any] | None,
) -> None:
    # Optional async persistence keeps Telegram response path more responsive.
    if ASYNC_PERSIST:
        await asyncio.to_thread(save_messages, user_key, chat_id, messages, preview)
        if isinstance(task_state, dict):
            await asyncio.to_thread(set_task_state, user_key, chat_id, task_state)
        return
    save_messages(user_key, chat_id, messages, preview)
    if isinstance(task_state, dict):
        set_task_state(user_key, chat_id, task_state)


def _strip_think_trace(text: str) -> str:
    cleaned = text or ""
    lower = cleaned.lower()
    if "</think>" in lower:
        idx = lower.rfind("</think>")
        return cleaned[idx + len("</think>") :].strip()
    return cleaned


def _run_question(question: str, user_key: str, chat_id: str) -> str:
    # Synchronous execution path (used when progress streaming is disabled).
    app, groundedness_llm = build_app()
    runs_dir = BASE_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = runs_dir / f"telegram_run_{run_id}.jsonl"

    history = load_messages(user_key, chat_id)
    suggest_new_chat = should_suggest_new_chat(question, history if isinstance(history, list) else [])
    messages = history + [HumanMessage(content=question)]
    draft = get_draft(user_key, chat_id)
    flags = get_flags(user_key, chat_id)
    task_state = get_task_state(user_key, chat_id)
    memory_context = retrieve_memory_context(user_key, chat_id, question)
    initial_state: AgentState = {
        "messages": messages,
        "user_key": user_key,
        "chat_id": chat_id,
        "draft": draft,
        "flags": flags,
        "task_state": task_state,
        "memory_context": memory_context,
        "memory_backend": graph_memory_backend_name(),
    }
    prev_len: int | None = None
    final_state = None

    for step_idx, state in enumerate(app.stream(initial_state, stream_mode="values")):
        step_start = now_ms()
        messages = state.get("messages", [])
        new_messages, prev_len = _extract_step_delta(prev_len, messages)
        step_end = now_ms()

        guardrail_checks_result = guardrail_checks(question, state.get("plan"), messages)
        role = _infer_role(new_messages, state)
        plan_step_index = state.get("step_index", step_idx)
        log_record = {
            "run_id": run_id,
            "step_index": plan_step_index,
            "role": role,
            "model": _role_to_model(role),
            "action": _role_to_action(role),
            "tool_calls": _extract_tool_calls(new_messages),
            "tool_results": _extract_tool_results(new_messages),
            "latency_ms": duration_ms(step_start, step_end),
            "guardrail_checks": guardrail_checks_result,
            "failure_mode": classify_failure(guardrail_checks_result),
            "planner_reasoning": state.get("planner_reasoning", ""),
            "groundedness_check": state.get("groundedness_check"),
            "errors": None,
            "plan": state.get("plan"),
            "messages": _summarize_messages(new_messages),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        final_state = state

    if final_state is None:
        return "No final answer produced."

    final_msg = final_state["messages"][-1]
    final_text = _strip_think_trace(str(getattr(final_msg, "content", "")))

    groundedness_result = _run_groundedness_check(
        question=question,
        answer_text=final_text,
        messages=final_state.get("messages", []),
        groundedness_llm=groundedness_llm,
    )
    if groundedness_result is not None:
        _append_jsonl(log_path, _groundedness_log_record(run_id, groundedness_llm, groundedness_result))

    if suggest_new_chat and new_chat_tip() not in final_text:
        final_text = f"{final_text}\n\n{new_chat_tip()}"
    preview = question[:80]
    save_messages(user_key, chat_id, final_state["messages"], preview)
    if isinstance(final_state.get("task_state"), dict):
        set_task_state(user_key, chat_id, final_state["task_state"])
    ingest_turn_memory(user_key, chat_id, question, final_text)
    return final_text


async def _run_question_with_progress(
    question: str,
    user_key: str,
    chat_id: str,
    context: Any,
    progress_msg: Any,
) -> str:
    """Run graph with throttled progress updates and one final edited message.

    Strategy avoids per-token Telegram spam while still showing active progress.
    """
    app, groundedness_llm = build_app()
    runs_dir = BASE_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = runs_dir / f"telegram_run_{run_id}.jsonl"

    if ASYNC_PERSIST:
        history, draft, flags, task_state, memory_context = await asyncio.gather(
            asyncio.to_thread(load_messages, user_key, chat_id),
            asyncio.to_thread(get_draft, user_key, chat_id),
            asyncio.to_thread(get_flags, user_key, chat_id),
            asyncio.to_thread(get_task_state, user_key, chat_id),
            asyncio.to_thread(retrieve_memory_context, user_key, chat_id, question),
        )
    else:
        history = load_messages(user_key, chat_id)
        draft = get_draft(user_key, chat_id)
        flags = get_flags(user_key, chat_id)
        task_state = get_task_state(user_key, chat_id)
        memory_context = retrieve_memory_context(user_key, chat_id, question)
    suggest_new_chat = should_suggest_new_chat(question, history if isinstance(history, list) else [])
    messages = history + [HumanMessage(content=question)]
    initial_state: AgentState = {
        "messages": messages,
        "user_key": user_key,
        "chat_id": chat_id,
        "draft": draft,
        "flags": flags,
        "task_state": task_state,
        "memory_context": memory_context,
        "memory_backend": graph_memory_backend_name(),
    }

    stage_text = "Planning..."
    displayed_text = ""
    token_buffer = ""
    token_passthrough = False
    thinking_detected = False
    last_edit_ts = 0.0
    edit_interval_sec = 1.2
    prev_len: int | None = None
    step_counter = 0
    last_values_time = now_ms()
    final_state = None
    pending_log_tasks: List[asyncio.Task] = []
    stop_heartbeat = asyncio.Event()

    async def _push_progress(force: bool = False) -> None:
        nonlocal last_edit_ts
        now = time.time()
        if not force and (now - last_edit_ts) < edit_interval_sec:
            return

        if displayed_text:
            preview = displayed_text.strip()
            preview = preview[-1200:]
            text = f"{stage_text}\n\n{preview}"
        else:
            text = stage_text
        try:
            await progress_msg.edit_text(text)
            last_edit_ts = now
        except Exception:
            # Ignore edit collisions / no-op edits.
            pass

    async def _progress_heartbeat() -> None:
        if not TELEGRAM_PROGRESS:
            return
        pulse = 0
        while not stop_heartbeat.is_set():
            try:
                await context.bot.send_chat_action(chat_id=progress_msg.chat_id, action="typing")
            except Exception:
                pass
            pulse += 1
            if pulse % 2 == 0:
                await _push_progress(force=True)
            try:
                await asyncio.wait_for(stop_heartbeat.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                continue

    heartbeat_task = asyncio.create_task(_progress_heartbeat())

    try:
        async for event in app.astream(
            initial_state,
            stream_mode=["values", "messages"],
        ):
            if not isinstance(event, tuple) or len(event) != 2:
                continue
            mode, payload = event

            if mode == "messages":
                if not isinstance(payload, tuple) or len(payload) != 2:
                    continue
                msg, meta = payload
                metadata = meta if isinstance(meta, dict) else {}
                node = str(metadata.get("langgraph_node", "")).lower().strip()
                if node != "answerer":
                    continue
                if not msg.__class__.__name__.endswith("Chunk"):
                    continue
                token = str(getattr(msg, "content", ""))
                if not token:
                    continue

                if token_passthrough:
                    displayed_text += token
                    await _push_progress(force=False)
                    continue

                token_buffer += token
                if len(token_buffer) < 120 and "</think>" not in token_buffer:
                    continue

                lower_buffer = token_buffer.lower()
                if not thinking_detected:
                    early_window = lower_buffer[:400]
                    if (
                        "<think>" in early_window
                        or "think" in early_window
                        or "let me" in early_window
                        or "let's" in early_window
                        or "i need to" in early_window
                    ):
                        thinking_detected = True

                if thinking_detected and "</think>" not in token_buffer:
                    continue

                if "</think>" in token_buffer:
                    tail = token_buffer.rsplit("</think>", 1)[-1]
                    if tail:
                        displayed_text += tail
                    token_buffer = ""
                    token_passthrough = True
                    await _push_progress(force=False)
                    continue

                displayed_text += token_buffer
                token_buffer = ""
                token_passthrough = True
                await _push_progress(force=False)
                continue

            if mode != "values" or not isinstance(payload, dict):
                continue

            state = payload
            messages = state.get("messages", [])
            new_messages, prev_len = _extract_step_delta(prev_len, messages)

            if step_counter == 0 and not state.get("plan"):
                final_state = state
                last_values_time = now_ms()
                step_counter += 1
                continue

            role = _infer_role_from_state(new_messages, state)
            if role == "planner":
                stage_text = "Planning..."
            elif role == "researcher":
                stage_text = "Researching..."
            elif role == "tool":
                stage_text = "Using tools..."
            elif role == "mailer":
                stage_text = "Drafting email..."
            else:
                stage_text = "Writing answer..."
            await _push_progress(force=False)

            now_ms_value = now_ms()
            latency_ms = duration_ms(last_values_time, now_ms_value)
            last_values_time = now_ms_value

            checks = guardrail_checks(question, state.get("plan"), messages)
            plan_step_index = state.get("step_index", step_counter - 1)
            log_record = {
                "run_id": run_id,
                "step_index": plan_step_index,
                "role": role,
                "model": _role_to_model(role),
                "action": _role_to_action(role),
                "tool_calls": _extract_tool_calls(new_messages),
                "tool_results": _extract_tool_results(new_messages),
                "latency_ms": latency_ms,
                "guardrail_checks": checks,
                "failure_mode": classify_failure(checks),
                "planner_reasoning": state.get("planner_reasoning", ""),
                "groundedness_check": state.get("groundedness_check"),
                "errors": None,
                "plan": state.get("plan"),
                "messages": _summarize_messages(new_messages),
            }
            if ASYNC_PERSIST:
                pending_log_tasks.append(
                    asyncio.create_task(asyncio.to_thread(_append_jsonl, log_path, log_record))
                )
            else:
                _append_jsonl(log_path, log_record)
            final_state = state
            step_counter += 1
    finally:
        stop_heartbeat.set()
        try:
            await heartbeat_task
        except Exception:
            pass

    if final_state is None:
        await progress_msg.edit_text("No final answer produced.")
        return "No final answer produced."

    final_msg = final_state["messages"][-1]
    final_text = str(getattr(final_msg, "content", ""))
    final_text = _strip_think_trace(final_text)
    judge_text = final_text
    if suggest_new_chat and new_chat_tip() not in final_text:
        final_text = f"{final_text}\n\n{new_chat_tip()}"
    if len(final_text) > 4000:
        preview = final_text[:3900] + "\n\n[truncated]"
        await progress_msg.edit_text(preview)
    else:
        await progress_msg.edit_text(final_text)

    if pending_log_tasks:
        await asyncio.gather(*pending_log_tasks, return_exceptions=True)

    groundedness_result = _run_groundedness_check(
        question=question,
        answer_text=judge_text,
        messages=final_state.get("messages", []),
        groundedness_llm=groundedness_llm,
    )
    if groundedness_result is not None:
        groundedness_log = _groundedness_log_record(run_id, groundedness_llm, groundedness_result)
        if ASYNC_PERSIST:
            await asyncio.to_thread(_append_jsonl, log_path, groundedness_log)
        else:
            _append_jsonl(log_path, groundedness_log)

    preview = question[:80]
    await _save_state_async(
        user_key=user_key,
        chat_id=chat_id,
        messages=final_state["messages"],
        preview=preview,
        task_state=final_state.get("task_state"),
    )
    if ASYNC_PERSIST:
        await asyncio.to_thread(ingest_turn_memory, user_key, chat_id, question, final_text)
    else:
        ingest_turn_memory(user_key, chat_id, question, final_text)
    return final_text


def _latest_telegram_run() -> str:
    runs_dir = BASE_DIR / "runs"
    if not runs_dir.exists():
        return "none"
    candidates = sorted(runs_dir.glob("telegram_run_*.jsonl"), reverse=True)
    if not candidates:
        return "none"
    return candidates[0].name


def _load_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    model_name = os.getenv("WHISPER_MODEL", "small")
    try:
        whisper = importlib.import_module("whisper")
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: openai-whisper. Install with: pip install openai-whisper"
        ) from exc

    _WHISPER_MODEL = whisper.load_model(model_name)
    return _WHISPER_MODEL


def _transcribe_audio(file_path: str) -> str:
    model = _load_whisper_model()
    result = model.transcribe(file_path, fp16=False)
    text = str(result.get("text", "")).strip()
    return text


async def _handle_start(update: Any, context: Any) -> None:
    await update.message.reply_text(
        "Send me a question and I will answer using the hierarchical QA system.\n"
        "Use /history to browse chats, /open to switch, /new_chat to start fresh."
    )


async def _handle_status(update: Any, context: Any) -> None:
    latest_run = _latest_telegram_run()
    text = (
        "Status:\n"
        f"- planner: {PLANNER_MODEL}\n"
        f"- researcher: {RESEARCHER_MODEL}\n"
        f"- answerer: {ANSWERER_MODEL}\n"
        f"- mailer: {MAILER_MODEL}\n"
        f"- latest_log: {latest_run}"
    )
    await update.message.reply_text(text)


async def _handle_stop(update: Any, context: Any) -> None:
    user_key = _get_user_key(update)
    chat_id = get_active_chat(user_key)
    clear_draft(user_key, chat_id)
    set_flags(user_key, chat_id, {"followup_reset": True})
    await update.message.reply_text("Cancelled. Ready for a new request.")


async def _handle_new_chat(update: Any, context: Any) -> None:
    user_key = _get_user_key(update)
    chat_id = new_chat(user_key)
    await update.message.reply_text(f"Started new chat: {chat_id}")


def _format_chats_for_display(chats: List[Dict[str, str]], limit: int = 20) -> str:
    if not chats:
        return "No chats found."
    lines = ["Chat History:"]
    for idx, chat in enumerate(chats[: max(1, int(limit))], start=1):
        title = str(chat.get("title", "")).strip() or "New Chat"
        created = str(chat.get("created_at", ""))
        last_active = str(chat.get("last_active", ""))
        preview = str(chat.get("preview", ""))
        lines.append(f"{idx}. {title}")
        lines.append(f"   id: {chat['id']}")
        lines.append(f"   created: {created}")
        lines.append(f"   last: {last_active}")
        if preview:
            lines.append(f"   preview: {preview}")
    return "\n".join(lines)


def _history_keyboard(chats: List[Dict[str, str]], max_buttons: int = 10) -> Any:
    try:
        telegram = importlib.import_module("telegram")
        InlineKeyboardButton = getattr(telegram, "InlineKeyboardButton")
        InlineKeyboardMarkup = getattr(telegram, "InlineKeyboardMarkup")
    except Exception:
        return None
    rows = []
    for idx, chat in enumerate(chats[: max(1, int(max_buttons))], start=1):
        chat_id = str(chat.get("id", ""))
        title = str(chat.get("title", "")).strip() or "New Chat"
        button_text = f"{idx}. {title[:28]}"
        rows.append([InlineKeyboardButton(button_text, callback_data=f"openchat:{chat_id}")])
    if not rows:
        return None
    return InlineKeyboardMarkup(rows)


async def _handle_chats(update: Any, context: Any) -> None:
    user_key = _get_user_key(update)
    chats = list_chats(user_key)
    if not chats:
        await update.message.reply_text("No chats found.")
        return
    await update.message.reply_text(_format_chats_for_display(chats, limit=20))


async def _handle_history(update: Any, context: Any) -> None:
    user_key = _get_user_key(update)
    chats = list_chats(user_key)
    if not chats:
        await update.message.reply_text("No chats found.")
        return
    text = _format_chats_for_display(chats, limit=15)
    keyboard = _history_keyboard(chats, max_buttons=10)
    if keyboard is None:
        await update.message.reply_text(text)
        return
    await update.message.reply_text(text, reply_markup=keyboard)


async def _handle_open(update: Any, context: Any) -> None:
    if not update.message or not update.message.text:
        return
    parts = update.message.text.strip().split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /open <index_or_chat_id>")
        return
    selector = parts[1].strip()
    user_key = _get_user_key(update)
    chat_id = resolve_chat_selector(user_key, selector)
    if not chat_id:
        await update.message.reply_text(f"Chat not found: {selector}")
        return
    if switch_chat(user_key, chat_id):
        await update.message.reply_text(f"Opened chat: {chat_id}")
    else:
        await update.message.reply_text(f"Chat not found: {chat_id}")


async def _handle_rename(update: Any, context: Any) -> None:
    if not update.message or not update.message.text:
        return
    parts = update.message.text.strip().split(maxsplit=2)
    if len(parts) < 3:
        await update.message.reply_text("Usage: /rename <index_or_chat_id> <new_title>")
        return
    selector = parts[1].strip()
    new_title = parts[2].strip()
    user_key = _get_user_key(update)
    chat_id = resolve_chat_selector(user_key, selector)
    if not chat_id:
        await update.message.reply_text(f"Chat not found: {selector}")
        return
    if rename_chat(user_key, chat_id, new_title):
        await update.message.reply_text(f"Renamed chat {chat_id}.")
    else:
        await update.message.reply_text("Rename failed.")


async def _handle_search(update: Any, context: Any) -> None:
    if not update.message or not update.message.text:
        return
    parts = update.message.text.strip().split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /search <keyword>")
        return
    query = parts[1].strip()
    user_key = _get_user_key(update)
    rows = search_chats(user_key, query, limit=20)
    if not rows:
        await update.message.reply_text("No matching chats.")
        return
    await update.message.reply_text(_format_chats_for_display(rows, limit=20))


async def _handle_switch(update: Any, context: Any) -> None:
    if not update.message or not update.message.text:
        return
    parts = update.message.text.strip().split()
    if len(parts) < 2:
        await update.message.reply_text("Usage: /switch <chat_id>")
        return
    chat_id = parts[1].strip()
    user_key = _get_user_key(update)
    if switch_chat(user_key, chat_id):
        await update.message.reply_text(f"Switched to chat: {chat_id}")
    else:
        await update.message.reply_text(f"Chat not found: {chat_id}")


async def _handle_openchat_callback(update: Any, context: Any) -> None:
    callback = getattr(update, "callback_query", None)
    if callback is None:
        return
    data = str(getattr(callback, "data", "") or "")
    if not data.startswith("openchat:"):
        return
    chat_id = data.split(":", 1)[1].strip()
    user_key = _get_user_key(update)
    switched = switch_chat(user_key, chat_id)
    await callback.answer()
    if switched:
        await callback.edit_message_text(f"Opened chat: {chat_id}")
    else:
        await callback.edit_message_text(f"Chat not found: {chat_id}")


def _format_contacts() -> str:
    contacts = list_contacts()
    if not contacts:
        return "No allowlisted contacts found."
    lines = ["Allowlisted contacts:"]
    for contact in contacts:
        name = str(contact.get("name", ""))
        email = str(contact.get("email", ""))
        aliases = contact.get("aliases", [])
        aliases_list = aliases if isinstance(aliases, list) else []
        aliases_text = ", ".join(str(a) for a in aliases_list) if aliases_list else ""
        if aliases_text:
            lines.append(f"- {name} <{email}> (aliases: {aliases_text})")
        else:
            lines.append(f"- {name} <{email}>")
    return "\n".join(lines)


def _parse_email_args(text: str) -> dict:
    args = {}
    try:
        tokens = shlex.split(text)
    except ValueError:
        return args
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        args[key.strip().lower()] = value.strip()
    return args


def _format_draft_for_display(draft: dict) -> str:
    if not isinstance(draft, dict) or not draft:
        return "No active draft."
    if draft.get("pending"):
        to_name = str(draft.get("canonical_name") or draft.get("to_name") or "")
        subject = str(draft.get("subject") or "")
        tone = str(draft.get("tone") or "")
        lines = [
            "Active draft:",
            "- status: pending confirmation",
            f"- to: {to_name}",
            f"- subject: {subject}",
        ]
        if tone:
            lines.append(f"- tone: {tone}")
        lines.append("Reply `yes` to send, `no` to cancel, or provide edits.")
        return "\n".join(lines)
    if str(draft.get("stage", "")).lower().strip() == "body":
        to_name = str(draft.get("canonical_name") or draft.get("to_name") or "")
        tone = str(draft.get("tone") or "")
        lines = [
            "Active draft:",
            "- status: waiting for body",
            f"- to: {to_name}",
        ]
        if tone:
            lines.append(f"- tone: {tone}")
        lines.append("Send the email body text to continue.")
        return "\n".join(lines)
    return "Draft exists but is incomplete. Continue with an email instruction."


def _get_user_key(update: Any) -> str:
    user = getattr(update, "effective_user", None)
    if user and getattr(user, "id", None) is not None:
        return f"tg_{user.id}"
    return "tg_unknown"


async def _handle_message(update: Any, context: Any) -> None:
    if not update.message or not update.message.text:
        return
    user_key = _get_user_key(update)
    chat_id = get_active_chat(user_key)
    question = update.message.text.strip()
    lock = _chat_lock(user_key, chat_id)
    if lock.locked():
        await update.message.reply_text("Still working on your previous request. Please wait a moment.")
        return
    async with lock:
        if TELEGRAM_PROGRESS:
            progress = await update.message.reply_text("Working on it...")
            try:
                await _run_question_with_progress(question, user_key, chat_id, context, progress)
            except Exception as exc:
                try:
                    await progress.edit_text(f"Error: {exc}")
                except Exception:
                    await update.message.reply_text(f"Error: {exc}")
            return
        try:
            final_text = await asyncio.to_thread(_run_question, question, user_key, chat_id)
            await update.message.reply_text(final_text[:4000] if final_text else "No final answer produced.")
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")


async def _handle_text_input(update: Any, context: Any, text: str) -> bool:
    if not text:
        return False
    user_key = _get_user_key(update)
    chat_id = get_active_chat(user_key)
    lock = _chat_lock(user_key, chat_id)
    if lock.locked():
        await update.message.reply_text("Still working on your previous request. Please wait a moment.")
        return False
    async with lock:
        progress = await update.message.reply_text("Working on it...")
        await _run_question_with_progress(text, user_key, chat_id, context, progress)
    return True


async def _handle_contacts(update: Any, context: Any) -> None:
    await update.message.reply_text(_format_contacts())


async def _handle_draft(update: Any, context: Any) -> None:
    user_key = _get_user_key(update)
    chat_id = get_active_chat(user_key)
    draft = get_draft(user_key, chat_id)
    await update.message.reply_text(_format_draft_for_display(draft))


async def _handle_email(update: Any, context: Any) -> None:
    if not update.message or not update.message.text:
        return
    args = _parse_email_args(update.message.text)
    to_name = args.get("to") or ""
    body = args.get("body") or ""
    tone = args.get("tone") or "formal conversational"

    if not to_name or not body:
        await update.message.reply_text(
            "Usage: /email to=<name> body=\"...\""
        )
        return

    user_key = _get_user_key(update)
    chat_id = get_active_chat(user_key)
    text = f"email to {to_name} about {body}"
    if tone:
        text = f"{text} tone {tone}"
    lock = _chat_lock(user_key, chat_id)
    if lock.locked():
        await update.message.reply_text("Still working on your previous request. Please wait a moment.")
        return
    async with lock:
        progress = await update.message.reply_text("Working on it...")
        await _run_question_with_progress(text, user_key, chat_id, context, progress)


async def _handle_voice(update: Any, context: Any) -> None:
    if not update.message or not update.message.voice:
        return
    await update.message.reply_text("Transcribing voice message...")
    tmp_path = None
    try:
        voice = update.message.voice
        file_obj = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_file:
            tmp_path = tmp_file.name
        await file_obj.download_to_drive(tmp_path)

        transcript = await asyncio.to_thread(_transcribe_audio, tmp_path)
        if not transcript:
            await update.message.reply_text("Could not transcribe audio.")
            return

        await update.message.reply_text(f"Transcription: {transcript}")

        if await _handle_text_input(update, context, transcript):
            return
    except Exception as exc:
        await update.message.reply_text(f"Error: {exc}")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


async def _handle_telegram_error(update: Any, context: Any) -> None:
    err = getattr(context, "error", None)
    text = str(err or "")
    if "terminated by other getUpdates request" in text:
        print(
            "Telegram polling conflict: another bot instance is already running "
            "with this token. Stop the other instance and start this one again."
        )
        return
    print(f"Telegram bot error: {text}")


def main() -> None:
    load_dotenv()
    _configure_tracing()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in the environment.")

    try:
        telegram_ext = importlib.import_module("telegram.ext")
        ApplicationBuilder = getattr(telegram_ext, "ApplicationBuilder")
        CommandHandler = getattr(telegram_ext, "CommandHandler")
        CallbackQueryHandler = getattr(telegram_ext, "CallbackQueryHandler")
        MessageHandler = getattr(telegram_ext, "MessageHandler")
        filters = getattr(telegram_ext, "filters")
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: python-telegram-bot. Install with: pip install python-telegram-bot"
        ) from exc

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", _handle_start))
    app.add_handler(CommandHandler("status", _handle_status))
    app.add_handler(CommandHandler("stop", _handle_stop))
    app.add_handler(CommandHandler("exit", _handle_stop))
    app.add_handler(CommandHandler("new_chat", _handle_new_chat))
    app.add_handler(CommandHandler("history", _handle_history))
    app.add_handler(CommandHandler("chats", _handle_chats))
    app.add_handler(CommandHandler("open", _handle_open))
    app.add_handler(CommandHandler("rename", _handle_rename))
    app.add_handler(CommandHandler("search", _handle_search))
    app.add_handler(CommandHandler("switch", _handle_switch))
    app.add_handler(CommandHandler("contacts", _handle_contacts))
    app.add_handler(CommandHandler("draft", _handle_draft))
    app.add_handler(CommandHandler("email", _handle_email))
    app.add_handler(CallbackQueryHandler(_handle_openchat_callback, pattern=r"^openchat:"))
    app.add_handler(MessageHandler(filters.VOICE, _handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    app.add_error_handler(_handle_telegram_error)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
