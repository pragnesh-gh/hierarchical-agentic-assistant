"""CLI entrypoint that runs the graph and writes JSONL logs."""

import json
import os
import shlex
from datetime import datetime, timezone
from typing import Any, List, Tuple, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from chat_sessions import (
    clear_draft,
    get_active_chat,
    get_draft,
    get_flags,
    get_task_state,
    list_chats,
    load_messages,
    new_chat,
    rename_chat,
    save_messages,
    search_chats,
    set_flags,
    set_task_state,
    resolve_chat_selector,
    switch_chat,
)
from chat_intel import new_chat_tip, should_suggest_new_chat
from config import ANSWERER_MODEL, BASE_DIR, MAILER_MODEL, PLANNER_MODEL, RESEARCHER_MODEL
from contacts import list_contacts
from graph_memory import (
    graph_memory_backend_name,
    ingest_turn_memory,
    retrieve_memory_context,
)
from graph import build_app
from guardrails import check_groundedness, classify_failure, classify_query_source, guardrail_checks
from metrics import duration_ms, now_ms
from redaction import redact_payload
from state import AgentState


def setup_langsmith() -> None:
    """Enable LangSmith tracing when TRACE=1."""
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "hierarchical-agentic-qa")
    allow_raw = os.getenv("TRACE_ALLOW_RAW", "0") == "1"
    if not allow_raw:
        os.environ["LANGCHAIN_HIDE_INPUTS"] = "true"
        os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true"


def disable_langsmith() -> None:
    """Keep tracing disabled unless explicitly requested."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_HIDE_INPUTS"] = "true"
    os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true"


def _summarize_messages(messages: List[Any]) -> List[dict]:
    summarized = []
    for msg in messages:
        msg_type = getattr(msg, "type", "unknown")
        entry = {"type": msg_type}

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = redact_payload(tool_calls)

        if msg_type == "tool":
            entry["name"] = getattr(msg, "name", None)
        content = str(getattr(msg, "content", ""))[:500]
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
    for msg in messages:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            tool_calls.extend(calls)
    return redact_payload(tool_calls) if tool_calls else []


def _extract_tool_results(messages: List[Any]) -> List[dict]:
    results = []
    for msg in messages:
        if getattr(msg, "type", "") == "tool":
            name = getattr(msg, "name", None)
            content = str(getattr(msg, "content", ""))[:500]
            results.append({"name": name, "content": redact_payload(content)})
    return results


def _infer_role(messages: List[Any], state: AgentState) -> str:
    if any(getattr(msg, "type", "") == "tool" for msg in messages):
        return "tool"
    if any(str(getattr(msg, "content", "")).startswith("[Planner]") for msg in messages):
        return "planner"
    if any(getattr(msg, "tool_calls", None) for msg in messages):
        return "researcher"

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
    if role == "mailer":
        return MAILER_MODEL
    if role == "answerer":
        return ANSWERER_MODEL
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


def _parse_kv_args(text: str) -> dict:
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


def _print_contacts() -> None:
    contacts = list_contacts()
    if not contacts:
        print("No allowlisted contacts found.")
        return

    print("Allowlisted contacts:")
    for contact in contacts:
        name = str(contact.get("name", ""))
        email = str(contact.get("email", ""))
        aliases = contact.get("aliases", [])
        aliases_list = aliases if isinstance(aliases, list) else []
        aliases_text = ", ".join(str(alias) for alias in aliases_list) if aliases_list else ""
        if aliases_text:
            print(f"- {name} <{email}> (aliases: {aliases_text})")
        else:
            print(f"- {name} <{email}>")


def _handle_email_command(text: str) -> str:
    args = _parse_kv_args(text)
    to_name = args.get("to") or ""
    body = args.get("body") or ""
    tone = args.get("tone") or ""

    parts = ["email to", to_name]
    if body:
        parts.append(f"about {body}")
    if tone:
        parts.append(f"tone {tone}")
    return " ".join(part for part in parts if part).strip()


def _format_draft_for_display(draft: dict) -> str:
    if not isinstance(draft, dict) or not draft:
        return "No active draft."
    if draft.get("pending"):
        to_name = str(draft.get("canonical_name") or draft.get("to_name") or "")
        subject = str(draft.get("subject") or "")
        tone = str(draft.get("tone") or "")
        lines = [
            "Active draft:",
            f"- status: pending confirmation",
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
            f"- status: waiting for body",
            f"- to: {to_name}",
        ]
        if tone:
            lines.append(f"- tone: {tone}")
        lines.append("Send the email body text to continue.")
        return "\n".join(lines)
    return "Draft exists but is incomplete. Continue with an email instruction."


def _print_chat_rows(rows: List[dict], header: str = "Chats:") -> None:
    if not rows:
        print("No chats found.")
        return
    print(header)
    for idx, chat in enumerate(rows, start=1):
        title = str(chat.get("title", "")).strip() or "New Chat"
        created = str(chat.get("created_at", ""))
        last_active = str(chat.get("last_active", ""))
        preview = str(chat.get("preview", ""))
        print(f"{idx}. {title} | id={chat['id']} | created={created} | last={last_active}")
        if preview:
            print(f"   preview: {preview}")


def _print_step_summary(role: str, new_messages: List[Any]) -> None:
    if role == "planner":
        print("[planner] plan updated")
        return
    if role == "researcher":
        tools = []
        for msg in new_messages:
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                continue
            for call in calls:
                if isinstance(call, dict) and call.get("name"):
                    tools.append(str(call["name"]))
        if tools:
            print(f"[researcher] calling tools: {', '.join(tools)}")
        else:
            print("[researcher] planning tool usage")
        return
    if role == "tool":
        names = [
            str(getattr(msg, "name", "tool"))
            for msg in new_messages
            if getattr(msg, "type", "") == "tool"
        ]
        if names:
            print(f"[tool] completed: {', '.join(names)}")
        else:
            print("[tool] completed")


def _stream_run(
    app: Any,
    initial_state: AgentState,
    run_config: dict,
    question: str,
    run_id: str,
    log_path: Any,
) -> Tuple[AgentState | None, bool]:
    # Stream both graph state updates and answer tokens; only answerer tokens are printed.
    prev_len: int | None = None
    final_state: AgentState | None = None
    streamed_tokens = False
    printed_stream_header = False
    token_buffer = ""
    token_passthrough = False
    thinking_detected = False

    step_counter = 0
    last_values_time = now_ms()

    for event in app.stream(
        cast(AgentState, initial_state),
        config=cast(Any, run_config),
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
                if not printed_stream_header:
                    print("\nStreaming response:\n")
                    printed_stream_header = True
                print(token, end="", flush=True)
                streamed_tokens = True
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
                    if not printed_stream_header:
                        print("\nStreaming response:\n")
                        printed_stream_header = True
                    print(tail, end="", flush=True)
                    streamed_tokens = True
                token_buffer = ""
                token_passthrough = True
                continue

            if not printed_stream_header:
                print("\nStreaming response:\n")
                printed_stream_header = True
            print(token_buffer, end="", flush=True)
            streamed_tokens = True
            token_buffer = ""
            token_passthrough = True
            continue

        if mode != "values" or not isinstance(payload, dict):
            continue

        state = cast(AgentState, payload)
        messages = state.get("messages", [])
        new_messages, prev_len = _extract_step_delta(prev_len, messages)

        # First values event is the initial input state.
        if step_counter == 0 and not state.get("plan"):
            final_state = state
            last_values_time = now_ms()
            step_counter += 1
            continue

        now = now_ms()
        latency_ms = duration_ms(last_values_time, now)
        last_values_time = now

        role = _infer_role(new_messages, state)
        _print_step_summary(role, new_messages)

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
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        final_state = state
        step_counter += 1

    if token_buffer and not token_passthrough:
        if not printed_stream_header:
            print("\nStreaming response:\n")
            printed_stream_header = True
        print(token_buffer, end="", flush=True)
        streamed_tokens = True

    if streamed_tokens:
        print()
    return final_state, streamed_tokens


def main() -> None:
    """Run the CLI loop and write structured JSONL logs."""
    trace_on = os.getenv("TRACE", "0") == "1"
    if trace_on:
        setup_langsmith()
    else:
        disable_langsmith()

    question = input(
        "Enter your question (or /email, /draft, /contacts, /new_chat, /history, /open, /rename, /search): "
    )

    lowered = question.strip().lower()
    if lowered.startswith("/contacts"):
        _print_contacts()
        return
    if lowered.startswith("/draft"):
        chat_id = get_active_chat("cli")
        draft = get_draft("cli", chat_id)
        print(_format_draft_for_display(draft))
        return
    if lowered.startswith("/new_chat"):
        chat_id = new_chat("cli")
        print(f"Started new chat: {chat_id}")
        return
    if lowered.startswith("/chats") or lowered.startswith("/history"):
        chats = list_chats("cli")
        _print_chat_rows(chats, header="Chat History:")
        return
    if lowered.startswith("/open"):
        parts = question.strip().split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /open <index_or_chat_id>")
            return
        selector = parts[1].strip()
        chat_id = resolve_chat_selector("cli", selector)
        if not chat_id:
            print(f"Chat not found: {selector}")
            return
        if switch_chat("cli", chat_id):
            print(f"Opened chat: {chat_id}")
        else:
            print(f"Chat not found: {chat_id}")
        return
    if lowered.startswith("/rename"):
        parts = question.strip().split(maxsplit=2)
        if len(parts) < 3:
            print("Usage: /rename <index_or_chat_id> <new_title>")
            return
        selector = parts[1].strip()
        new_title = parts[2].strip()
        chat_id = resolve_chat_selector("cli", selector)
        if not chat_id:
            print(f"Chat not found: {selector}")
            return
        if rename_chat("cli", chat_id, new_title):
            print(f"Renamed chat {chat_id}.")
        else:
            print("Rename failed.")
        return
    if lowered.startswith("/search"):
        parts = question.strip().split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /search <keyword>")
            return
        rows = search_chats("cli", parts[1].strip(), limit=20)
        _print_chat_rows(rows, header="Search Results:")
        return
    if lowered.startswith("/switch"):
        parts = question.strip().split()
        if len(parts) < 2:
            print("Usage: /switch <chat_id>")
            return
        chat_id = parts[1].strip()
        if switch_chat("cli", chat_id):
            print(f"Switched to chat: {chat_id}")
        else:
            print(f"Chat not found: {chat_id}")
        return
    if lowered.startswith("/stop") or lowered.startswith("/exit"):
        chat_id = get_active_chat("cli")
        clear_draft("cli", chat_id)
        set_flags("cli", chat_id, {"followup_reset": True})
        print("Cancelled. Ready for a new request.")
        return
    if lowered.startswith("/email"):
        question = _handle_email_command(question)

    app, groundedness_llm = build_app()
    runs_dir = BASE_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = runs_dir / f"run_{run_id}.jsonl"

    chat_id = get_active_chat("cli")
    history = load_messages("cli", chat_id)
    suggest_new_chat = should_suggest_new_chat(question, history if isinstance(history, list) else [])
    message_list: List[Any] = []
    if isinstance(history, list):
        message_list.extend(history)
    message_list.append(HumanMessage(content=question))

    draft = get_draft("cli", chat_id)
    flags = get_flags("cli", chat_id)
    task_state = get_task_state("cli", chat_id)
    memory_context = retrieve_memory_context("cli", chat_id, question)
    initial_state: AgentState = {
        "messages": message_list,
        "user_key": "cli",
        "chat_id": chat_id,
        "draft": draft,
        "flags": flags,
        "task_state": task_state,
        "memory_context": memory_context,
        "memory_backend": graph_memory_backend_name(),
    }

    run_config = {
        "metadata": {
            "lab": "hierarchical-agentic-qa",
            "entrypoint": "cli",
            "description": "Planner + Researcher + Answerer + Mailer graph",
        },
        "tags": ["hierarchical-agentic-qa", "multi-agent", "ollama"],
    }

    final_state, streamed_tokens = _stream_run(
        app=app,
        initial_state=initial_state,
        run_config=run_config,
        question=question,
        run_id=run_id,
        log_path=log_path,
    )

    if final_state is None:
        print("No final state produced by the graph.")
        return

    final_msg = final_state["messages"][-1]
    final_text = str(getattr(final_msg, "content", ""))
    if not streamed_tokens:
        print("\nFinal answer:\n")
        print(final_text)
    else:
        if "\nSources:\n" in final_text:
            sources_text = final_text.split("\nSources:\n", 1)[1].strip()
            print("\nSources:")
            print(sources_text)
        elif "Sources: None" in final_text:
            print("\nSources: None")

    # Inline groundedness check: runs after streaming is complete, no token leakage.
    groundedness_result = None
    all_messages = final_state.get("messages", [])
    source_type = classify_query_source(question)
    if groundedness_llm is not None and source_type in {"pdf", "web", "both", "unknown"}:
        groundedness_result = check_groundedness(final_text, all_messages, groundedness_llm)
        verdict = groundedness_result.get("verdict", "")
        if verdict and verdict not in {"grounded", "skipped", "no_tool_output", "no_answer"}:
            reason = groundedness_result.get("reason", "")
            print(f"\n[groundedness: {verdict}] {reason}")

    # Append groundedness result to the last JSONL entry.
    if groundedness_result is not None:
        groundedness_log = {
            "run_id": run_id,
            "step_index": -1,
            "role": "groundedness_judge",
            "model": str(getattr(groundedness_llm, "model", "unknown")),
            "action": "groundedness_check",
            "groundedness_check": groundedness_result,
            "latency_ms": None,
            "errors": None,
        }
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(groundedness_log, ensure_ascii=False) + "\n")

    if suggest_new_chat:
        print(f"\n{new_chat_tip()}")

    save_messages("cli", chat_id, final_state["messages"], question[:80])
    if isinstance(final_state.get("task_state"), dict):
        set_task_state("cli", chat_id, cast(dict, final_state["task_state"]))
    ingest_turn_memory("cli", chat_id, question, final_text)


if __name__ == "__main__":
    main()
