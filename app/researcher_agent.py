"""Researcher agent that selects and calls tools for evidence gathering."""

from __future__ import annotations

from uuid import uuid4
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage

from config import ASYNC_TOOLS, ASYNC_TIMEOUT_PDF_SEC, ASYNC_TIMEOUT_WEB_SEC
from tools_pdf import retrieve_context, DEFAULT_K
from tools_web import tavily_search, DEFAULT_MAX_RESULTS
from state import AgentState
from intent_utils import effective_query, compact_conversation


def create_researcher(llm):
    """Create the Researcher agent with PDF + web tools."""
    pdf_tool = retrieve_context
    web_tool = tavily_search
    tool_map = {
        "retrieve_context": pdf_tool,
        "tavily_search": web_tool,
    }
    llm_with_tools = llm.bind_tools([pdf_tool, web_tool])

    prompt = ChatPromptTemplate.from_template(
        """/no_think
You are the Researcher node.

Tools:
- retrieve_context: Deep Work PDF retrieval
- tavily_search: web search

Planner-selected tools for this step: {planned_tools}
Rules:
- If planned_tools is not "none", call only from that list.
- Call at least one tool before any text response.
- Use tavily_search for current/recent/live info.
- Use retrieve_context for book-specific questions.

After tool calls, provide short bullet notes and end with: research complete

Question:
{query}

Recent conversation:
{messages}
"""
    )

    def researcher(state: AgentState):
        """Invoke the researcher prompt and emit tool calls."""
        messages = state.get("messages", [])
        flags = state.get("flags", {})
        followup_reset = bool(flags.get("followup_reset")) if isinstance(flags, dict) else False
        plan = state.get("plan", [])
        email_hint = False
        if isinstance(plan, list):
            email_hint = any(
                str(step.get("action", "")).lower().strip() == "mailer"
                for step in plan
                if isinstance(step, dict)
            )
        query_text = effective_query(
            messages, followup_reset=followup_reset, email_hint=email_hint
        )

        step_index = state.get("step_index", -1)
        planned_tools: List[str] = []
        if isinstance(plan, list) and 0 <= step_index < len(plan):
            raw_tools = plan[step_index].get("tools", []) or []
            if isinstance(raw_tools, list):
                planned_tools = [str(t).strip() for t in raw_tools if str(t).strip()]

        # When planner already selected tools, bypass model tool-selection and call directly.
        if planned_tools:
            tool_calls = []
            for tool_name in planned_tools:
                if tool_name == "retrieve_context":
                    args = {"query": query_text, "k": DEFAULT_K}
                elif tool_name == "tavily_search":
                    args = {"query": query_text, "max_results": DEFAULT_MAX_RESULTS}
                else:
                    continue
                tool_calls.append(
                    {
                        "name": tool_name,
                        "args": args,
                        "id": str(uuid4()),
                        "type": "tool_call",
                    }
                )
            if tool_calls:
                return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}

        chain = prompt | llm_with_tools
        response = chain.invoke(
            {
                "query": query_text,
                "messages": compact_conversation(messages),
                "planned_tools": ", ".join(planned_tools) if planned_tools else "none",
            }
        )
        return {"messages": [response]}

    def _latest_tool_calls(messages: List[Any]) -> List[Dict[str, Any]]:
        # Read tool calls from the latest researcher AI message only.
        for msg in reversed(messages):
            calls = getattr(msg, "tool_calls", None)
            if calls:
                return [c for c in calls if isinstance(c, dict)]
            if getattr(msg, "type", "") == "human":
                break
        return []

    def _tool_timeout(name: str) -> int:
        if name == "retrieve_context":
            return max(1, int(ASYNC_TIMEOUT_PDF_SEC))
        if name == "tavily_search":
            return max(1, int(ASYNC_TIMEOUT_WEB_SEC))
        return max(1, int(max(ASYNC_TIMEOUT_PDF_SEC, ASYNC_TIMEOUT_WEB_SEC)))

    def _execute_call(call: Dict[str, Any]) -> ToolMessage:
        name = str(call.get("name", "")).strip()
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        call_id = str(call.get("id") or uuid4())
        tool = tool_map.get(name)
        if tool is None:
            return ToolMessage(
                name=name or "unknown_tool",
                tool_call_id=call_id,
                content=f"TOOL_ERROR: unknown tool '{name}'",
            )
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool.invoke, args)
                content = str(future.result(timeout=_tool_timeout(name)))
        except Exception as exc:
            content = f"TOOL_ERROR: {exc}"
        return ToolMessage(name=name, tool_call_id=call_id, content=content)

    def research_tools(state: AgentState):
        """
        Execute tool calls from the latest researcher message.
        Parallel tool execution is enabled for multi-tool calls (e.g. book+web).
        """
        messages = state.get("messages", [])
        calls = _latest_tool_calls(messages)
        if not calls:
            return {"messages": []}
        if ASYNC_TOOLS and len(calls) > 1:
            tool_messages: List[ToolMessage] = []
            with ThreadPoolExecutor(max_workers=min(len(calls), 4)) as executor:
                future_map: Dict[Any, Tuple[str, str, int]] = {}
                for call in calls:
                    name = str(call.get("name", "")).strip()
                    args = call.get("args", {})
                    if not isinstance(args, dict):
                        args = {}
                    call_id = str(call.get("id") or uuid4())
                    tool = tool_map.get(name)
                    if tool is None:
                        tool_messages.append(
                            ToolMessage(
                                name=name or "unknown_tool",
                                tool_call_id=call_id,
                                content=f"TOOL_ERROR: unknown tool '{name}'",
                            )
                        )
                        continue
                    future = executor.submit(tool.invoke, args)
                    future_map[future] = (name, call_id, _tool_timeout(name))
                for future, (name, call_id, timeout_sec) in future_map.items():
                    try:
                        result = future.result(timeout=timeout_sec)
                        content = str(result)
                    except FuturesTimeoutError:
                        content = f"TOOL_ERROR: timed out after {timeout_sec}s"
                    except Exception as exc:
                        content = f"TOOL_ERROR: {exc}"
                    tool_messages.append(
                        ToolMessage(name=name, tool_call_id=call_id, content=content)
                    )
            return {"messages": tool_messages}

        tool_messages: List[ToolMessage] = []
        for call in calls:
            tool_messages.append(_execute_call(call))
        return {"messages": tool_messages}

    return researcher, research_tools
