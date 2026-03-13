"""Answerer agent that synthesizes the final response."""

import re

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chat_sessions import normalize_task_state, set_flags
from guardrails import classify_query_source
from identity import load_identity_text
from intent_utils import compact_conversation, effective_query, is_no_email_only


def create_answer_chain(llm):
    """Build the answer-generation chain."""

    prompt = ChatPromptTemplate.from_template(
        """/no_think
Write the final user-facing answer.

Assistant identity:
{identity}

Rules:
- Use only information present in the provided context.
- Be concise and factual.
- Use long-term memory only for continuity and pronoun resolution.
- If tool evidence conflicts with long-term memory hints, prefer tool evidence.
- Do not include a Sources section (it is appended separately).

User question:
{query}

Long-term memory hints:
{memory_context}

Recent context:
{messages}

Final answer:"""
    )
    return prompt | llm | StrOutputParser()


def create_answerer(llm):
    """Return the LangGraph answerer node."""

    chain = create_answer_chain(llm)
    identity_text = load_identity_text()

    def _latest_human_index(messages):
        for idx in range(len(messages) - 1, -1, -1):
            if getattr(messages[idx], "type", "") == "human":
                return idx
        return -1

    def _extract_sources(messages):
        # Collect citations from tool outputs so answer prompt can stay concise.
        pdf_citations = set()
        web_urls = set()

        for msg in messages:
            if getattr(msg, "type", "") != "tool":
                continue
            name = getattr(msg, "name", "")
            content = str(getattr(msg, "content", ""))

            if name == "retrieve_context":
                for match in re.findall(r"\[p\.(\d+)\]", content):
                    pdf_citations.add(f"[p.{match}]")
            if name == "tavily_search":
                for match in re.findall(r"https?://\S+", content):
                    web_urls.add(match.rstrip(").,;"))

        return sorted(pdf_citations), sorted(web_urls)

    def _strip_think_trace(text: str) -> str:
        cleaned = text or ""
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if "</think>" in cleaned:
            cleaned = cleaned.rsplit("</think>", 1)[-1]
        return cleaned.strip() or text

    def _strip_sources_block(text: str) -> str:
        content = text or ""
        if "\n\nSources:\n" in content:
            return content.split("\n\nSources:\n", 1)[0].strip()
        if "\n\nSources: None" in content:
            return content.split("\n\nSources: None", 1)[0].strip()
        return content.strip()

    def _filter_rejected(messages, rejected_items):
        # Exclude previously rejected answers from short-term context to reduce repetition.
        if not isinstance(rejected_items, list) or not rejected_items:
            return messages
        rejected_snippets = []
        for item in rejected_items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if text:
                rejected_snippets.append(text[:180])
        if not rejected_snippets:
            return messages

        filtered = []
        for msg in messages:
            if getattr(msg, "type", "") != "ai":
                filtered.append(msg)
                continue
            content = str(getattr(msg, "content", ""))
            if any(snippet and snippet in content for snippet in rejected_snippets):
                continue
            filtered.append(msg)
        return filtered

    def _conversational_reply(query: str) -> str:
        # Fast deterministic response for basic smalltalk.
        normalized = (query or "").lower().strip()
        if normalized in {"yes", "y", "no", "n"}:
            return "Please tell me what you want to do next."
        if "how are you" in normalized:
            return "I am doing well, thank you. How can I help you today?"
        if "who are you" in normalized or "what are you" in normalized:
            return "I am Arjun. I help with Q&A, research, and email drafting with calm, practical guidance."
        if "what can you do" in normalized or "help me" in normalized:
            return "I can answer questions, run web/PDF research, and draft emails for your contacts."
        return "Hello! How can I help you today?"

    def answerer_node(state):
        messages = state.get("messages", [])
        task_state = normalize_task_state(state.get("task_state", {}))
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

        user_query = effective_query(
            messages, followup_reset=followup_reset, email_hint=email_hint
        )
        if classify_query_source(user_query) == "conversational":
            short_reply = _conversational_reply(user_query)
            task_state["active_task"] = "smalltalk"
            return {"messages": [AIMessage(content=f"{short_reply}\n\nSources: None")], "task_state": task_state}
        if is_no_email_only(user_query):
            user_key = state.get("user_key")
            chat_id = state.get("chat_id")
            if user_key and chat_id:
                set_flags(user_key, chat_id, {"followup_reset": True})
            task_state["active_task"] = "qa"
            return {
                "messages": [
                    AIMessage(
                        content="Okay, I won't email anyone. What would you like to do next?"
                    )
                ],
                "task_state": task_state,
            }

        latest_human_idx = _latest_human_index(messages)
        context_messages = messages
        if latest_human_idx >= 0:
            context_messages = messages[:latest_human_idx] + messages[latest_human_idx + 1 :]
        # Build compact context for synthesis while avoiding rejected generations.
        context_messages = _filter_rejected(
            context_messages, task_state.get("rejected_answers", [])
        )

        context_text = compact_conversation(
            context_messages, max_messages=12, max_chars=500
        )

        final_text = chain.invoke(
            {
                "query": user_query,
                "messages": context_text,
                "identity": identity_text,
                "memory_context": str(state.get("memory_context", "") or "").strip() or "none",
            }
        )
        final_text = _strip_think_trace(final_text)

        pdf_citations, web_urls = _extract_sources(messages)
        sources_lines = [f"- {citation}" for citation in pdf_citations]
        sources_lines.extend(f"- {url}" for url in web_urls)
        answer_core = _strip_sources_block(final_text)
        task_state["active_task"] = "qa"
        task_state["last_answer"] = {
            "text": answer_core[:1800],
            "sources": sources_lines,
            "accepted": True,
        }

        if sources_lines:
            final_text = f"{final_text}\n\nSources:\n" + "\n".join(sources_lines)
        else:
            final_text = f"{final_text}\n\nSources: None"

        return {"messages": [AIMessage(content=final_text)], "task_state": task_state}

    return answerer_node
