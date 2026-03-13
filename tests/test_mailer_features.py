import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from mailer_agent import (
    _edit_prompt,
    _handle_send_confirmation,
    _handle_new_email,
    _intent_prompt,
    _email_prompt,
    _match_email_intent,
    _normalize_email_body,
)  # noqa: E402


class TestMailerFeatures(unittest.TestCase):
    def test_identity_line_always_present(self) -> None:
        body = _normalize_email_body(
            raw_body="Tesla Optimus Gen 3 is expected in Q1 2026.",
            recipient="Pragnesh",
            signature_name="Pragnesh Kumar",
            include_sources=False,
            sources_lines=[],
            allowed_urls=set(),
        )
        self.assertIn("I am Arjun, Pragnesh's AI assistant.", body)
        self.assertIn("Hello Pragnesh!", body)

    def test_direct_email_fact_transfer(self) -> None:
        def stub_llm(_):
            return AIMessage(content='{"to_name":"","body":"","tone":""}')

        llm = RunnableLambda(stub_llm)
        messages = [
            AIMessage(
                content=(
                    "OpenAI signed a deal with the Department of Defense for AI systems.\n\n"
                    "Sources:\n- https://example.com/openai-dod"
                )
            ),
            AIMessage(
                content=(
                    "Tesla is preparing Optimus Gen 3 for Q1 2026 factory deployment.\n\n"
                    "Sources:\n- https://example.com/tesla-optimus"
                )
            ),
            HumanMessage(
                content=(
                    "can you send raj@example.com a summary of what we discussed and also ask "
                    "when he is free this week to talk about this"
                )
            ),
        ]

        with patch("mailer_agent.set_draft", lambda *args, **kwargs: None):
            result = _handle_new_email(
                messages=messages,
                user_text=(
                    "can you send raj@example.com a summary of what we discussed and also ask "
                    "when he is free this week to talk about this"
                ),
                user_key="test",
                chat_id="chat",
                llm=llm,
                identity_text="Arjun",
                email_prompt=_email_prompt(),
                intent_prompt=_intent_prompt(),
                default_tone="formal conversational",
                signature_name="Pragnesh Kumar",
                sources_lines=[],
                sources_block_text="",
                task_state={"last_answer": {"text": "Tesla is preparing Optimus Gen 3 for Q1 2026.", "accepted": True}},
                memory_context=(
                    "Long-term memory hints:\n"
                    "- (1) User: openai update Assistant: OpenAI signed a deal with the Department of Defense.\n"
                    "- (2) User: tesla robots Assistant: Tesla is preparing Optimus Gen 3 for Q1 2026."
                ),
            )

        self.assertIn("messages", result)
        self.assertTrue(result["messages"])
        out = str(result["messages"][0].content)
        self.assertIn("Confirm send?", out)
        self.assertIn("<raj@example.com>", out)
        self.assertIn("OpenAI", out)
        self.assertIn("Tesla", out)
        self.assertIn("Could you let me know when you are free this week to discuss this?", out)
        self.assertIn("I am Arjun, Pragnesh's AI assistant.", out)

    def test_match_email_intent_handles_reminding_phrase(self) -> None:
        name, body = _match_email_intent(
            "Hey, can you send a mail to Pragnesh reminding him that he has an interview tomorrow at 10am?"
        )
        self.assertEqual(name.lower(), "pragnesh")
        self.assertIn("interview", body.lower())

    def test_new_email_falls_back_when_llm_body_ignores_request(self) -> None:
        def stub_llm(_):
            return AIMessage(
                content=(
                    '{"subject":"Pragnesh\'s AI Assistant",'
                    '"body":"I am Arjun, Pragnesh\'s AI assistant."}'
                )
            )

        llm = RunnableLambda(stub_llm)
        messages = [
            HumanMessage(
                content="send a mail to pragnesh about the movie next week. remind him to book the tickets"
            )
        ]

        with patch("mailer_agent.set_draft", lambda *args, **kwargs: None):
            result = _handle_new_email(
                messages=messages,
                user_text="send a mail to pragnesh about the movie next week. remind him to book the tickets",
                user_key="test",
                chat_id="chat",
                llm=llm,
                identity_text="Arjun",
                email_prompt=_email_prompt(),
                intent_prompt=_intent_prompt(),
                default_tone="formal conversational",
                signature_name="Pragnesh Kumar",
                sources_lines=[],
                sources_block_text="",
                task_state={"last_answer": {"text": "", "accepted": True}},
                memory_context="",
            )

        out = str(result["messages"][0].content)
        self.assertIn("Confirm send?", out)
        self.assertIn("movie", out.lower())
        self.assertIn("ticket", out.lower())

    def test_send_confirmation_edit_uses_research_context(self) -> None:
        seen = {"prompt_text": ""}

        def stub_llm(payload):
            seen["prompt_text"] = str(payload)
            return AIMessage(
                content=(
                    '{"subject":"T20 World Cup Winner 2026",'
                    '"body":"The 2026 winner details are included from research."}'
                )
            )

        llm = RunnableLambda(stub_llm)
        draft = {
            "pending": True,
            "to_name": "Praket",
            "to_email": "ppkinhome@gmail.com",
            "canonical_name": "Praket",
            "subject": "T20 update",
            "body": "Hello Praket!\n\nOld body text.\n\nThank you\nPragnesh Kumar",
            "tone": "introduce yourself",
            "include_sources": False,
        }
        messages = [
            ToolMessage(
                name="tavily_search",
                tool_call_id="tool-1",
                content="T20 World Cup 2026 winner details source https://example.com/winner",
            )
        ]

        with patch("mailer_agent.set_draft", lambda *args, **kwargs: None):
            result = _handle_send_confirmation(
                draft=draft,
                user_text="do the research on the internet and update the draft",
                user_key="test",
                chat_id="chat",
                messages=messages,
                llm=llm,
                identity_text="Arjun",
                edit_prompt=_edit_prompt(),
                default_tone="introduce yourself",
                signature_name="Pragnesh Kumar",
                sources_lines=[],
                sources_block_text="",
            )

        self.assertIn("research context", seen["prompt_text"].lower())
        self.assertIn("t20 world cup", seen["prompt_text"].lower())
        self.assertIn("example.com/winner", seen["prompt_text"].lower())
        self.assertIn("Confirm send?", str(result["messages"][0].content))


if __name__ == "__main__":
    unittest.main()
