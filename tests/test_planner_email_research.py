import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from planner_agent import (  # noqa: E402
    _email_requires_research,
    _extract_email_topic,
    _has_topic_context,
)


class _FakeMessage:
    """Minimal stand-in for LangChain message objects."""

    def __init__(self, type_: str, content: str):
        self.type = type_
        self.content = content


class TestPlannerEmailResearch(unittest.TestCase):
    # ------------------------------------------------------------------
    # Original tests (unchanged)
    # ------------------------------------------------------------------
    def test_no_research_for_simple_reminder_email(self) -> None:
        text = "send a mail to pragnesh about the movie next week and remind him to book tickets"
        self.assertFalse(_email_requires_research(text))

    def test_research_for_freshness_requests(self) -> None:
        text = "send an email to pragnesh with the latest AI news today"
        self.assertTrue(_email_requires_research(text))

    def test_research_for_explicit_web_request(self) -> None:
        text = "send an email to pragnesh about marty supreme and use the web"
        self.assertTrue(_email_requires_research(text))

    # ------------------------------------------------------------------
    # Deictic guard: forwarding prior content → no research
    # ------------------------------------------------------------------
    def test_no_research_for_deictic_send_this(self) -> None:
        self.assertFalse(_email_requires_research("send this information to John"))

    def test_no_research_for_deictic_same_answer(self) -> None:
        self.assertFalse(_email_requires_research("share the same answer with Praket"))

    # ------------------------------------------------------------------
    # Instruction verb guard → no research
    # ------------------------------------------------------------------
    def test_no_research_for_remind(self) -> None:
        self.assertFalse(_email_requires_research("remind Praket about the birthday party"))

    def test_no_research_for_tell(self) -> None:
        self.assertFalse(_email_requires_research("tell John about the deadline tomorrow"))

    def test_no_research_for_book(self) -> None:
        self.assertFalse(_email_requires_research("email Praket and book the restaurant"))

    # ------------------------------------------------------------------
    # Generic topic (<2 keywords) → no research
    # ------------------------------------------------------------------
    def test_no_research_for_generic_topic(self) -> None:
        self.assertFalse(
            _email_requires_research("Send an email to John about the meeting")
        )

    # ------------------------------------------------------------------
    # Context-aware: topic with no context → research needed
    # ------------------------------------------------------------------
    def test_research_for_topic_no_context(self) -> None:
        text = "email Praket about the new Hail Mary movie"
        self.assertTrue(_email_requires_research(text, messages=[]))

    def test_research_for_cricket_tournament(self) -> None:
        text = "email John about the cricket tournament results"
        self.assertTrue(_email_requires_research(text, messages=[]))

    # ------------------------------------------------------------------
    # Context-aware: topic WITH existing context → no research
    # ------------------------------------------------------------------
    def test_no_research_when_context_exists(self) -> None:
        text = "email Praket about the new Hail Mary movie"
        messages = [
            _FakeMessage("ai", "The Hail Mary movie is based on the novel by Andy Weir.")
        ]
        self.assertFalse(_email_requires_research(text, messages=messages))

    def test_no_research_when_task_state_has_answer(self) -> None:
        text = "email Praket about the new Hail Mary movie"
        task_state = {
            "last_answer": {
                "text": "Project Hail Mary is a movie based on the book by Andy Weir.",
                "sources": [],
                "accepted": True,
            }
        }
        self.assertFalse(
            _email_requires_research(text, messages=[], task_state=task_state)
        )


class TestExtractEmailTopic(unittest.TestCase):
    def test_about_clause(self) -> None:
        self.assertEqual(
            _extract_email_topic("email Praket about the new Hail Mary movie"),
            "the new hail mary movie",
        )

    def test_regarding_clause(self) -> None:
        self.assertEqual(
            _extract_email_topic("send a mail regarding the cricket tournament"),
            "the cricket tournament",
        )

    def test_no_topic(self) -> None:
        self.assertEqual(_extract_email_topic("send this to John"), "")

    def test_strips_trailing_instruction(self) -> None:
        topic = _extract_email_topic(
            "email Praket about the movie and remind him to come"
        )
        self.assertEqual(topic, "the movie")


class TestHasTopicContext(unittest.TestCase):
    def test_generic_topic_returns_true(self) -> None:
        # "meeting" is a single keyword → too generic → returns True (skip research)
        self.assertTrue(_has_topic_context("the meeting", [], {}, ""))

    def test_no_context_returns_false(self) -> None:
        self.assertFalse(
            _has_topic_context("the new hail mary movie", [], {}, "")
        )

    def test_ai_message_provides_context(self) -> None:
        messages = [
            _FakeMessage("ai", "The Hail Mary movie is a sci-fi film.")
        ]
        self.assertTrue(
            _has_topic_context("the new hail mary movie", messages, {}, "")
        )

    def test_tool_message_provides_context(self) -> None:
        messages = [
            _FakeMessage("tool", "Project Hail Mary is a movie adaptation.")
        ]
        self.assertTrue(
            _has_topic_context("the new hail mary movie", messages, {}, "")
        )

    def test_planner_debug_messages_ignored(self) -> None:
        messages = [
            _FakeMessage("ai", "[Planner] hail mary movie route detected")
        ]
        self.assertFalse(
            _has_topic_context("the new hail mary movie", messages, {}, "")
        )

    def test_task_state_last_answer(self) -> None:
        task_state = {
            "last_answer": {"text": "The Hail Mary movie stars Ryan Gosling."}
        }
        self.assertTrue(
            _has_topic_context("the new hail mary movie", [], task_state, "")
        )

    def test_memory_context(self) -> None:
        memory = "- (1) User asked about the Hail Mary movie release"
        self.assertTrue(
            _has_topic_context("the new hail mary movie", [], {}, memory)
        )


if __name__ == "__main__":
    unittest.main()
