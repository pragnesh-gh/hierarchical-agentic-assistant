import sys
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from chat_intel import should_suggest_new_chat  # noqa: E402


class TestChatIntel(unittest.TestCase):
    def test_suggests_new_chat_on_topic_shift(self) -> None:
        history = [HumanMessage(content="Latest Tesla Optimus robot news today?")]
        self.assertTrue(
            should_suggest_new_chat(
                "Can you explain OAuth token refresh flow in Gmail API?",
                history,
            )
        )

    def test_no_suggestion_for_continuation(self) -> None:
        history = [HumanMessage(content="Latest Tesla Optimus robot news today?")]
        self.assertFalse(
            should_suggest_new_chat(
                "Send this information to Pragnesh and ask when he is free.",
                history,
            )
        )


if __name__ == "__main__":
    unittest.main()

