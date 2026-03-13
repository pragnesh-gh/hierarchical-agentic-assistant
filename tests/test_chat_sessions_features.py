import secrets
import sys
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import chat_sessions  # noqa: E402


class TestChatSessionFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = REPO_ROOT / "tests" / ".tmp"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.orig_path = chat_sessions.SESSIONS_PATH
        self.test_path = self.tmp_root / f"sessions_{secrets.token_hex(5)}.json"
        chat_sessions.SESSIONS_PATH = self.test_path

    def tearDown(self) -> None:
        chat_sessions.SESSIONS_PATH = self.orig_path
        try:
            self.test_path.unlink(missing_ok=True)
        except Exception:
            pass

    def test_titles_and_selector(self) -> None:
        user = "u_test"
        chat_id = chat_sessions.new_chat(user)
        msgs = [
            HumanMessage(content="OpenAI and Tesla updates from today"),
            AIMessage(content="Here are the latest updates."),
        ]
        chat_sessions.save_messages(user, chat_id, msgs, "OpenAI and Tesla updates from today")
        rows = chat_sessions.list_chats(user)
        self.assertTrue(rows)
        self.assertIn("title", rows[0])
        self.assertTrue(str(rows[0]["title"]).strip())

        resolved = chat_sessions.resolve_chat_selector(user, "1")
        self.assertEqual(resolved, rows[0]["id"])

    def test_rename_and_search(self) -> None:
        user = "u_test"
        chat_id = chat_sessions.new_chat(user)
        ok = chat_sessions.rename_chat(user, chat_id, "My Important Strategy Chat")
        self.assertTrue(ok)

        rows = chat_sessions.search_chats(user, "strategy", limit=10)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["id"], chat_id)


if __name__ == "__main__":
    unittest.main()

