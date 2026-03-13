import secrets
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from graph_memory import LocalGraphMemoryBackend  # noqa: E402


class TestGraphMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = REPO_ROOT / "tests" / ".tmp"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.mem_path = self.tmp_root / f"graph_memory_{secrets.token_hex(6)}.json"
        self.backend = LocalGraphMemoryBackend(self.mem_path, max_facts_per_user=50)

    def tearDown(self) -> None:
        try:
            self.mem_path.unlink(missing_ok=True)
        except Exception:
            pass

    def test_retrieves_relevant_turn(self) -> None:
        self.backend.ingest(
            user_key="u1",
            chat_id="c1",
            human="When is Ustaad Bhagat Singh releasing?",
            ai="It is scheduled for theatrical release on March 26, 2026.",
        )
        self.backend.ingest(
            user_key="u1",
            chat_id="c1",
            human="What is Deep Work about?",
            ai="Deep Work is about focused, distraction-free work.",
        )

        hits = self.backend.retrieve(
            user_key="u1",
            chat_id="c1",
            query="send this information to pragnesh about ustaad bhagat singh date",
            top_k=3,
        )
        self.assertTrue(hits)
        top = hits[0]["text"].lower()
        self.assertIn("ustaad bhagat singh", top)
        self.assertIn("march 26, 2026", top)

    def test_isolates_users(self) -> None:
        self.backend.ingest(
            user_key="u1",
            chat_id="c1",
            human="Remember this private note",
            ai="Private for user one.",
        )
        self.backend.ingest(
            user_key="u2",
            chat_id="c9",
            human="Remember this other note",
            ai="Private for user two.",
        )

        hits_u1 = self.backend.retrieve("u1", "c1", "private note", top_k=5)
        text_blob = " ".join(hit.get("text", "") for hit in hits_u1).lower()
        self.assertIn("user one", text_blob)
        self.assertNotIn("user two", text_blob)

    def test_max_facts_trim_per_user(self) -> None:
        small_backend = LocalGraphMemoryBackend(self.mem_path, max_facts_per_user=3)
        for idx in range(6):
            small_backend.ingest(
                user_key="trim-user",
                chat_id="c1",
                human=f"Question {idx}",
                ai=f"Answer {idx}",
            )
        hits = small_backend.retrieve("trim-user", "c1", "question", top_k=10)
        blob = " ".join(hit.get("text", "") for hit in hits).lower()
        self.assertNotIn("question 0", blob)
        self.assertIn("question 5", blob)

    def test_deictic_followup_prefers_factual_turn_over_smalltalk(self) -> None:
        self.backend.ingest(
            user_key="u1",
            chat_id="c1",
            human="When is Ustaad Bhagat Singh releasing?",
            ai="Ustaad Bhagat Singh releases on March 26, 2026.\n\nSources:\n- https://example.com/ustaad",
        )
        self.backend.ingest(
            user_key="u1",
            chat_id="c1",
            human="Hey how are you?",
            ai="I am doing well, thank you. How can I help you today?\n\nSources: None",
        )

        hits = self.backend.retrieve(
            user_key="u1",
            chat_id="c1",
            query="send this information to pragnesh and ask when he is free to watch it",
            top_k=2,
        )
        self.assertTrue(hits)
        top = hits[0]["text"].lower()
        self.assertIn("ustaad bhagat singh", top)
        self.assertIn("march 26, 2026", top)

    def test_summary_query_retrieves_multiple_topics(self) -> None:
        self.backend.ingest(
            user_key="u1",
            chat_id="c1",
            human="What happened with OpenAI today?",
            ai="OpenAI announced a new feature for enterprise customers.",
        )
        self.backend.ingest(
            user_key="u1",
            chat_id="c1",
            human="Any Claude news?",
            ai="Claude service had a temporary outage and then recovered.",
        )
        self.backend.ingest(
            user_key="u1",
            chat_id="c1",
            human="More OpenAI detail?",
            ai="OpenAI also published pricing changes.",
        )

        hits = self.backend.retrieve(
            user_key="u1",
            chat_id="c1",
            query="summarize today news updates about openai and claude",
            top_k=3,
        )
        self.assertTrue(hits)
        blob = " ".join(hit.get("text", "") for hit in hits).lower()
        self.assertIn("openai", blob)
        self.assertIn("claude", blob)

    def test_default_retrieval_scoped_to_current_chat(self) -> None:
        self.backend.ingest(
            user_key="u1",
            chat_id="chat-a",
            human="OpenAI update?",
            ai="OpenAI announced a new deployment deal.",
        )
        self.backend.ingest(
            user_key="u1",
            chat_id="chat-b",
            human="Tesla robots?",
            ai="Tesla Optimus Gen 3 is targeted for Q1 2026.",
        )

        hits_same_chat = self.backend.retrieve(
            user_key="u1",
            chat_id="chat-a",
            query="tesla robots latest",
            top_k=5,
        )
        self.assertFalse(hits_same_chat)

        hits_cross_chat = self.backend.retrieve(
            user_key="u1",
            chat_id="chat-a",
            query="from previous chat tesla robots latest",
            top_k=5,
        )
        self.assertTrue(hits_cross_chat)
        blob = " ".join(hit.get("text", "") for hit in hits_cross_chat).lower()
        self.assertIn("tesla optimus gen 3", blob)


if __name__ == "__main__":
    unittest.main()
