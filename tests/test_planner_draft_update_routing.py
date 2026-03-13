import sys
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from planner_agent import create_supervisor  # noqa: E402


class TestPlannerDraftUpdateRouting(unittest.TestCase):
    def test_draft_update_with_research_routes_researcher_then_mailer(self) -> None:
        supervisor = create_supervisor(llm=None)
        state = {
            "messages": [
                HumanMessage(
                    content="can you do the research about the winner using the internet and then update the draft"
                )
            ],
            "draft": {
                "pending": True,
                "to_name": "Praket",
                "to_email": "ppkinhome@gmail.com",
                "subject": "T20 update",
                "body": "Old body",
            },
            "flags": {},
            "task_state": {},
            "plan": [],
            "step_index": -1,
        }

        out = supervisor(state)
        self.assertEqual(out.get("next"), "researcher")
        plan = out.get("plan", [])
        self.assertTrue(isinstance(plan, list) and len(plan) >= 2)
        self.assertEqual(str(plan[0].get("action", "")), "researcher")
        self.assertIn("tavily_search", plan[0].get("tools", []))
        self.assertEqual(str(plan[-1].get("action", "")), "mailer")

    def test_draft_update_without_research_routes_mailer(self) -> None:
        supervisor = create_supervisor(llm=None)
        state = {
            "messages": [HumanMessage(content="please update the draft and make it shorter")],
            "draft": {
                "pending": True,
                "to_name": "Praket",
                "to_email": "ppkinhome@gmail.com",
                "subject": "T20 update",
                "body": "Old body",
            },
            "flags": {},
            "task_state": {},
            "plan": [],
            "step_index": -1,
        }

        out = supervisor(state)
        self.assertEqual(out.get("next"), "mailer")


if __name__ == "__main__":
    unittest.main()
