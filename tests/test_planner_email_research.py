import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from planner_agent import _email_requires_research  # noqa: E402


class TestPlannerEmailResearch(unittest.TestCase):
    def test_no_research_for_simple_reminder_email(self) -> None:
        text = "send a mail to pragnesh about the movie next week and remind him to book tickets"
        self.assertFalse(_email_requires_research(text))

    def test_research_for_freshness_requests(self) -> None:
        text = "send an email to pragnesh with the latest AI news today"
        self.assertTrue(_email_requires_research(text))

    def test_research_for_explicit_web_request(self) -> None:
        text = "send an email to pragnesh about marty supreme and use the web"
        self.assertTrue(_email_requires_research(text))


if __name__ == "__main__":
    unittest.main()
