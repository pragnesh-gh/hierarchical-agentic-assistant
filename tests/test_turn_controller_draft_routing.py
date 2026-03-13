import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from turn_controller import _heuristic_control  # noqa: E402


class TestTurnControllerDraftRouting(unittest.TestCase):
    def test_recipient_stage_routes_to_edit_draft(self) -> None:
        control = _heuristic_control(
            latest_user="Pragnesh",
            draft={"stage": "recipient"},
            task_state={},
        )
        self.assertEqual(control.get("intent"), "edit_draft")

    def test_explicit_email_request_with_existing_draft_routes_to_email(self) -> None:
        control = _heuristic_control(
            latest_user="send a mail to pragnesh about next week movie tickets",
            draft={"pending": True, "to_name": "Pragnesh"},
            task_state={},
        )
        self.assertEqual(control.get("intent"), "email")

    def test_natural_confirm_with_existing_draft_routes_to_confirm_send(self) -> None:
        control = _heuristic_control(
            latest_user="go ahead and send it",
            draft={"pending": True, "to_name": "Pragnesh"},
            task_state={},
        )
        self.assertEqual(control.get("intent"), "confirm_send")


if __name__ == "__main__":
    unittest.main()
