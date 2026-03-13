import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from intent_utils import parse_confirmation_intent  # noqa: E402


class TestConfirmationIntent(unittest.TestCase):
    def test_confirm_variants(self) -> None:
        cases = [
            "yes",
            "y",
            "yes please",
            "sure",
            "go ahead",
            "go ahead and send",
            "please send it",
            "looks good, send it",
            "approved",
            "confirm",
        ]
        for text in cases:
            self.assertEqual(parse_confirmation_intent(text), "confirm", msg=text)

    def test_decline_variants(self) -> None:
        cases = [
            "no",
            "nope",
            "don't send",
            "do not send this",
            "not now",
            "hold off",
            "cancel it",
            "stop",
            "exit",
        ]
        for text in cases:
            self.assertEqual(parse_confirmation_intent(text), "decline", msg=text)

    def test_ambiguous_and_edit(self) -> None:
        cases = [
            "yes but make it shorter",
            "change the tone to formal",
            "update subject line",
            "can you tweak this",
            "send a mail to pragnesh about tomorrow meeting",
            "",
        ]
        for text in cases:
            self.assertEqual(parse_confirmation_intent(text), "unknown", msg=text)

    def test_negative_wins_when_mixed(self) -> None:
        self.assertEqual(parse_confirmation_intent("yes, but don't send yet"), "decline")


if __name__ == "__main__":
    unittest.main()
