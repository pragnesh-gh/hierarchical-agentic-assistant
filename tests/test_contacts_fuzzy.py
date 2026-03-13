import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import contacts  # noqa: E402


class TestContactsFuzzy(unittest.TestCase):
    def test_fuzzy_resolve_contact_typo(self) -> None:
        fake_contacts = [
            {"name": "Pragnesh", "email": "p@example.com", "aliases": ["Me"]},
            {"name": "Praveen", "email": "v@example.com", "aliases": ["Dad"]},
        ]
        with patch("contacts.load_contacts", return_value=fake_contacts):
            match, matches = contacts.resolve_contact("Prakanish")

        self.assertFalse(matches)
        self.assertIsNotNone(match)
        self.assertEqual(str(match.get("name", "")), "Pragnesh")


if __name__ == "__main__":
    unittest.main()
