"""Contact allowlist utilities for email sending."""

import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from config import CONTACTS_ALLOWLIST


def load_contacts() -> List[Dict[str, object]]:
    if not CONTACTS_ALLOWLIST.exists():
        return []
    with open(CONTACTS_ALLOWLIST, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    allowed = data.get("allowed", []) if isinstance(data, dict) else []
    return allowed if isinstance(allowed, list) else []


def list_contacts() -> List[Dict[str, object]]:
    return load_contacts()


def resolve_contact(name: str) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    """Resolve a contact by exact name or alias (case-insensitive).

    Returns (match, matches). If multiple matches, match is None and matches
    contains all candidates for clarification.
    """
    normalized = name.strip().lower()
    if not normalized:
        return None, []

    normalized_compact = re.sub(r"[^a-z0-9]", "", normalized)

    matches: List[Dict[str, object]] = []
    fuzzy_candidates: List[Tuple[float, Dict[str, object]]] = []
    for contact in load_contacts():
        contact_name = str(contact.get("name", "")).strip()
        aliases = contact.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        all_names = [contact_name] + [str(a).strip() for a in aliases]
        all_names = [n for n in all_names if n]
        normalized_names = [n.lower() for n in all_names]
        compact_names = [re.sub(r"[^a-z0-9]", "", n.lower()) for n in all_names]
        if normalized in normalized_names or normalized_compact in compact_names:
            matches.append(contact)
            continue

        if normalized_compact:
            score = 0.0
            for candidate in compact_names:
                if not candidate:
                    continue
                ratio = SequenceMatcher(None, normalized_compact, candidate).ratio()
                if ratio > score:
                    score = ratio
            if score > 0:
                fuzzy_candidates.append((score, contact))

    if len(matches) == 1:
        return matches[0], []
    if len(matches) > 1:
        return None, matches

    # Fuzzy fallback for minor typos / transcription drift.
    if fuzzy_candidates:
        ranked = sorted(fuzzy_candidates, key=lambda item: item[0], reverse=True)
        best_score, best_contact = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score >= 0.70 and (best_score - second_score) >= 0.08:
            return best_contact, []
    return None, []
