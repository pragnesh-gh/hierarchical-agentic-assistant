"""Simple persistent chat memory stored in JSON."""

import json
from pathlib import Path
from typing import Dict, List, cast

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from config import DATA_DIR


MEMORY_PATH = DATA_DIR / "chat_memory.json"
MAX_TURNS = 6


def _load_raw() -> Dict[str, List[Dict[str, str]]]:
    if not MEMORY_PATH.exists():
        return {}
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _save_raw(data: Dict[str, List[Dict[str, str]]]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_history(user_key: str) -> List[BaseMessage]:
    data = _load_raw()
    items = data.get(user_key, [])
    items_list = cast(List[Dict[str, str]], items)
    messages: List[BaseMessage] = []
    for item in items_list:
        role = item.get("role")
        content = item.get("content", "")
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def save_turn(user_key: str, human: str, ai: str) -> None:
    data = _load_raw()
    items = data.get(user_key, [])
    items.append({"role": "human", "content": human})
    items.append({"role": "ai", "content": ai})
    if len(items) > MAX_TURNS * 2:
        items = items[-MAX_TURNS * 2 :]
    data[user_key] = items
    _save_raw(data)
