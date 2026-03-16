"""Shared state schema for LangGraph nodes and messages."""

# lab4/state.py
from typing import TypedDict, Annotated, Literal, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class EmailFrame(TypedDict, total=False):
    stage: str
    recipient: str
    topic: str
    body: str
    pending_confirmation: bool


class LastAnswer(TypedDict, total=False):
    text: str
    sources: List[str]
    accepted: bool


class RejectedAnswer(TypedDict, total=False):
    text: str
    reason: str


class TaskState(TypedDict, total=False):
    active_task: str
    email_frame: EmailFrame
    last_answer: LastAnswer
    rejected_answers: List[RejectedAnswer]
    last_contact: str
    preferences: Dict[str, Any]


class DraftState(TypedDict, total=False):
    stage: str
    recipient: str
    topic: str
    body: str
    pending: bool
    pending_confirmation: bool


class FlagState(TypedDict, total=False):
    followup_reset: bool


# AgentState is the single shared object that all nodes read/write during one run.
# total=False means every key is optional, so each node can update only its own fields.
class AgentState(TypedDict, total=False):
    # Full conversation for this run. `add_messages` appends deltas instead of replacing.
    messages: Annotated[List[BaseMessage], add_messages]

    # Router output from supervisor: which node executes next.
    next: Literal["researcher", "answerer", "mailer", "FINISH"]

    # Planner output: normalized step list used by supervisor for deterministic routing.
    plan: List[Dict[str, Any]]

    # Index of last completed step. Starts at -1 before any step executes.
    step_index: int

    # Stable user identifier (CLI/Telegram).
    user_key: str

    # Active chat thread identifier under the user.
    chat_id: str

    # Current email draft frame (recipient, body, pending confirmation, etc.).
    draft: DraftState

    # Per-chat ephemeral control flags (for follow-up behavior toggles).
    flags: FlagState

    # Structured per-chat task memory (active task, last answer, rejected answers, etc.).
    task_state: TaskState

    # Turn-level controller classification (qa/email/retry/reset/etc.) for current input.
    turn_intent: str

    # Retrieved long-term memory hints injected into prompts.
    memory_context: str

    # Name of backend serving long-term memory (graphiti/local/disabled).
    memory_backend: str

    # Raw LLM output from planner before JSON parsing (for reasoning trace analysis).
    planner_reasoning: str

    # Inline groundedness check result from the answer judge.
    groundedness_check: Dict[str, Any]
