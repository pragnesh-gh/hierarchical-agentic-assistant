# Architecture

This document describes how Hierarchical Agentic QA is designed to work end-to-end.

## 1. High-Level Runtime

```text
User Input
  -> Session Load (messages, draft, flags, task_state)
  -> Graph Memory Retrieval (chat-scoped by default)
  -> LangGraph App
     -> supervisor (planner)
     -> researcher (optional)
     -> research_tools (optional)
     -> supervisor (loop)
     -> answerer or mailer
  -> Post-Graph Evaluation
     -> groundedness judge (inline, after streaming completes)
     -> failure mode classification
  -> Persist Session + Task State
  -> Write JSONL trace (with eval fields)
  -> Ingest turn into graph memory
  -> Return response
```

The system is role-based and stateful. Each turn runs through a shared `AgentState`.

## 2. Agent Roles

### supervisor (`planner_agent.py`)
Responsibilities:
- classify turn intent (`qa`, `email`, `smalltalk`, `confirm_send`, etc.)
- build/repair/validate plan JSON
- set `state["next"]`
- enforce route safety (for example draft confirmation routing)
- capture raw LLM reasoning before JSON parsing (stored as `planner_reasoning`)

The supervisor entry point is a ~40-line orchestrator that delegates to 4 focused helpers:
- `_resolve_intent_and_state()` — classifies intent, applies state mutations (retry/reset/email)
- `_handle_active_draft()` — handles draft fast-paths (confirm/edit/research+send), returns early or defers
- `_generate_plan()` — generates plan via fast-path or LLM with repair/validate/fallback chain
- `_route_next_step()` — advances step pointer, returns (next_action, next_index)

Output:
- `plan`
- `step_index`
- `next`
- `turn_intent`
- `planner_reasoning` (raw LLM output or fast-path label for trace analysis)

Planning modes:
- **Fast-path**: for high-confidence intents (conversational, clear pdf/web, email). Skips LLM entirely. Logged as `[fast-path] source=pdf` etc.
- **LLM-planned**: for ambiguous queries where the planner model is invoked. The raw output is captured verbatim before JSON parsing/repair.

### researcher (`researcher_agent.py`)
Responsibilities:
- read planned tools for current step
- call `retrieve_context` (PDF) and/or `tavily_search` (web)
- emit tool calls (parallel fan-out for multi-tool plan steps)

### answerer (`answer_agent.py`)
Responsibilities:
- synthesize final response from prior context + tool outputs
- append sources section
- update `task_state.last_answer`

### mailer (`mailer_agent.py`)
Responsibilities:
- parse recipient/body/tone intent
- manage draft state machine
- deterministic fact-transfer for `send this/same information/summary` style requests
- require confirmation before send
- send via Gmail tool

Draft stages:
- `recipient`
- `body`
- `pending` (confirm)

### groundedness judge (post-graph, `guardrails.py`)
Responsibilities:
- runs after the graph finishes and streaming completes (avoids token leakage into the displayed answer)
- compares the final answer text against tool outputs
- produces a verdict: `grounded`, `ungrounded`, `partial`, or error states
- result is written as a separate JSONL trace entry

## 3. State Model

Core runtime state (`state.py`):
- `messages`
- `plan`
- `step_index`
- `next`
- `draft` — typed as `DraftState` (TypedDict)
- `flags` — typed as `FlagState` (TypedDict)
- `task_state` — typed as `TaskState` (TypedDict, contains `EmailFrame`, `LastAnswer`, `RejectedAnswer`)
- `memory_context`
- `memory_backend`
- `turn_intent`
- `planner_reasoning` (raw planner output for trace analysis)
- `groundedness_check` (judge result dict)

TypedDicts in `state.py` enforce schema contracts:
- `EmailFrame` — stage, recipient, topic, body, pending_confirmation
- `LastAnswer` — text, sources, accepted
- `RejectedAnswer` — text, reason
- `TaskState` — active_task, email_frame, last_answer, rejected_answers, last_contact, preferences
- `DraftState` — stage, recipient, topic, body, pending, pending_confirmation
- `FlagState` — followup_reset

Persistent per-user/per-chat state (`chat_sessions.py`):
- active chat id
- chat metadata (id, title, created_at, last_active, preview)
- chat messages
- draft state
- flags
- task_state

Session I/O uses `SessionCache` — a context manager that loads once and saves once per turn, with `threading.Lock` for concurrent access safety. This replaced the previous pattern of load/save on every public function call.

## 4. Session and Chat Lifecycle

### Chat segmentation
- `/new_chat` creates a new isolated chat thread.
- each chat has its own short-term conversational state.
- `/history`, `/open`, `/rename`, `/search` provide chat management UX.

### Topic shift suggestion
A lightweight heuristic (`chat_intel.py`) can append a tip:
- `Tip: This looks like a new topic. Use /new_chat to keep chats segmented.`

This is advisory only; it does not force chat switching.

## 5. Memory Layers

### Short-term memory
Stored in `data/chat_sessions.json`.
- recent in-chat turns
- draft progression
- task frame (`last_answer`, `email_frame`, etc.)

### Long-term graph memory
Handled by `graph_memory.py`.
Backends:
- Graphiti (if configured and available)
- local JSON graph fallback

Default retrieval policy:
- scoped to current `chat_id`
- cross-chat retrieval only if user explicitly asks (for example: `from previous chat`)

Reason:
- minimizes topic contamination and incorrect follow-up merges.

## 6. Email Pipeline

### Entry methods
- natural language (`send this to pragnesh ...`)
- explicit command (`/email ...`)
- direct email in text (`send raj@example.com a summary ...`)

### Fact-transfer mode
Triggered by summary/deictic markers.
Behavior:
- assemble fact bundle from recent AI answers + `last_answer` + memory hints
- compose deterministic concise body
- add explicit availability question if requested
- avoid vague abstractions

### Identity and confirmation
- normalized body always includes: `I am Arjun, Pragnesh's AI assistant.`
- confirmation step always shown before send (`Reply yes/no`)

## 7. Tools

### PDF tool (`tools_pdf.py`)
- FAISS-backed retrieval over `Deep_Work.pdf`

### Web tool (`tools_web.py`)
- Tavily search for recent/current information

### Email tool (`tools_email.py`)
- Gmail send via OAuth token
- supports:
  - send by allowlisted contact name
  - send to explicit email address (post-confirmation)

## 8. Frontends

### CLI (`run_cli.py`)
- single-turn execution loop
- live token streaming for answerer output
- local commands for chat navigation and draft inspection
- post-graph groundedness check with verdict display

### Telegram (`mcp/telegram_server/bot.py`)
- polling bot
- progress message updates (`planning/research/writing`)
- chat management commands and history button callbacks
- per-chat lock to prevent overlapping requests

## 9. Evaluation and Observability

### Guardrails (`guardrails.py`)
Structural pass/fail checks run at every graph step:
- `plan_parseable`: did the planner produce a valid step list?
- `tool_called_before_response`: was evidence gathered before answering?
- `tool_choice_correct`: did the query type match the tools selected?
- `pdf_citations_present`: are `[p.X]` citations in the answer when PDF was used?
- `web_sources_present`: are URLs in the answer when web was used?

### Failure mode classification (`guardrails.py`)
Maps each guardrail result to a single human-readable failure category:
- `plan_generation_failure` — planner output could not be parsed
- `routing_error` — wrong tool selected for the query type
- `tool_skip` — answered without gathering evidence first
- `citation_missing` — used PDF but no page citations in answer
- `source_missing` — used web but no URLs in answer
- `pass` — all checks passed

Priority order matters: plan issues are checked first, then routing, then execution, then formatting.

### Groundedness check (`guardrails.py`)
An LLM-as-judge evaluation that compares the final answer against tool outputs:
- Prompt asks: "is every claim in the answer supported by the source evidence?"
- Verdicts: `grounded`, `ungrounded`, `partial`
- Runs inline (after graph completes, before response is saved)
- Judge model defaults to reusing the answerer model (avoids VRAM swap on low-memory hardware), overridable via `GROUNDEDNESS_MODEL` env var
- Context window set to 1024 tokens (answers + tool outputs should be <750 words)

### Planner reasoning capture
The raw LLM output from the planner is saved before JSON parsing:
- Fast-path decisions are logged as `[fast-path] source=pdf` etc.
- LLM-planned decisions capture the full model output verbatim
- Enables post-hoc analysis of why the planner chose a particular route

### JSONL traces (`runs/`)
Every graph step writes one JSON line with:
- `run_id`, `step_index`, `role`, `model`, `action`
- `tool_calls`, `tool_results`
- `latency_ms`
- `guardrail_checks` (the 6-field dict)
- `failure_mode` (single category string)
- `planner_reasoning` (raw planner output)
- `groundedness_check` (null during graph steps, populated in final judge entry)

The groundedness judge writes a separate final entry with `role: "groundedness_judge"`.

### Eval runner (`eval_runner.py`)
- Modes: `smoke` (3 questions), `small` (all questions, 1 preset), `full` (all questions, multi-preset sweep)
- Each question result includes: `guardrail_checks`, `failure_mode`, `planner_reasoning`, `groundedness_check`, `latency_ms`
- Results stored in `eval/results/` as timestamped JSON files

### Vocabulary (`vocabulary.py`)
Single source of truth for all intent detection markers used across modules:
- `SUMMARY_MARKERS` — phrases indicating "send this/previous answer"
- `DEICTIC_RE` — regex for deictic references ("this", "that", "it")
- `FRESHNESS_HINTS` — keywords indicating need for live/recent data
- `WEB_HINTS` — keywords indicating web search needed
- `CONVERSATIONAL_STARTS` — greetings and small talk patterns

Consumers: `mailer_agent.py`, `turn_controller.py`, `graph_memory.py`, `intent_utils.py`, `guardrails.py`, `planner_agent.py`

### Unit tests (`tests/`)
- `test_repair_plan.py`: 51 tests across 8 classes covering all plan repair invariants (10 invariants asserted per repair)
- `test_eval_improvements.py`: failure classification (8 cases), tool output extraction (3 cases), groundedness check (8 cases including error handling)
- `test_graph_memory.py`: fact retrieval, user isolation, trimming, cross-chat scope
- `test_chat_intel.py`: topic shift detection
- `test_chat_sessions_features.py`: chat creation, rename, search
- `test_mailer_features.py`: email fact-transfer, identity line

## 10. Operational Notes

- Traces/logs are written in JSONL under `runs/`.
- Async is used for I/O-heavy operations (tools/persistence), not uncontrolled parallel LLM planning.
- Role models are configured through `config.py` / `.env`.
- The groundedness judge is configured via `GROUNDEDNESS_MODEL`, `GROUNDEDNESS_NUM_CTX`, `GROUNDEDNESS_TEMPERATURE` in `config.py`.

## 11. Extension Points

Recommended future additions:
- archive/pin chat metadata and filters
- stronger summarization memory indexing per chat topic
- deterministic email templates per intent type
- richer Telegram history UI (pagination and categories)
- expand eval dataset beyond 10 questions
- per-node latency breakdown in traces
- cost/token tracking per role
- CI/CD regression pipeline using eval runner
