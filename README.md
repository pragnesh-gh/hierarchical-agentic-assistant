# Hierarchical Agentic QA

Hierarchical Agentic QA is a local-first multi-agent assistant built with LangGraph.

It routes each user turn through specialized roles:
- `planner` (intent + plan)
- `researcher` (tool calls)
- `answerer` (final answer)
- `mailer` (draft + confirm + send)

The project supports:
- PDF RAG (`Deep Work`)
- live web search
- Gmail email drafting/sending with confirmation
- Telegram and CLI frontends
- per-user multi-chat sessions
- long-term graph memory with chat-scoped retrieval by default

For full internals, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Key Features
- Role-based orchestration using LangGraph state transitions.
- Deterministic-safe email confirmation flow (`yes/no`) before send.
- Chat history with titles, rename, search, and switching.
- Topic-shift tip suggesting `/new_chat` when likely needed.
- Long-term memory retrieval constrained to current chat unless explicit cross-chat request.
- Inline groundedness check: LLM-as-judge verifies answer claims against tool evidence.
- Failure mode classification: every run is categorized (routing_error, tool_skip, etc.).
- Planner reasoning capture: raw LLM output saved before JSON parsing for trace analysis.

For the full pipeline explained from first principles, see [first_principle_pipeline.md](./first_principle_pipeline.md).

## Repository Layout
```text
app/
  config.py
  state.py                    # AgentState + TypedDicts (TaskState, DraftState, etc.)
  graph.py
  planner_agent.py            # supervisor orchestrator + 4 focused helpers
  researcher_agent.py
  answer_agent.py
  mailer_agent.py
  chat_sessions.py            # SessionCache context manager + threading lock
  graph_memory.py             # logging at all fallback transitions
  vocabulary.py               # single source of truth for intent markers
  chat_intel.py
  intent_utils.py
  turn_controller.py
  guardrails.py
  tools_pdf.py
  tools_web.py
  tools_email.py
  run_cli.py
mcp/telegram_server/
  bot.py
tests/
  test_repair_plan.py         # 51 tests for plan repair invariants
  test_eval_improvements.py
  test_graph_memory.py
  test_chat_intel.py
  test_chat_sessions_features.py
  test_mailer_features.py
data/
  Deep_Work.pdf               # local source document, gitignored
  faiss_index/                # local generated index, gitignored
  chat_sessions.json          # local runtime state, gitignored
  graph_memory.json           # local runtime state, gitignored
  contacts_allowlist.json     # local personal data, gitignored
eval/
  questions.jsonl
  rubric.py
  results/
runs/                         # local trace logs, gitignored
```

## Requirements
- Python environment (project commonly used with `conda activate ollama`)
- Local Ollama runtime
- Optional Tavily API key for web search
- Optional Gmail OAuth files for sending email
- Optional Telegram bot token

Suggested packages:
```bash
pip install langchain langchain-core langchain-community langchain-ollama langchain-tavily langgraph pypdf faiss-cpu python-dotenv python-telegram-bot openai-whisper google-api-python-client google-auth-httplib2 google-auth-oauthlib graphiti-core
```

## Quick Start
1. Build the PDF index:
```bash
python app/build_index.py
```

2. Run CLI:
```bash
python app/run_cli.py
```

3. Run Telegram bot:
```bash
python mcp/telegram_server/bot.py
```

## CLI Commands
- `/contacts`
- `/draft`
- `/new_chat`
- `/history`
- `/open <index_or_chat_id>`
- `/rename <index_or_chat_id> <new_title>`
- `/search <keyword>`
- `/switch <chat_id>`
- `/stop` or `/exit`
- `/email to=<name_or_email> body="..." tone="..."`

## Telegram Commands
- `/start`
- `/status`
- `/contacts`
- `/draft`
- `/new_chat`
- `/history`
- `/open <index_or_chat_id>`
- `/rename <index_or_chat_id> <new_title>`
- `/search <keyword>`
- `/switch <chat_id>`
- `/stop` or `/exit`
- `/email to=<name_or_email> body="..." tone="..."`

`/history` also includes inline buttons to open recent chats.

## Memory Model
- `chat_sessions.json`: short-term per-chat conversational state.
- `graph_memory.json`: long-term graph-like memory store (fallback backend).
- Default retrieval scope: current `chat_id`.
- Cross-chat retrieval occurs only when user explicitly asks (for example: `from previous chat`).

## Email Behavior
- Email always goes through draft confirmation.
- Natural-language and explicit `/email` paths supported.
- Direct recipient email in message text is supported (for example: `send raj@example.com a summary`).
- Outbound emails include identity line: `I am Arjun, Pragnesh's AI assistant.`

## Model and Runtime Config
Main settings are in `app/config.py` and `.env`.

Important toggles:
- `MODEL_PRESET` (`production`, `max_accuracy`, `speed`)
- `PLANNER_MODEL`, `RESEARCHER_MODEL`, `ANSWERER_MODEL`, `MAILER_MODEL`
- `NUM_CTX`, role-specific `*_NUM_CTX`
- `ASYNC_TOOLS`, `ASYNC_PERSIST`, `TELEGRAM_PROGRESS`
- `GRAPH_MEMORY_ENABLED`, `GRAPH_MEMORY_BACKEND`, `GRAPH_MEMORY_TOP_K`
- `GRAPHITI_URI`, `GRAPHITI_USER`, `GRAPHITI_PASSWORD` (optional Graphiti backend)

## Evaluation
Run all tests (98 tests including evaluation, observability, and plan repair invariants):
```bash
python -m pytest tests/ -v
```

Run benchmark scripts:
```bash
python app/role_benchmark.py
python app/bench_qwen35.py
```

Run the eval suite (includes groundedness checks):
```bash
EVAL_MODE=smoke python app/eval_runner.py   # 3 questions, fast
EVAL_MODE=small python app/eval_runner.py   # all questions, 1 preset
EVAL_MODE=full python app/eval_runner.py    # all questions, multi-preset
```

Override the groundedness judge model on better hardware:
```bash
GROUNDEDNESS_MODEL=qwen3.5:4b python app/run_cli.py
```

## Gmail Setup
1. Put OAuth credentials in `secrets/credentials.json`.
2. Generate token:
```bash
python app/gmail_oauth.py
```
3. Keep `data/contacts_allowlist.json` updated for named contact resolution.

## Notes
- `app/memory.py` is legacy; active session memory is handled by `chat_sessions.py`.
- JSONL run traces are written to `runs/`.
- See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed control flow and data flow.
