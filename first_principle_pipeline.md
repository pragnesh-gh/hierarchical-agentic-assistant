# First Principles Pipeline

This document explains what happens when you type a question and press enter,
step by step, in plain English. It covers the full pipeline, the decisions
we made, and the reasoning behind them.

---

## The Core Idea

An agent is: **LLM + Memory + Tools**.

Our system is not one agent. It is four agents working together in a hierarchy:

```
You (the user)
  |
  v
Planner (the boss — decides what to do)
  |
  +---> Researcher (the worker — gathers evidence using tools)
  |         |
  |         +---> PDF tool (searches the book)
  |         +---> Web tool (searches the internet)
  |
  +---> Answerer (the writer — turns evidence into a final answer)
  |
  +---> Mailer (the assistant — drafts and sends emails)
```

The Planner never answers questions directly. It just makes a plan.
The Researcher never writes final answers. It just gathers evidence.
The Answerer never calls tools. It just writes based on what the Researcher found.

This separation is the "hierarchical" part. Each role does one thing well.

---

## What Happens When You Ask a Question

### Step 1: Session Loading

**Where:** `run_cli.py`, lines 512-534

Before the graph even starts, the system loads your context:

```
chat_id       = get_active_chat("cli")          # which conversation thread you're in
history       = load_messages("cli", chat_id)    # your previous messages in this chat
draft         = get_draft("cli", chat_id)        # any pending email draft
flags         = get_flags("cli", chat_id)        # control toggles (like followup_reset)
task_state    = get_task_state("cli", chat_id)   # structured memory (last answer, email frame)
memory_context = retrieve_memory_context(...)    # long-term memory facts relevant to your question
```

**Why this matters:** Without this, every question would start from scratch.
The agent would not know what you asked 2 minutes ago. `task_state` carries
things like your last accepted answer (so the mailer can "send this info").
`memory_context` carries long-term facts from older chats.

**Key variable:** `AgentState` — this is the single shared object that all
agents read and write during one turn. Defined in `state.py`.

### Step 2: The Planner Decides What To Do

**Where:** `planner_agent.py`, the `supervisor()` orchestrator (delegates to `_resolve_intent_and_state()`, `_handle_active_draft()`, `_generate_plan()`, `_route_next_step()`)

The planner's job is to look at your question and produce a plan — a list
of steps like:

```json
[
  {"id": 0, "action": "researcher", "tools": ["retrieve_context"]},
  {"id": 1, "action": "answerer"}
]
```

This means: first, search the PDF. Then, write the answer.

#### How the planner decides (two paths):

**Fast-path (no LLM call):**
Most questions are predictable. If your question contains "deep work" or
"Cal Newport", we know it needs the PDF tool. If it contains "latest" or
"today", we know it needs the web tool. If it is "hello", we know it is
small talk.

The function `classify_query_source()` in `guardrails.py` does this check.
It is just keyword matching — no model needed, takes <1ms.

For these clear cases, the planner skips the LLM entirely and builds a
deterministic fallback plan. This saves 2-5 seconds of model inference time.

**LLM-planned (model call):**
For ambiguous questions (where keywords do not clearly indicate pdf, web,
or email), the planner calls the LLM. The LLM outputs raw text that should
be JSON. We then:
1. Try to parse it as JSON
2. If that fails, search for `{...}` brackets in the text
3. Run `_repair_plan()` to fix common issues (missing tools, wrong step count)
4. Run `_validate_plan()` to verify the structure
5. If all that fails, fall back to a safe default plan

**New: Planner reasoning capture**
We now save the raw LLM output (or the fast-path label) in
`state["planner_reasoning"]`. This goes into the JSONL trace.

**Why we added this:** If something goes wrong later, we can look at the
trace and see exactly what the planner was thinking. Was it a fast-path
decision? Or did the LLM produce garbled JSON that got repaired into
something wrong? Without this, all we had was the final plan — the
"what", not the "why".

### Step 3: The Researcher Gathers Evidence

**Where:** `researcher_agent.py`

The researcher reads the plan and calls the specified tools:

- `retrieve_context` — searches the FAISS vector index built from `Deep_Work.pdf`.
  Returns page-tagged text chunks like `[p.42] Deep work requires focus...`
- `tavily_search` — calls the Tavily API for live web results.
  Returns URLs and snippets.

**Parallel execution (async):**
When the plan says to call both tools, they run in parallel using
`ThreadPoolExecutor`. Each tool has a timeout:

```python
ASYNC_TIMEOUT_PDF_SEC = 15    # PDF search max wait
ASYNC_TIMEOUT_WEB_SEC = 20    # web search max wait
```

**Why parallel:** If PDF search takes 3 seconds and web search takes 5 seconds,
running them one after another costs 8 seconds. Running them at the same time
costs only 5 seconds. On a laptop, this matters.

**Important:** The researcher never writes the final answer. It just puts tool
results into the message history for the answerer to use.

### Step 4: The Supervisor Loops Back

**Where:** `graph.py`, the `supervisor_router` function

After the researcher finishes, control goes back to the supervisor (planner).
The supervisor increments the step pointer and routes to the next step.

```
step_index was 0 (researcher) → now becomes 1 (answerer)
```

This is the graph structure:

```
supervisor → researcher → research_tools → supervisor → answerer → END
                                              ↑
                                    (loops back for each step in the plan)
```

### Step 5: The Answerer Writes the Final Response

**Where:** `answer_agent.py`

The answerer gets all the messages so far (your question, tool outputs,
memory context) and writes a final answer.

Rules enforced in the prompt:
- Use only information from the provided context (no making things up)
- If tool evidence and memory conflict, prefer tool evidence
- Do not include a Sources section (that is appended separately by code)

After the LLM generates the answer, the code:
1. Strips any `<think>` tags (internal model reasoning)
2. Extracts PDF citations (`[p.X]`) and web URLs from tool messages
3. Appends a clean Sources section
4. Stores the answer in `task_state["last_answer"]` (so the mailer can use it later)

### Step 6: Post-Graph Evaluation (New)

**Where:** `run_cli.py`, after streaming completes

After the user sees the answer, three evaluation steps run:

#### 6a. Guardrail checks
Already existed. Checks structural things: was the plan parseable? Was the
right tool called? Are citations present?

#### 6b. Failure mode classification (new)

**Where:** `guardrails.py`, `classify_failure()`

Takes the guardrail check results and assigns a single human-readable category:

```
plan_generation_failure  — planner could not produce a valid plan
routing_error            — wrong tool was selected for the question type
tool_skip                — answered without calling any tool first
citation_missing         — used PDF but forgot to cite page numbers
source_missing           — used web but forgot to include URLs
pass                     — everything looks good
```

**Why we added this:** Before, we had 6 boolean flags. After 100 runs,
you would have a table of True/False values that is hard to read.
Now you can just count: "42 passes, 3 routing errors, 5 citation missing".
You instantly know where to focus improvement effort.

**Why the priority order matters:** If the plan itself is broken,
it does not matter if the tool choice was wrong. So we check plan issues
first, then routing, then tool execution, then output formatting. Each
failure has one label, not five.

#### 6c. Groundedness check (new)

**Where:** `guardrails.py`, `check_groundedness()`

This is the most important new addition. It answers the question:
**"Did the answer make claims that are not actually in the evidence?"**

How it works:
1. Extract all text from tool responses (PDF chunks, web results)
2. Take the answer text (without the Sources section)
3. Send both to an LLM with this prompt:

```
Given these SOURCES and this ANSWER,
is every claim in the ANSWER supported by the SOURCES?
Return: {"verdict": "grounded" or "ungrounded" or "partial", "reason": "..."}
```

4. Parse the response and store it in the JSONL trace

**Why this runs after the graph, not inside it:**
We tried putting it inside the answerer node first. The problem:
when the judge model generates tokens, LangGraph's streaming picks them up
and prints them to the user mixed in with the actual answer. Moving the
check to after the graph finishes completely avoids this. The user sees a
clean answer. The judge runs silently. The result goes only into the trace.

**Why we reuse the answerer model for judging:**
On a 4GB VRAM laptop, two different models cannot be loaded at the same time.
If the answerer is `qwen3.5:2b` and the judge is `qwen3.5:4b`, Ollama has to
unload one and load the other. This takes time and sometimes crashes.

By defaulting the judge to the same model as the answerer, we reuse what is
already in VRAM. Zero swap overhead. You can override this with the
`GROUNDEDNESS_MODEL` environment variable on better hardware.

**Context window for the judge (`GROUNDEDNESS_NUM_CTX = 1024`):**
The judge only needs to read: the answer (~200 words) + tool outputs (~500 words).
1024 tokens is enough. We do not need the full 3072 context window that the
answerer uses. Smaller context = faster inference = less memory.

### Step 7: Trace Writing

**Where:** `run_cli.py`, the `_stream_run()` function and the post-graph section

Every step in the graph writes one JSONL line to `runs/run_YYYYMMDD_HHMMSS.jsonl`:

```json
{
  "run_id": "20260310_100716",
  "step_index": 0,
  "role": "planner",
  "model": "qwen3.5:2b",
  "action": "plan",
  "tool_calls": [],
  "tool_results": [],
  "latency_ms": 11505.86,
  "guardrail_checks": { ... },
  "failure_mode": "pass",
  "planner_reasoning": "[fast-path] source=pdf",
  "groundedness_check": null,
  "errors": null,
  "plan": [ ... ],
  "messages": [ ... ]
}
```

The groundedness judge writes a final separate entry:

```json
{
  "run_id": "20260310_100716",
  "step_index": -1,
  "role": "groundedness_judge",
  "model": "qwen3.5:2b",
  "action": "groundedness_check",
  "groundedness_check": {
    "verdict": "grounded",
    "reason": "All claims supported by source text."
  }
}
```

### Step 8: Persist and Remember

**Where:** `run_cli.py`, end of `main()`

```python
save_messages("cli", chat_id, final_state["messages"], question[:80])
set_task_state("cli", chat_id, final_state["task_state"])
ingest_turn_memory("cli", chat_id, question, final_text)
```

- Messages are saved so next turn has conversation history
- Task state is saved so the mailer can reference `last_answer`
- The turn is ingested into graph memory for long-term recall

---

## Key Configuration Variables

All in `app/config.py`. Override any of them via `.env` or environment variables.

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `MODEL_PRESET` | `production` | Which model set: `production` (2b), `max_accuracy` (4b), `speed` (1.7b) |
| `PLANNER_MODEL` | from preset | Model for plan generation |
| `RESEARCHER_MODEL` | from preset | Model for tool selection |
| `ANSWERER_MODEL` | from preset | Model for answer writing |
| `TEMPERATURE` | `0.2` | Low = more deterministic. We want consistent answers. |
| `NUM_CTX` | `3072` | Global context window (tokens). Higher = more memory but can see more. |
| `PLANNER_NUM_CTX` | `2048` | Planner gets less context. Plans are short, and lower context = faster. |
| `GROUNDEDNESS_MODEL` | (empty = reuse answerer) | Judge model. Set to `qwen3.5:4b` on better hardware. |
| `GROUNDEDNESS_NUM_CTX` | `1024` | Judge needs minimal context. Keeps it fast. |
| `GROUNDEDNESS_TEMPERATURE` | `0.1` | Very low. We want the judge to be as consistent as possible. |
| `ASYNC_TOOLS` | `True` | Run multiple tools in parallel (PDF + web at the same time) |
| `ASYNC_TIMEOUT_PDF_SEC` | `15` | Max wait for PDF search |
| `ASYNC_TIMEOUT_WEB_SEC` | `20` | Max wait for web search |
| `KEEP_ALIVE` | `20m` | How long Ollama keeps a model loaded in VRAM after use |
| `GRAPH_MEMORY_ENABLED` | `True` | Long-term memory on/off |
| `GRAPH_MEMORY_TOP_K` | `6` | How many memory facts to retrieve per question |

---

## Why We Made These Decisions

### "Why not use one big model for everything?"

Because each role needs different things:
- The planner needs to be fast (it runs on every turn). A 2b model is enough for routing.
- The researcher does not need to be smart — it just needs to call tools correctly.
- The answerer needs the most capability — it writes the final prose.
- The judge needs to be consistent, not creative. Low temperature, small context.

With separate models per role, you can optimize each one independently.
A 4b answerer with a 1.7b planner gives better results than a 4b model doing everything,
because the planner context window stays small and inference stays fast.

### "Why fast-path routing instead of always calling the LLM?"

Because most questions are obvious. "Summarize deep work" clearly needs the PDF.
"What's the latest cricket score" clearly needs the web. Calling the LLM to figure
this out wastes 2-5 seconds on every single turn.

The fast-path uses `classify_query_source()` — a set of keyword rules that takes <1ms.
The LLM is only called when the keywords are ambiguous (the "unknown" case).

### "Why did we add failure mode classification?"

Before, a failed run gave you 6 booleans:
```
plan_parseable: True, tool_choice_correct: False, tool_called: True, ...
```
You had to mentally figure out what went wrong. With 100 failed runs, this is
impossible to analyze.

Now you get one label: `routing_error`. You can count these across runs.
"70% of failures are routing errors" tells you exactly what to fix:
the tool selection logic, not the plan generation or the citation formatting.

### "Why capture planner reasoning?"

The plan tells you WHAT the planner decided. The reasoning tells you WHY.

If the plan says `[researcher → answerer]` and the tools list says `[tavily_search]`,
but the question was about the book — that is a routing error. But was it because:
- The fast-path misclassified the query? (check: `planner_reasoning` says `[fast-path] source=web`)
- The LLM produced a wrong plan? (check: `planner_reasoning` shows the raw model output)
- The plan was correct but got corrupted during repair? (check: compare raw output vs final plan)

Without the reasoning, you can only see the symptom. With it, you can find the cause.

### "Why groundedness check? And why inline?"

The groundedness check answers the most important question in any RAG system:
**"Did the model make stuff up?"**

The answer might sound confident and well-written, but contain a claim that
is not in any of the tool outputs. This is called a **hallucination** (the term
used in AI for when a model generates plausible-sounding but wrong information).

We run it inline (before saving) rather than as a separate batch process because:
- You get immediate feedback in the trace
- The eval runner can include it per-question
- It uses the model already loaded in VRAM, so cost is just a few extra seconds

We show the verdict to the user only when it is NOT grounded (partial or ungrounded).
If everything is fine, the user sees nothing extra. No noise.

### "Why did the groundedness check move from inside the graph to after it?"

We first tried putting `check_groundedness()` inside the answerer node.
The problem: when the judge model generates its response tokens, LangGraph's
streaming system picks them up and displays them mixed into the user's answer.
The user would see the answer text followed by raw JSON like
`{"verdict": "partial", "reason": "..."}`.

Moving it after the graph finishes means the streaming is already complete.
The judge runs silently. The user sees a clean answer. The verdict goes only
into the JSONL trace (and optionally prints a one-line summary for non-grounded cases).

---

## File Map (updated)

```
app/
  config.py              Settings: paths, model names, context windows, timeouts
  state.py               AgentState + 6 TypedDicts (TaskState, DraftState, etc.)
  graph.py               Wires nodes + edges into LangGraph. Creates groundedness LLM.
  planner_agent.py       Supervisor orchestrator + 4 focused helpers
  researcher_agent.py    Tool caller: PDF retrieval, web search (parallel)
  answer_agent.py        Final answer synthesis with source extraction
  mailer_agent.py        Email draft/edit/send with confirmation flow
  guardrails.py          Pass/fail checks + failure classification + groundedness judge
  vocabulary.py          Single source of truth for intent markers (5 constant sets)
  intent_utils.py        Shared helpers: effective_query, detect_email_intent, etc.
  turn_controller.py     Turn-level intent classification
  chat_sessions.py       SessionCache context manager + threading lock
  graph_memory.py        Long-term memory with logging at all fallback paths
  chat_intel.py          Topic shift detection heuristic
  tools_pdf.py           FAISS-backed PDF retrieval (cached singleton)
  tools_web.py           Tavily web search wrapper
  tools_email.py         Gmail send via OAuth
  contacts.py            Allowlist contact resolution with aliases
  identity.py            Assistant identity text loader
  redaction.py           PII/secret masking before logging
  metrics.py             Timing helpers (now_ms, duration_ms)
  run_cli.py             CLI entry point: graph execution, streaming, tracing
  eval_runner.py         Batch evaluation across model presets
  role_benchmark.py      Per-role model benchmarking
  build_index.py         Builds the FAISS index from PDF

eval/
  questions.jsonl        10 test questions (pdf, web, hybrid, email)
  rubric.py              Applies guardrail checks to eval runs
  results/               Timestamped eval JSON files

tests/
  test_repair_plan.py          Plan repair invariants (51 tests, 10 invariants)
  test_eval_improvements.py    Failure classification + groundedness (19 tests)
  test_graph_memory.py         Memory retrieval scope and scoring (7 tests)
  test_chat_intel.py           Topic shift detection (2 tests)
  test_chat_sessions_features.py  Chat management (2 tests)
  test_mailer_features.py      Email fact-transfer and identity (2 tests)

runs/                    JSONL traces, one file per CLI run
data/                    PDF, FAISS index, chat sessions, contacts, memory
mcp/telegram_server/     Telegram bot frontend
```

---

## How This Connects to the Corti Research

We studied Corti's agentic framework and found three things they have not built:

1. **No orchestrator decision quality metrics** — their orchestrator picks experts
   but there is no record of WHY, and no way to score if the choice was right.
   → We added `planner_reasoning` and `failure_mode` to solve this for our system.

2. **No output quality scoring** — their system knows a task completed but not
   whether the output was actually correct or hallucinated.
   → We added `check_groundedness` which is inspired by their FactsR system
   (which uses a draft→evaluate→refine loop for clinical documentation, but
   only exists on their text generation side, not on the agentic side).

3. **No failure taxonomy** — their `TaskState` has a single `failed` state
   with no subcategories.
   → We added `classify_failure()` with 6 distinct categories so you can
   analyze patterns across runs.

The key insight from Corti: they already proved the "judge model" pattern
works in healthcare (FactsR gets 94% groundedness). We applied the same
concept — a separate evaluation step that checks output quality — to the
orchestrator layer where they have not done it yet.
