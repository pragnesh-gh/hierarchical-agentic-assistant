# Evaluation & Observability Improvements — TODO

## Point 1: Failure Mode Classification

**File:** `app/guardrails.py`
**Change:** Add `classify_failure(checks: dict) -> str` function.
Takes the existing `guardrail_checks` output dict and returns a single failure category string.

Categories:
- `plan_generation_failure` — plan was not parseable
- `routing_error` — wrong tool selected for the query type
- `tool_skip` — answered without calling any tool first
- `citation_missing` — used PDF tool but no [p.X] citation in answer
- `source_missing` — used web tool but no URL in answer
- `pass` — all checks passed

**File:** `app/run_cli.py`
**Change:** After `guardrail_checks` is called, also call `classify_failure` and add `"failure_mode"` key to the `log_record` dict that gets written to JSONL.

**File:** `app/eval_runner.py`
**Change:** Same — add `"failure_mode"` to each question result alongside `guardrail_checks`.

---

## Point 2: Planner Reasoning Capture

**File:** `app/planner_agent.py`
**Change:** After the LLM chain produces `raw_plan` (the raw string output before JSON parsing), store it in the state so it flows into the trace.

Approach: Add a `"planner_reasoning"` key to the returned state dict from the supervisor node. This captures what the LLM said verbatim before it was parsed/repaired into a structured plan.

**File:** `app/state.py`
**Change:** Add `planner_reasoning: str` field to `AgentState` TypedDict.

**File:** `app/run_cli.py`
**Change:** Include `state.get("planner_reasoning")` in the `log_record` dict written to JSONL.

**File:** `app/eval_runner.py`
**Change:** Same — include planner_reasoning in question results.

---

## Point 3: Inline Answer Groundedness Check

**File:** `app/guardrails.py`
**Change:** Add `check_groundedness(answer_text: str, tool_outputs: list[str], llm) -> dict` function.

This function:
1. Extracts all tool output text from researcher messages
2. Sends a focused prompt to qwen3.5:4b asking:
   "Given these sources, is every claim in the answer supported? Return grounded/ungrounded/partial and a 1-sentence reason."
3. Parses the response into `{"groundedness": "grounded"|"ungrounded"|"partial", "reason": "..."}`

Model config:
- Model: qwen3.5:4b
- num_ctx: 1024 (answer + tool outputs should be <750 words)
- temperature: 0.1 (deterministic judgment)
- keep_alive: "20m" (reuse VRAM-loaded instance, same as other models)

**File:** `app/answer_agent.py`
**Change:** After the answer is assembled but before returning, call `check_groundedness` inline. Add the result to the returned state as `"groundedness_check"`.

**File:** `app/state.py`
**Change:** Add `groundedness_check: dict` field to `AgentState`.

**File:** `app/run_cli.py`
**Change:** Include `state.get("groundedness_check")` in the JSONL log_record.

**File:** `app/config.py`
**Change:** Add `GROUNDEDNESS_MODEL`, `GROUNDEDNESS_NUM_CTX = 1024`, and `GROUNDEDNESS_TEMPERATURE = 0.1` constants.

---

## Testing Plan

1. **Unit test for classify_failure:** Feed it known check dicts, verify correct category.
2. **Unit test for check_groundedness prompt parsing:** Verify it correctly parses grounded/ungrounded/partial from model output.
3. **Integration test:** Run 3 eval questions (smoke mode) end-to-end with Ollama, verify:
   - JSONL traces contain `failure_mode` field
   - JSONL traces contain `planner_reasoning` field (non-empty for LLM-planned turns)
   - JSONL traces contain `groundedness_check` field with valid structure
4. **Verify no latency regression beyond expected judge call time.**
