# Brittleness Review Log

## Review — 2026-03-15 (Initial)

**Scope**: Full project — all Python files in `app/`, `tests/`, `mcp/`
**Files scanned**: 27 Python modules + 11 test files
**Issues found**: 6
**Status**: All fixed

### Findings

#### [God Function] — `supervisor()` in planner_agent.py
- **File**: `app/planner_agent.py:466-730`
- **Severity**: High
- **Problem**: 264-line function handling 7 responsibilities (intent classification, email state, retry/reset, draft detection, plan generation, routing, debug messages). ~8 early return paths made control flow untraceable.
- **Why you'd miss it**: Grew incrementally — each feature added 20-30 lines, no single commit felt too large.
- **Suggested fix**: Split into `_resolve_intent_and_state()`, `_handle_active_draft()`, `_generate_plan()`, `_route_next_step()`.
- **Status**: Fixed (2026-03-15) — supervisor is now ~40-line orchestrator calling 4 helpers

#### [I/O Amplification] — chat_sessions.py load/save per operation
- **File**: `app/chat_sessions.py`
- **Severity**: High
- **Problem**: Every public function did full `_load_sessions()` + `_save_sessions()`. Single turn = ~8 reads + ~4 writes of entire JSON. No file locking for concurrent access.
- **Why you'd miss it**: Single-user local testing — file is small, reads are fast, no concurrent requests during dev.
- **Suggested fix**: Session-scoped cache (load once, save once per turn) + threading lock.
- **Status**: Fixed (2026-03-15) — `SessionCache` context manager + `threading.Lock` added

#### [Duplicated Vocabularies] — Intent markers scattered across files
- **File**: `app/mailer_agent.py:32-46`, `app/turn_controller.py:64-79`, `app/graph_memory.py:82-86`, `app/planner_agent.py:70-100`
- **Severity**: Medium
- **Problem**: Same concept ("send previous answer") detected by different keyword lists in 4 files. Lists overlapped but were NOT identical — system could disagree with itself.
- **Why you'd miss it**: Each file authored in isolation; keywords added empirically to fix individual failures without vocabulary audit.
- **Suggested fix**: Create `vocabulary.py` as single source of truth.
- **Status**: Fixed (2026-03-15) — `vocabulary.py` created with 5 unified constant sets, 6 consumer files updated

#### [Untyped State Bags] — Dict[str, Any] for task_state, draft, flags
- **File**: `app/state.py:31,33,37`
- **Severity**: Medium
- **Problem**: Schema lived in `_default_task_state()` but consumers in 4+ files did `.get()` chains. Key typos silently create new keys. No compile-time enforcement.
- **Why you'd miss it**: Python duck typing makes it "just work." `normalize_task_state` safety net catches most issues at runtime.
- **Suggested fix**: Replace with TypedDicts for `TaskState`, `EmailFrame`, `DraftState`, `FlagState`.
- **Status**: Fixed (2026-03-15) — 6 TypedDicts added to `state.py`, type annotations updated across 5 files

#### [Untested Defensive Logic] — _repair_plan() has 15+ branches, 0 tests
- **File**: `app/planner_agent.py:197-357`
- **Severity**: Medium
- **Problem**: 160-line function with 15+ conditional branches. Each fixes a real failure, but interactions are non-obvious. Invariants (researcher before answerer, exactly one terminal step) never asserted.
- **Why you'd miss it**: Built reactively — every branch handles an observed failure. Fallback catches anything repair doesn't.
- **Suggested fix**: Add unit tests asserting post-repair invariants across diverse inputs.
- **Status**: Fixed (2026-03-15) — 51 tests in `tests/test_repair_plan.py` covering 10 invariants

#### [Silent Degradation] — Memory backend falls back without logging
- **File**: `app/graph_memory.py:656-668`
- **Severity**: Medium
- **Problem**: Graphiti → local JSON fallback is silent. If both fail, memory is silently dropped. No log, no metric, no notification. Data can bifurcate.
- **Why you'd miss it**: Designed as robustness feature (never crashes). But robustness without observability is a trap.
- **Suggested fix**: Add logging at every fallback transition.
- **Status**: Fixed (2026-03-15) — `logging.getLogger(__name__)` with INFO/WARNING/ERROR at all transition points

---

## Fix Summary — 2026-03-15

**Baseline**: 47 tests | **Final**: 98 tests | **Regressions**: 0

| # | Issue | Fix | Tests |
|---|-------|-----|-------|
| 1 | God function | Split into 4 helpers + orchestrator | 98/98 |
| 2 | I/O amplification | SessionCache + threading.Lock | 98/98 |
| 3 | Duplicated vocabularies | vocabulary.py (5 constant sets) | 98/98 |
| 4 | Untyped state bags | 6 TypedDicts in state.py | 98/98 |
| 5 | Untested repair logic | 51 new tests in test_repair_plan.py | 98/98 |
| 6 | Silent degradation | logging at all fallback paths | 98/98 |

### Files Created
- `app/vocabulary.py`
- `tests/test_repair_plan.py`

### Files Modified
- `app/planner_agent.py`, `app/chat_sessions.py`, `app/state.py`, `app/graph_memory.py`
- `app/mailer_agent.py`, `app/turn_controller.py`, `app/intent_utils.py`, `app/guardrails.py`
