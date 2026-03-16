"""Comprehensive unit tests for _repair_plan and _validate_plan."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from planner_agent import (  # noqa: E402
    _repair_plan,
    _validate_plan,
    _fallback_plan_for_query,
    MIN_STEPS,
    MAX_STEPS,
    MIN_STEPS_EMAIL,
    ALLOWED_ACTIONS,
    ALLOWED_TOOLS,
)


# ---------------------------------------------------------------------------
# Query fixtures – chosen to trigger specific classify_query_source results
# ---------------------------------------------------------------------------
QA_PDF_QUERY = "What does Cal Newport say about deep work?"
QA_WEB_QUERY = "What is the latest score in the T20 match today?"
QA_BOTH_QUERY = "Compare Cal Newport deep work ideas with current productivity news"
QA_UNKNOWN_QUERY = "Explain quantum entanglement"
EMAIL_SIMPLE_QUERY = "Send an email to John about the meeting"
EMAIL_RESEARCH_QUERY = "Send an email to John with the latest news today"
CONVERSATIONAL_QUERY = "hello"


# ---------------------------------------------------------------------------
# Helper: build a well-formed QA plan (pdf source)
# ---------------------------------------------------------------------------
def _valid_qa_plan_pdf():
    return [
        {"id": 0, "action": "researcher", "description": "Retrieve context.", "tools": ["retrieve_context"]},
        {"id": 1, "action": "answerer", "description": "Synthesize answer."},
    ]


def _valid_qa_plan_web():
    return [
        {"id": 0, "action": "researcher", "description": "Search the web.", "tools": ["tavily_search"]},
        {"id": 1, "action": "answerer", "description": "Synthesize answer."},
    ]


def _valid_email_plan():
    return [
        {"id": 0, "action": "mailer", "description": "Draft and send the email."},
    ]


def _valid_email_plan_with_research():
    return [
        {"id": 0, "action": "researcher", "description": "Search the web.", "tools": ["tavily_search"]},
        {"id": 1, "action": "mailer", "description": "Draft and send the email."},
    ]


# ===================================================================
# Shared post-repair invariant assertions
# ===================================================================
def assert_repair_invariants(tc: unittest.TestCase, plan, query, email_hint=False):
    """Assert every invariant that must hold after _repair_plan."""
    # 1. Return type: list of dicts
    tc.assertIsInstance(plan, list)
    for step in plan:
        tc.assertIsInstance(step, dict)

    # 2. Non-empty
    tc.assertGreater(len(plan), 0)

    # 3. Sequential ids
    for i, step in enumerate(plan):
        tc.assertEqual(step["id"], i, f"Step {i} has wrong id {step.get('id')}")

    # 4. Valid actions
    for step in plan:
        tc.assertIn(step["action"], ALLOWED_ACTIONS)

    # 7. Step count bounds
    from planner_agent import _detect_email_intent  # local import to avoid circular
    from guardrails import classify_query_source

    email_intent = _detect_email_intent(query, email_hint)
    source = classify_query_source(query)
    conversational = source == "conversational"

    if conversational:
        min_steps = 1
    elif email_intent:
        min_steps = MIN_STEPS_EMAIL
    else:
        min_steps = MIN_STEPS
    tc.assertGreaterEqual(len(plan), min_steps)
    tc.assertLessEqual(len(plan), MAX_STEPS)

    # 8. Tool validity on researcher steps
    for step in plan:
        if step["action"] == "researcher":
            tc.assertIn("tools", step)
            tc.assertIsInstance(step["tools"], list)
            tc.assertGreater(len(step["tools"]), 0)
            for tool in step["tools"]:
                tc.assertIn(tool, ALLOWED_TOOLS)

    # 9. No tools on non-researcher steps
    for step in plan:
        if step["action"] in ("answerer", "mailer"):
            tools = step.get("tools")
            tc.assertTrue(tools is None or tools == [],
                          f"Non-researcher step has tools: {tools}")

    # 5/6. Ordering constraints
    if conversational:
        tc.assertEqual(plan[0]["action"], "answerer")
    elif email_intent:
        tc.assertEqual(plan[-1]["action"], "mailer")
    else:
        tc.assertEqual(plan[0]["action"], "researcher")
        tc.assertEqual(plan[-1]["action"], "answerer")

    # Mailer not present for non-email
    if not email_intent:
        for step in plan:
            tc.assertNotEqual(step["action"], "mailer",
                              "Mailer should not appear in non-email plan")


# ===================================================================
# Tests for _repair_plan
# ===================================================================
class TestRepairPlanValidInput(unittest.TestCase):
    """Well-formed input should pass through (with possible re-indexing)."""

    def test_repair_valid_qa_plan_unchanged(self):
        plan = _valid_qa_plan_pdf()
        result = _repair_plan(plan, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["action"], "researcher")
        self.assertEqual(result[1]["action"], "answerer")
        self.assertIn("retrieve_context", result[0]["tools"])

    def test_repair_valid_web_plan_unchanged(self):
        plan = _valid_qa_plan_web()
        result = _repair_plan(plan, QA_WEB_QUERY)
        assert_repair_invariants(self, result, QA_WEB_QUERY)
        self.assertEqual(result[0]["action"], "researcher")
        self.assertIn("tavily_search", result[0]["tools"])

    def test_repair_valid_email_plan_unchanged(self):
        plan = _valid_email_plan()
        result = _repair_plan(plan, EMAIL_SIMPLE_QUERY)
        assert_repair_invariants(self, result, EMAIL_SIMPLE_QUERY)
        self.assertEqual(result[-1]["action"], "mailer")

    def test_repair_valid_email_plan_with_research_unchanged(self):
        plan = _valid_email_plan_with_research()
        result = _repair_plan(plan, EMAIL_RESEARCH_QUERY)
        assert_repair_invariants(self, result, EMAIL_RESEARCH_QUERY)
        self.assertEqual(result[-1]["action"], "mailer")
        self.assertTrue(any(s["action"] == "researcher" for s in result))


class TestRepairPlanGarbageInput(unittest.TestCase):
    """Empty/garbage input must produce a valid fallback."""

    def test_repair_empty_list_returns_fallback(self):
        result = _repair_plan([], QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        self.assertGreaterEqual(len(result), MIN_STEPS)

    def test_repair_non_list_returns_fallback(self):
        for garbage in ("a string", 42, None, {"key": "value"}, True):
            result = _repair_plan(garbage, QA_PDF_QUERY)
            assert_repair_invariants(self, result, QA_PDF_QUERY)

    def test_repair_list_of_strings_returns_fallback(self):
        result = _repair_plan(["step1", "step2"], QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        self.assertGreaterEqual(len(result), MIN_STEPS)


class TestRepairPlanActionFiltering(unittest.TestCase):
    """Unknown or disallowed actions must be filtered out."""

    def test_repair_removes_unknown_actions(self):
        plan = [
            {"action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
            {"action": "bogus_action", "description": "Bad step."},
            {"action": "answerer", "description": "Answer."},
        ]
        result = _repair_plan(plan, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        actions = [s["action"] for s in result]
        self.assertNotIn("bogus_action", actions)

    def test_repair_removes_mailer_from_non_email(self):
        plan = [
            {"action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
            {"action": "mailer", "description": "Email."},
            {"action": "answerer", "description": "Answer."},
        ]
        result = _repair_plan(plan, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        actions = [s["action"] for s in result]
        self.assertNotIn("mailer", actions)


class TestRepairPlanStepInjection(unittest.TestCase):
    """Missing terminal steps should be injected."""

    def test_repair_injects_researcher_when_missing_for_qa(self):
        plan = [
            {"action": "answerer", "description": "Answer."},
        ]
        result = _repair_plan(plan, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        self.assertEqual(result[0]["action"], "researcher")
        self.assertEqual(result[-1]["action"], "answerer")

    def test_repair_appends_answerer_when_missing_for_qa(self):
        plan = [
            {"action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
            {"action": "researcher", "description": "More search.", "tools": ["retrieve_context"]},
        ]
        result = _repair_plan(plan, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        self.assertEqual(result[-1]["action"], "answerer")

    def test_repair_appends_mailer_when_missing_for_email(self):
        plan = [
            {"action": "researcher", "description": "Search.", "tools": ["tavily_search"]},
        ]
        result = _repair_plan(plan, EMAIL_RESEARCH_QUERY)
        assert_repair_invariants(self, result, EMAIL_RESEARCH_QUERY)
        self.assertEqual(result[-1]["action"], "mailer")


class TestRepairPlanStepCounts(unittest.TestCase):
    """Enforce MIN_STEPS and MAX_STEPS bounds."""

    def test_repair_pads_to_min_steps(self):
        # A single researcher for a QA query requires padding to MIN_STEPS=2
        plan = [
            {"action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
        ]
        result = _repair_plan(plan, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        self.assertGreaterEqual(len(result), MIN_STEPS)

    def test_repair_trims_to_max_steps(self):
        # Build a plan with more than MAX_STEPS steps
        oversized = []
        for i in range(MAX_STEPS + 4):
            oversized.append({"action": "researcher", "description": f"Step {i}.", "tools": ["retrieve_context"]})
        oversized.append({"action": "answerer", "description": "Final answer."})
        result = _repair_plan(oversized, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        self.assertLessEqual(len(result), MAX_STEPS)

    def test_repair_trim_preserves_final_answerer(self):
        oversized = []
        for i in range(MAX_STEPS + 3):
            oversized.append({"action": "researcher", "description": f"Step {i}.", "tools": ["retrieve_context"]})
        result = _repair_plan(oversized, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        self.assertEqual(result[-1]["action"], "answerer")

    def test_repair_trim_preserves_final_mailer(self):
        oversized = []
        for i in range(MAX_STEPS + 3):
            oversized.append({"action": "researcher", "description": f"Step {i}.", "tools": ["tavily_search"]})
        oversized.append({"action": "mailer", "description": "Send email."})
        result = _repair_plan(oversized, EMAIL_RESEARCH_QUERY)
        assert_repair_invariants(self, result, EMAIL_RESEARCH_QUERY)
        self.assertEqual(result[-1]["action"], "mailer")


class TestRepairPlanToolFiltering(unittest.TestCase):
    """Invalid tools must be stripped; missing required tools injected."""

    def test_repair_filters_invalid_tools(self):
        plan = [
            {"action": "researcher", "description": "Search.",
             "tools": ["retrieve_context", "fake_tool", "another_bad"]},
            {"action": "answerer", "description": "Answer."},
        ]
        result = _repair_plan(plan, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        for step in result:
            if step["action"] == "researcher":
                for tool in step["tools"]:
                    self.assertIn(tool, ALLOWED_TOOLS)

    def test_repair_injects_missing_required_tools(self):
        # PDF query requires retrieve_context; plan has none listed
        plan = [
            {"action": "researcher", "description": "Search.", "tools": []},
            {"action": "answerer", "description": "Answer."},
        ]
        result = _repair_plan(plan, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        all_tools = set()
        for step in result:
            if step["action"] == "researcher":
                all_tools.update(step.get("tools", []))
        self.assertIn("retrieve_context", all_tools)

    def test_repair_injects_retrieve_context_for_pdf(self):
        plan = [
            {"action": "researcher", "description": "Search.", "tools": ["tavily_search"]},
            {"action": "answerer", "description": "Answer."},
        ]
        result = _repair_plan(plan, QA_PDF_QUERY)
        assert_repair_invariants(self, result, QA_PDF_QUERY)
        all_tools = set()
        for step in result:
            if step["action"] == "researcher":
                all_tools.update(step.get("tools", []))
        self.assertIn("retrieve_context", all_tools)

    def test_repair_injects_tavily_search_for_web(self):
        plan = [
            {"action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
            {"action": "answerer", "description": "Answer."},
        ]
        result = _repair_plan(plan, QA_WEB_QUERY)
        assert_repair_invariants(self, result, QA_WEB_QUERY)
        all_tools = set()
        for step in result:
            if step["action"] == "researcher":
                all_tools.update(step.get("tools", []))
        self.assertIn("tavily_search", all_tools)


class TestRepairPlanConversational(unittest.TestCase):
    """Conversational queries produce a single answerer step."""

    def test_repair_conversational_returns_single_answerer(self):
        plan = [
            {"action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
            {"action": "answerer", "description": "Answer."},
        ]
        result = _repair_plan(plan, CONVERSATIONAL_QUERY)
        assert_repair_invariants(self, result, CONVERSATIONAL_QUERY)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["action"], "answerer")

    def test_repair_conversational_with_empty_list(self):
        result = _repair_plan([], CONVERSATIONAL_QUERY)
        # Empty list is still a list, so conversational path triggers
        assert_repair_invariants(self, result, CONVERSATIONAL_QUERY)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["action"], "answerer")

    def test_repair_conversational_with_garbage(self):
        result = _repair_plan("not a list", CONVERSATIONAL_QUERY)
        assert_repair_invariants(self, result, CONVERSATIONAL_QUERY)
        self.assertEqual(len(result), 1)


# ===================================================================
# Tests for _validate_plan
# ===================================================================
class TestValidatePlan(unittest.TestCase):
    """Tests for the _validate_plan function."""

    def test_validate_rejects_empty_plan(self):
        self.assertFalse(_validate_plan([], QA_PDF_QUERY))

    def test_validate_rejects_non_list(self):
        self.assertFalse(_validate_plan("not a list", QA_PDF_QUERY))
        self.assertFalse(_validate_plan(None, QA_PDF_QUERY))
        self.assertFalse(_validate_plan(42, QA_PDF_QUERY))

    def test_validate_rejects_wrong_first_step(self):
        plan = [
            {"id": 0, "action": "answerer", "description": "Wrong first."},
            {"id": 1, "action": "answerer", "description": "Answer."},
        ]
        self.assertFalse(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_rejects_wrong_last_step_qa(self):
        plan = [
            {"id": 0, "action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
            {"id": 1, "action": "researcher", "description": "More.", "tools": ["retrieve_context"]},
        ]
        self.assertFalse(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_rejects_tools_on_answerer(self):
        plan = [
            {"id": 0, "action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
            {"id": 1, "action": "answerer", "description": "Answer.", "tools": ["retrieve_context"]},
        ]
        self.assertFalse(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_rejects_tools_on_mailer(self):
        plan = [
            {"id": 0, "action": "mailer", "description": "Email.", "tools": ["tavily_search"]},
        ]
        self.assertFalse(_validate_plan(plan, EMAIL_SIMPLE_QUERY))

    def test_validate_rejects_mailer_in_non_email(self):
        plan = [
            {"id": 0, "action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
            {"id": 1, "action": "mailer", "description": "Email."},
        ]
        self.assertFalse(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_rejects_unknown_action(self):
        plan = [
            {"id": 0, "action": "researcher", "description": "Search.", "tools": ["retrieve_context"]},
            {"id": 1, "action": "bogus", "description": "Bad."},
        ]
        self.assertFalse(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_rejects_invalid_tools(self):
        plan = [
            {"id": 0, "action": "researcher", "description": "Search.", "tools": ["bad_tool"]},
            {"id": 1, "action": "answerer", "description": "Answer."},
        ]
        self.assertFalse(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_rejects_empty_tools_on_researcher(self):
        plan = [
            {"id": 0, "action": "researcher", "description": "Search.", "tools": []},
            {"id": 1, "action": "answerer", "description": "Answer."},
        ]
        self.assertFalse(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_rejects_missing_tools_on_researcher(self):
        plan = [
            {"id": 0, "action": "researcher", "description": "Search."},
            {"id": 1, "action": "answerer", "description": "Answer."},
        ]
        self.assertFalse(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_rejects_oversized_plan(self):
        plan = []
        for i in range(MAX_STEPS + 1):
            plan.append({"id": i, "action": "researcher", "description": f"Step {i}.", "tools": ["retrieve_context"]})
        plan.append({"id": MAX_STEPS + 1, "action": "answerer", "description": "Answer."})
        self.assertFalse(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_accepts_valid_qa_plan(self):
        plan = _valid_qa_plan_pdf()
        self.assertTrue(_validate_plan(plan, QA_PDF_QUERY))

    def test_validate_accepts_valid_qa_web_plan(self):
        plan = _valid_qa_plan_web()
        self.assertTrue(_validate_plan(plan, QA_WEB_QUERY))

    def test_validate_accepts_valid_email_plan(self):
        plan = _valid_email_plan()
        self.assertTrue(_validate_plan(plan, EMAIL_SIMPLE_QUERY))

    def test_validate_accepts_valid_email_plan_with_research(self):
        plan = _valid_email_plan_with_research()
        self.assertTrue(_validate_plan(plan, EMAIL_RESEARCH_QUERY))

    def test_validate_accepts_conversational_plan(self):
        plan = [{"id": 0, "action": "answerer", "description": "Respond directly."}]
        self.assertTrue(_validate_plan(plan, CONVERSATIONAL_QUERY))

    def test_validate_rejects_wrong_email_last_step(self):
        plan = [
            {"id": 0, "action": "researcher", "description": "Search.", "tools": ["tavily_search"]},
            {"id": 1, "action": "answerer", "description": "Answer."},
        ]
        self.assertFalse(_validate_plan(plan, EMAIL_RESEARCH_QUERY))


# ===================================================================
# Tests for _repair_plan -> _validate_plan roundtrip
# ===================================================================
class TestRepairProducesValidPlan(unittest.TestCase):
    """Every repaired plan must pass _validate_plan."""

    def _roundtrip(self, steps, query, email_hint=False):
        result = _repair_plan(steps, query, email_hint)
        assert_repair_invariants(self, result, query, email_hint)
        self.assertTrue(_validate_plan(result, query, email_hint),
                        f"Repaired plan failed validation: {result}")

    def test_roundtrip_valid_pdf_plan(self):
        self._roundtrip(_valid_qa_plan_pdf(), QA_PDF_QUERY)

    def test_roundtrip_valid_web_plan(self):
        self._roundtrip(_valid_qa_plan_web(), QA_WEB_QUERY)

    def test_roundtrip_empty_list_pdf(self):
        self._roundtrip([], QA_PDF_QUERY)

    def test_roundtrip_garbage_input(self):
        self._roundtrip("garbage", QA_PDF_QUERY)

    def test_roundtrip_email_simple(self):
        self._roundtrip(_valid_email_plan(), EMAIL_SIMPLE_QUERY)

    def test_roundtrip_email_with_research(self):
        self._roundtrip(_valid_email_plan_with_research(), EMAIL_RESEARCH_QUERY)

    def test_roundtrip_conversational(self):
        self._roundtrip([], CONVERSATIONAL_QUERY)

    def test_roundtrip_oversized(self):
        oversized = [{"action": "researcher", "description": f"Step {i}.", "tools": ["retrieve_context"]}
                     for i in range(MAX_STEPS + 5)]
        self._roundtrip(oversized, QA_PDF_QUERY)

    def test_roundtrip_unknown_source(self):
        self._roundtrip([], QA_UNKNOWN_QUERY)

    def test_roundtrip_both_source(self):
        self._roundtrip([], QA_BOTH_QUERY)


if __name__ == "__main__":
    unittest.main()
