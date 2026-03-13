"""Tests for failure mode classification, planner reasoning capture, and groundedness check."""

import json
import sys
import os
import unittest
from unittest.mock import MagicMock

# Ensure app/ is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from guardrails import classify_failure, check_groundedness, extract_tool_outputs


class TestClassifyFailure(unittest.TestCase):
    """Point 1: failure mode classification maps guardrail results to categories."""

    def test_pass_all_checks(self):
        checks = {
            "plan_parseable": True,
            "tool_called_before_response": True,
            "tool_choice_correct": True,
            "pdf_citations_present": True,
            "web_sources_present": True,
        }
        self.assertEqual(classify_failure(checks), "pass")

    def test_plan_generation_failure(self):
        checks = {
            "plan_parseable": False,
            "tool_called_before_response": True,
            "tool_choice_correct": True,
            "pdf_citations_present": True,
            "web_sources_present": True,
        }
        self.assertEqual(classify_failure(checks), "plan_generation_failure")

    def test_routing_error(self):
        checks = {
            "plan_parseable": True,
            "tool_called_before_response": True,
            "tool_choice_correct": False,
            "pdf_citations_present": True,
            "web_sources_present": True,
        }
        self.assertEqual(classify_failure(checks), "routing_error")

    def test_tool_skip(self):
        checks = {
            "plan_parseable": True,
            "tool_called_before_response": False,
            "tool_choice_correct": True,
            "pdf_citations_present": True,
            "web_sources_present": True,
        }
        self.assertEqual(classify_failure(checks), "tool_skip")

    def test_citation_missing(self):
        checks = {
            "plan_parseable": True,
            "tool_called_before_response": True,
            "tool_choice_correct": True,
            "pdf_citations_present": False,
            "web_sources_present": True,
        }
        self.assertEqual(classify_failure(checks), "citation_missing")

    def test_source_missing(self):
        checks = {
            "plan_parseable": True,
            "tool_called_before_response": True,
            "tool_choice_correct": True,
            "pdf_citations_present": True,
            "web_sources_present": False,
        }
        self.assertEqual(classify_failure(checks), "source_missing")

    def test_empty_checks(self):
        self.assertEqual(classify_failure({}), "no_checks")
        self.assertEqual(classify_failure(None), "no_checks")

    def test_priority_order(self):
        """When multiple things fail, the highest priority category wins."""
        checks = {
            "plan_parseable": False,
            "tool_called_before_response": False,
            "tool_choice_correct": False,
            "pdf_citations_present": False,
            "web_sources_present": False,
        }
        # plan_generation_failure has highest priority
        self.assertEqual(classify_failure(checks), "plan_generation_failure")


class TestExtractToolOutputs(unittest.TestCase):
    """Verify tool output extraction from message lists."""

    def test_extracts_tool_messages(self):
        msg1 = MagicMock()
        msg1.type = "tool"
        msg1.content = "PDF chunk about deep work [p.42]"

        msg2 = MagicMock()
        msg2.type = "ai"
        msg2.content = "Final answer text"

        msg3 = MagicMock()
        msg3.type = "tool"
        msg3.content = "Web search result from tavily"

        result = extract_tool_outputs([msg1, msg2, msg3])
        self.assertIn("PDF chunk about deep work", result)
        self.assertIn("Web search result from tavily", result)
        self.assertNotIn("Final answer text", result)

    def test_empty_messages(self):
        result = extract_tool_outputs([])
        self.assertEqual(result, "")

    def test_no_tool_messages(self):
        msg = MagicMock()
        msg.type = "ai"
        msg.content = "Just an AI message"
        result = extract_tool_outputs([msg])
        self.assertEqual(result, "")


class TestCheckGroundedness(unittest.TestCase):
    """Point 3: groundedness check with mocked LLM."""

    def test_grounded_verdict(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"verdict": "grounded", "reason": "All claims supported."}'
        mock_llm.invoke.return_value = mock_response

        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "Deep work requires focused attention for extended periods."

        result = check_groundedness(
            "Deep work needs focused attention.",
            [tool_msg],
            mock_llm,
        )
        self.assertEqual(result["verdict"], "grounded")
        self.assertIn("supported", result["reason"].lower())

    def test_ungrounded_verdict(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"verdict": "ungrounded", "reason": "Claim about aliens not in source."}'
        mock_llm.invoke.return_value = mock_response

        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "Deep work is about concentration."

        result = check_groundedness(
            "Aliens invented deep work.",
            [tool_msg],
            mock_llm,
        )
        self.assertEqual(result["verdict"], "ungrounded")

    def test_no_tool_output_skips(self):
        mock_llm = MagicMock()
        result = check_groundedness("Some answer.", [], mock_llm)
        self.assertEqual(result["verdict"], "no_tool_output")
        mock_llm.invoke.assert_not_called()

    def test_empty_answer_skips(self):
        mock_llm = MagicMock()
        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "Some evidence."
        result = check_groundedness("", [tool_msg], mock_llm)
        self.assertEqual(result["verdict"], "no_answer")
        mock_llm.invoke.assert_not_called()

    def test_strips_sources_before_judging(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"verdict": "grounded", "reason": "OK"}'
        mock_llm.invoke.return_value = mock_response

        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "Evidence text here."

        check_groundedness(
            "Answer text.\n\nSources:\n- [p.5]\n- http://example.com",
            [tool_msg],
            mock_llm,
        )
        # Verify the prompt sent to LLM does not contain the Sources block.
        call_args = mock_llm.invoke.call_args[0][0]
        self.assertNotIn("http://example.com", call_args)

    def test_handles_think_tags_in_response(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '<think>Let me analyze...</think>{"verdict": "partial", "reason": "One claim missing."}'
        mock_llm.invoke.return_value = mock_response

        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "Some evidence."

        result = check_groundedness("Some answer.", [tool_msg], mock_llm)
        self.assertEqual(result["verdict"], "partial")

    def test_handles_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("Connection refused")

        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "Some evidence."

        result = check_groundedness("Some answer.", [tool_msg], mock_llm)
        self.assertEqual(result["verdict"], "error")
        self.assertIn("Connection refused", result["reason"])

    def test_handles_malformed_json_response(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "I think it is grounded but I cannot format JSON"
        mock_llm.invoke.return_value = mock_response

        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "Some evidence."

        result = check_groundedness("Some answer.", [tool_msg], mock_llm)
        self.assertEqual(result["verdict"], "parse_error")


if __name__ == "__main__":
    unittest.main()
