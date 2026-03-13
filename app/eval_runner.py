"""Run a fixed evaluation suite across model presets and save results."""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List

from langchain_core.messages import HumanMessage

from config import ANSWERER_MODEL, BASE_DIR, PLANNER_MODEL, RESEARCHER_MODEL
from graph import build_app
from guardrails import check_groundedness, classify_failure, guardrail_checks
from metrics import duration_ms, now_ms


EVAL_QUESTIONS_PATH = BASE_DIR / "eval" / "questions.jsonl"
EVAL_RESULTS_DIR = BASE_DIR / "eval" / "results"


def _load_questions() -> List[Dict[str, str]]:
    questions: List[Dict[str, str]] = []
    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as file:
        for raw in file:
            line = raw.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def _presets_full() -> List[Dict[str, str]]:
    """Full one-role-at-a-time sweep around your current defaults."""
    return [
        {
            "name": "baseline_all_qwen3_4b",
            "planner": "qwen3:4b",
            "researcher": "qwen3:4b",
            "answerer": "qwen3:4b",
        },
        {
            "name": "planner_qwen3_1_7b",
            "planner": "qwen3:1.7b",
            "researcher": RESEARCHER_MODEL,
            "answerer": ANSWERER_MODEL,
        },
        {
            "name": "researcher_qwen3_1_7b",
            "planner": PLANNER_MODEL,
            "researcher": "qwen3:1.7b",
            "answerer": ANSWERER_MODEL,
        },
        {
            "name": "answerer_qwen3_1_7b",
            "planner": PLANNER_MODEL,
            "researcher": RESEARCHER_MODEL,
            "answerer": "qwen3:1.7b",
        },
    ]


def _presets_small() -> List[Dict[str, str]]:
    """Accuracy-oriented default run for a single stable preset."""
    return [
        {
            "name": "baseline_all_qwen3_4b",
            "planner": "qwen3:4b",
            "researcher": "qwen3:4b",
            "answerer": "qwen3:4b",
        }
    ]


def _resolve_question_limit(mode: str) -> int | None:
    env_limit = os.getenv("EVAL_QUESTION_LIMIT", "").strip()
    if env_limit.isdigit():
        return int(env_limit)
    if mode == "smoke":
        return 3
    return None


def run_eval() -> None:
    """
    Modes:
    - smoke: single preset, 3 questions (fast health-check)
    - small: single preset, all questions (accuracy snapshot)
    - full: full sweep, all questions
    """
    questions = _load_questions()
    mode = os.getenv("EVAL_MODE", "small").lower().strip()
    presets = _presets_full() if mode == "full" else _presets_small()
    question_limit = _resolve_question_limit(mode)
    if question_limit:
        questions = questions[:question_limit]

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = EVAL_RESULTS_DIR / f"eval_{timestamp}.json"

    results: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "question_count": len(questions),
        "presets": [],
    }

    for preset in presets:
        app, groundedness_llm = build_app(
            planner_model=preset["planner"],
            researcher_model=preset["researcher"],
            answerer_model=preset["answerer"],
        )
        preset_result: Dict[str, object] = {
            "name": preset["name"],
            "model_map": {
                "planner": preset["planner"],
                "researcher": preset["researcher"],
                "answerer": preset["answerer"],
            },
            "questions": [],
        }
        cast_questions = preset_result["questions"]
        if not isinstance(cast_questions, list):
            cast_questions = []
            preset_result["questions"] = cast_questions
        results["presets"].append(preset_result)

        for item in questions:
            question_text = item["question"]
            start = now_ms()
            try:
                state = app.invoke({"messages": [HumanMessage(content=question_text)]})
                end = now_ms()
                messages = state.get("messages", [])
                plan = state.get("plan")
                checks = guardrail_checks(question_text, plan, messages)
                # Post-graph groundedness check.
                final_text = ""
                for msg in reversed(messages):
                    if getattr(msg, "type", "") == "ai":
                        final_text = str(getattr(msg, "content", ""))
                        break
                groundedness_result = None
                if groundedness_llm is not None and final_text:
                    groundedness_result = check_groundedness(
                        final_text, messages, groundedness_llm
                    )
                cast_questions.append(
                    {
                        "id": item.get("id"),
                        "question": question_text,
                        "latency_ms": duration_ms(start, end),
                        "guardrail_checks": checks,
                        "failure_mode": classify_failure(checks),
                        "planner_reasoning": state.get("planner_reasoning", ""),
                        "groundedness_check": groundedness_result,
                        "error": None,
                    }
                )
            except Exception as exc:
                end = now_ms()
                cast_questions.append(
                    {
                        "id": item.get("id"),
                        "question": question_text,
                        "latency_ms": duration_ms(start, end),
                        "guardrail_checks": None,
                        "error": str(exc),
                    }
                )

            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(results, file, ensure_ascii=False, indent=2)

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_eval()

