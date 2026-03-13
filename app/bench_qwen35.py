"""One-command benchmark for qwen3.5:2b vs qwen3.5:4b on this project."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


ROOT = Path(__file__).resolve().parents[1]
ROLE_BENCH = ROOT / "app" / "role_benchmark.py"
RESULTS_DIR = ROOT / "eval" / "results"


def _run_for_model(model: str) -> Path:
    env = os.environ.copy()
    env.update(
        {
            "BENCH_MODELS": model,
            "BENCH_BASE_MODEL": model,
            "BENCH_ROLES": env.get("BENCH_ROLES", "planner,researcher,answerer,mailer"),
            "BENCH_QUESTION_LIMIT": env.get("BENCH_QUESTION_LIMIT", "2"),
            "NUM_CTX": env.get("NUM_CTX", "1536"),
            "PLANNER_NUM_CTX": env.get("PLANNER_NUM_CTX", "1024"),
            "RESEARCHER_NUM_CTX": env.get("RESEARCHER_NUM_CTX", "1536"),
            "ANSWERER_NUM_CTX": env.get("ANSWERER_NUM_CTX", "1536"),
            "MAILER_NUM_CTX": env.get("MAILER_NUM_CTX", "1536"),
            "NUM_PREDICT": env.get("NUM_PREDICT", "256"),
            "KEEP_ALIVE": env.get("KEEP_ALIVE", "0"),
            "DISABLE_STREAMING": env.get("DISABLE_STREAMING", "1"),
        }
    )
    before = {p.name for p in RESULTS_DIR.glob("role_benchmark_*.json")}
    timeout_sec = int(env.get("BENCH_MODEL_TIMEOUT_SEC", "900"))
    try:
        proc = subprocess.run(
            [sys.executable, str(ROLE_BENCH)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Benchmark timed out for {model} after {timeout_sec}s") from exc
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"Benchmark failed for {model} (exit code {proc.returncode})")
    after = {p.name for p in RESULTS_DIR.glob("role_benchmark_*.json")}
    created = sorted(after - before)
    if created:
        return RESULTS_DIR / created[-1]
    candidates = sorted(RESULTS_DIR.glob("role_benchmark_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"No benchmark output found for {model}")
    return candidates[-1]


def _aggregate(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_role = data.get("by_role", {})
    strict_vals = []
    guard_vals = []
    latency_vals = []
    errors = 0
    for role_entries in by_role.values():
        if not role_entries:
            continue
        entry = role_entries[0]
        strict_vals.append(float(entry.get("strict_pass_rate", 0.0)))
        guard_vals.append(float(entry.get("guardrail_avg_score", 0.0)))
        latency_vals.append(float(entry.get("avg_latency_ms", 0.0)))
        errors += int(entry.get("errors", 0))
    n = max(len(strict_vals), 1)
    return {
        "file": str(path),
        "avg_strict": sum(strict_vals) / n,
        "avg_guardrail": sum(guard_vals) / n,
        "avg_latency_ms": (sum(latency_vals) / n) if latency_vals else 0.0,
        "total_errors": errors,
    }


def _rank_key(summary: Dict[str, Any]) -> Tuple[int, float, float, float]:
    # Guardrails + stability first, then latency.
    return (
        int(summary["total_errors"]),
        -float(summary["avg_strict"]),
        -float(summary["avg_guardrail"]),
        float(summary["avg_latency_ms"]),
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models = ["qwen3.5:2b", "qwen3.5:4b"]
    summaries: Dict[str, Dict[str, Any]] = {}
    for model in models:
        print(f"\n[bench_qwen35] Running benchmark for {model} ...")
        try:
            out_path = _run_for_model(model)
            summaries[model] = _aggregate(out_path)
        except Exception as exc:
            summaries[model] = {
                "file": "none",
                "avg_strict": 0.0,
                "avg_guardrail": 0.0,
                "avg_latency_ms": 10**9,
                "total_errors": 10**6,
                "error": str(exc),
            }

    ranked = sorted(models, key=lambda m: _rank_key(summaries[m]))
    print("\n[bench_qwen35] Summary:")
    for idx, model in enumerate(ranked, start=1):
        s = summaries[model]
        print(
            f"{idx}. {model} | strict={s['avg_strict']:.3f} "
            f"guardrail={s['avg_guardrail']:.3f} "
            f"errors={s['total_errors']} "
            f"lat={s['avg_latency_ms']:.1f}ms"
        )
        print(f"   file: {s['file']}")
        if s.get("error"):
            print(f"   error: {s['error']}")

    print(f"\nRecommended: {ranked[0]}")


if __name__ == "__main__":
    main()
