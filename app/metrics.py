"""Timing helpers for latency measurements."""

import time


def now_ms() -> float:
    """Return a monotonic timestamp in milliseconds."""
    return time.perf_counter() * 1000.0


def duration_ms(start_ms: float, end_ms: float) -> float:
    """Return a non-negative duration in milliseconds."""
    return max(0.0, end_ms - start_ms)
