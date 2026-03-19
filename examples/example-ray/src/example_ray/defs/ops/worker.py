"""Timed worker wrapper for Ray map operations."""

from __future__ import annotations

import time
from collections.abc import Callable

# --8<-- [start:timed_worker]


def timed_worker(
    row: dict,
    compute_fn: Callable[[dict], dict],
) -> dict:
    """Execute a compute function and record timing and error information."""
    start = time.monotonic()
    try:
        result = compute_fn(row)
        result["processing_time_seconds"] = time.monotonic() - start
        result["error"] = ""
        return result
    except Exception as exc:
        return {
            **row,
            "processing_time_seconds": time.monotonic() - start,
            "error": f"{type(exc).__name__}: {exc}",
        }


# --8<-- [end:timed_worker]
