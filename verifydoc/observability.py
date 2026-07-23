"""Optional observability: structured trace/event export for verifications (#14).

Decoupled from the pipeline (golden rule #2) — you call these where you run
:func:`verifydoc.verify`. :func:`verification_event` builds a structured dict
(pure, tested); :func:`log_verification` emits it as JSON via stdlib ``logging``
and, if OpenTelemetry is installed and a tracer configured, records a span too.
There is no hard dependency on ``opentelemetry`` — the OTel path is best-effort.

    from verifydoc import verify
    from verifydoc.observability import log_verification

    result = verify("invoice.pdf", schema=SCHEMA)
    log_verification(result)   # -> {"event": "verifydoc.verification", "n_review": 2, ...}
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from verifydoc.pipeline import VerifiedResult

logger = logging.getLogger("verifydoc")


def verification_event(result: VerifiedResult, duration_s: float | None = None) -> dict[str, Any]:
    """Build a structured telemetry event from a verification result (pure)."""
    fields = result.fields
    event: dict[str, Any] = {
        "event": "verifydoc.verification",
        "doc_id": result.doc_id,
        "threshold": result.threshold,
        "n_fields": len(fields),
        "n_accepted": result.n_accepted,
        "n_review": result.n_review,
        "n_grounded": sum(1 for f in fields if f.grounding is not None),
        "mean_confidence": (
            round(sum(f.confidence for f in fields) / len(fields), 4) if fields else 0.0
        ),
    }
    if duration_s is not None:
        event["duration_ms"] = round(duration_s * 1000.0, 1)
    return event


def log_verification(result: VerifiedResult, duration_s: float | None = None) -> dict[str, Any]:
    """Emit the verification event as JSON (stdlib logging) + optional OTel span."""
    event = verification_event(result, duration_s)
    logger.info(json.dumps(event))
    _record_otel_span(event)
    return event


def _record_otel_span(event: dict[str, Any]) -> None:  # pragma: no cover - optional dep
    try:
        from opentelemetry import trace
    except ImportError:
        return
    tracer = trace.get_tracer("verifydoc")
    with tracer.start_as_current_span(str(event["event"])) as span:
        for key, value in event.items():
            span.set_attribute(f"verifydoc.{key}", value)


@contextmanager
def timed() -> Iterator[Any]:
    """Context manager yielding an object whose ``.seconds`` is the elapsed time.

    with timed() as t:
        result = verify(...)
    log_verification(result, t.seconds)
    """

    class _T:
        seconds: float = 0.0

    handle = _T()
    start = time.perf_counter()
    try:
        yield handle
    finally:
        handle.seconds = time.perf_counter() - start
