"""Langfuse integration helpers.

The workflow uses these helpers to keep Langfuse optional and fail-open.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

from langfuse import get_client, propagate_attributes


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass
class LangfuseTraceContext:
    client: Any
    trace_id: str
    root_observation: Any


def langfuse_enabled() -> bool:
    """Return whether Langfuse tracing is enabled."""
    raw = os.getenv("IT_SUPPORT_LANGFUSE_ENABLED")
    if raw is None:
        return False
    return raw.strip().lower() in TRUE_VALUES


def get_langfuse_client():
    """Return the current Langfuse client when enabled and configured."""
    if not langfuse_enabled():
        return None
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    if not public_key or not secret_key:
        return None
    try:
        client = get_client()
    except Exception:
        return None

    if client is None:
        return None
    return client


def start_ticket_trace(
    *,
    ticket_text: str,
    session_id: str | None = None,
    user_id: str | None = None,
):
    """Start a root trace for one ticket."""
    client = get_langfuse_client()
    if client is None:
        return nullcontext(None)

    metadata = {
        "app": "agentic-rag-based-it-support",
        "app_version": os.getenv("IT_SUPPORT_APP_VERSION", "dev"),
        "environment": os.getenv("IT_SUPPORT_ENVIRONMENT", "local"),
        "trace_type": "it-support-ticket",
    }

    return _ticket_trace_context(
        client,
        ticket_text=ticket_text,
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
    )


@contextmanager
def _ticket_trace_context(
    client,
    *,
    ticket_text: str,
    session_id: str | None,
    user_id: str | None,
    metadata: dict[str, Any],
):
    trace_id = client.create_trace_id()
    try:
        with propagate_attributes(
            trace_name="it-support-ticket",
            session_id=session_id,
            user_id=user_id,
            tags=["it-support", "rag", "router", "local"],
            metadata=metadata,
            input={"ticket_text": ticket_text},
        ):
            with client.start_as_current_observation(
                trace_context={"trace_id": trace_id},
                name="it-support-ticket",
                as_type="chain",
                input={"ticket_text": ticket_text},
                metadata=metadata,
            ) as root_observation:
                yield LangfuseTraceContext(
                    client=client,
                    trace_id=trace_id,
                    root_observation=root_observation,
                )
    except Exception:
        yield None


def add_score(
    name: str,
    value: float | str,
    data_type: str | None = None,
    comment: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Attach a score to the current trace when available."""
    client = get_langfuse_client()
    if client is None:
        return

    kwargs: dict[str, Any] = {
        "name": name,
        "value": value,
    }
    if data_type:
        kwargs["data_type"] = data_type
    if comment:
        kwargs["comment"] = comment

    try:
        if trace_id:
            client.create_score(trace_id=trace_id, **kwargs)
        else:
            client.score_current_trace(**kwargs)
    except Exception:
        return


def flush_langfuse() -> None:
    """Flush pending Langfuse events when available."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        return
