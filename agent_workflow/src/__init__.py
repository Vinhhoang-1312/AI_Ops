# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Expose workflow orchestration utilities without eager graph imports."""

from .memory import ChatMessage, ShortTermChatMemory  # noqa: F401
from .state import WorkflowState  # noqa: F401

__all__ = ["SupportSession", "run_workflow", "ChatMessage", "ShortTermChatMemory", "WorkflowState"]


def __getattr__(name: str):
    if name in {"SupportSession", "run_workflow"}:
        from .graph import SupportSession, run_workflow

        return {"SupportSession": SupportSession, "run_workflow": run_workflow}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
