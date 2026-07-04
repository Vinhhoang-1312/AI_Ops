# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Data structures for the Router component."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RouterResult:
    """Result of the Router decision.

    Attributes:
        next_agent: One of ``"support_resolution_agent"``,
            ``"human_escalation_agent"``, or ``"fallback_agent"``.
        confidence: Confidence in the routing decision, from 0 to 1.
        reason: Explanation of why the Router chose that next agent.
        signals: Lightweight evidence used by the Router decision.
    """

    next_agent: str
    confidence: float
    reason: str
    signals: List[str] = field(default_factory=list)
