# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""State definitions for the agent workflow.

The workflow state holds intermediate information as a ticket is processed by
the Router component, retriever and response agents. The upgraded state also
records the short-term chat history used for the prompt and the rendered prompt
that would be sent to an LLM in a production implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from router_agent.src.schemas import RouterResult


@dataclass
class WorkflowState:
    """Container for maintaining state during one workflow execution."""

    ticket_text: str
    router_result: Optional[RouterResult] = None
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    final_response: Optional[str] = None
    selected_agent: Optional[str] = None
    chat_history: str = "No previous conversation."
    rendered_prompt: Optional[str] = None
    error: Optional[str] = None
