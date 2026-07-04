# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Fallback response path for requests that should not invoke support agents."""

from __future__ import annotations

from agent_workflow.src.prompts import FALLBACK_PROMPT_TEMPLATE, render_template
from agents.schemas import AgentOutput
from router_agent.src.schemas import RouterResult


FALLBACK_RESPONSE = "Sorry I might not able to process this query"


def generate_fallback_result(
    ticket_text: str,
    router_result: RouterResult,
    chat_history: str = "No previous conversation.",
) -> AgentOutput:
    """Return the fixed fallback response for unsupported or non-actionable input."""

    prompt = render_template(
        FALLBACK_PROMPT_TEMPLATE,
        chat_history=chat_history,
        ticket_text=ticket_text,
        next_agent=router_result.next_agent,
        confidence=f"{router_result.confidence:.2f}",
        routing_reason=router_result.reason,
        signals=", ".join(router_result.signals) if router_result.signals else "none",
    )
    return AgentOutput(response=FALLBACK_RESPONSE, prompt=prompt)
