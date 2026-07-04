# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Human Escalation Agent.

The Human Escalation Agent prepares handoff summaries for tickets that should be
reviewed by a person. It returns both the user-facing escalation summary and the
rendered prompt.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from agent_workflow.src.prompts import (
    ESCALATION_RESPONSE_TEMPLATE,
    HUMAN_ESCALATION_PROMPT_TEMPLATE,
    format_retrieved_context,
    render_template,
)
from agents.llm import configured_openrouter_model, maybe_generate_text_result
from agents.schemas import AgentOutput
from agents.slack_notification import (
    SlackNotificationError,
    send_slack_message,
    slack_notifications_enabled,
)
from router_agent.src.schemas import RouterResult


LOGGER = logging.getLogger(__name__)


def _format_signals(router_result: RouterResult) -> str:
    return ", ".join(router_result.signals) if router_result.signals else "none"


def build_escalation_prompt(
    ticket_text: str,
    router_result: RouterResult,
    retrieved_docs: Optional[List[Dict]] = None,
    chat_history: str = "No previous conversation.",
) -> str:
    """Build the LLM-style prompt for the Human Escalation Agent."""
    docs = retrieved_docs or []
    return render_template(
        HUMAN_ESCALATION_PROMPT_TEMPLATE,
        chat_history=chat_history,
        ticket_text=ticket_text,
        next_agent=router_result.next_agent,
        confidence=f"{router_result.confidence:.2f}",
        routing_reason=router_result.reason,
        signals=_format_signals(router_result),
        retrieved_context=format_retrieved_context(docs, max_docs=2, max_chars=220),
    )


def generate_escalation_result(
    ticket_text: str,
    router_result: RouterResult,
    retrieved_docs: Optional[List[Dict]] = None,
    chat_history: str = "No previous conversation.",
) -> AgentOutput:
    """Prepare an escalation summary and retain the rendered prompt."""
    docs = retrieved_docs or []
    prompt = build_escalation_prompt(ticket_text, router_result, docs, chat_history)
    llm_response = maybe_generate_text_result(prompt)
    if llm_response:
        response = llm_response.text
        generation_mode = "llm"
        model = configured_openrouter_model()
        usage_details = llm_response.usage_details
        provider_usage = llm_response.provider_usage
    else:
        response = render_template(
            ESCALATION_RESPONSE_TEMPLATE,
            next_agent=router_result.next_agent,
            confidence=f"{router_result.confidence:.2f}",
            routing_reason=router_result.reason,
            signals=_format_signals(router_result),
            ticket_text=ticket_text,
            retrieved_context=format_retrieved_context(docs, max_docs=2, max_chars=180),
        )
        generation_mode = "template"
        model = "none"
        usage_details = None
        provider_usage = None
    _maybe_send_slack_notification(ticket_text, router_result, response)
    return AgentOutput(
        response=response,
        prompt=prompt,
        generation_mode=generation_mode,
        model=model,
        usage_details=usage_details,
        provider_usage=provider_usage,
    )


def generate_escalation_summary(
    ticket_text: str,
    router_result: RouterResult,
    retrieved_docs: Optional[List[Dict]] = None,
    chat_history: str = "No previous conversation.",
) -> str:
    """Backward-compatible wrapper returning only the escalation text."""
    return generate_escalation_result(ticket_text, router_result, retrieved_docs, chat_history).response


def _maybe_send_slack_notification(ticket_text: str, router_result: RouterResult, response: str) -> None:
    if not slack_notifications_enabled():
        return

    message = _build_slack_message(ticket_text, router_result, response)
    try:
        send_slack_message(message)
    except SlackNotificationError as exc:
        LOGGER.warning("Slack escalation notification failed: %s", exc)


def _build_slack_message(ticket_text: str, router_result: RouterResult, response: str) -> str:
    return "\n".join(
        [
            "Human escalation required",
            f"Decision: {router_result.next_agent}",
            f"Confidence: {router_result.confidence:.2f}",
            f"Reason: {router_result.reason}",
            f"Signals: {_format_signals(router_result)}",
            f"User request: {ticket_text}",
            f"Escalation summary: {response}",
        ]
    )
