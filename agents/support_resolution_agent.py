# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Support Resolution Agent.

The Support Resolution Agent handles tickets that the Router selected for
automatic handling. It returns the final response plus the rendered prompt used
for optional hosted-LLM generation.
"""

from __future__ import annotations

from typing import Dict, List

from agent_workflow.src.prompts import (
    NO_CONTEXT_RESPONSE_TEMPLATE,
    SUPPORT_RESOLUTION_PROMPT_TEMPLATE,
    SUPPORT_RESPONSE_TEMPLATE,
    format_retrieved_context,
    render_template,
)
from agents.llm import configured_openrouter_model, maybe_generate_text_result
from agents.schemas import AgentOutput
from router_agent.src.schemas import RouterResult


def _snippet(text: str, max_chars: int = 180) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _follow_up_note(content: str) -> str:
    lower_content = content.lower()
    if any(keyword in lower_content for keyword in ("approval", "contact", "human", "stop using")):
        return (
            "If you need further assistance or the issue involves permissions, approval, "
            "or safety, please contact support or your manager."
        )
    return ""


def _format_signals(router_result: RouterResult) -> str:
    return ", ".join(router_result.signals) if router_result.signals else "none"


def build_support_prompt(
    ticket_text: str,
    router_result: RouterResult,
    retrieved_docs: List[Dict],
    chat_history: str = "No previous conversation.",
) -> str:
    """Build the LLM-style prompt for the Support Resolution Agent."""
    return render_template(
        SUPPORT_RESOLUTION_PROMPT_TEMPLATE,
        chat_history=chat_history,
        ticket_text=ticket_text,
        next_agent=router_result.next_agent,
        confidence=f"{router_result.confidence:.2f}",
        routing_reason=router_result.reason,
        signals=_format_signals(router_result),
        retrieved_context=format_retrieved_context(retrieved_docs),
    )


def generate_response_result(
    ticket_text: str,
    router_result: RouterResult,
    retrieved_docs: List[Dict],
    chat_history: str = "No previous conversation.",
) -> AgentOutput:
    """Generate a support response and retain the rendered prompt."""
    prompt = build_support_prompt(ticket_text, router_result, retrieved_docs, chat_history)
    llm_response = maybe_generate_text_result(prompt)
    if llm_response:
        return AgentOutput(
            response=llm_response.text,
            prompt=prompt,
            generation_mode="llm",
            model=configured_openrouter_model(),
            usage_details=llm_response.usage_details,
            provider_usage=llm_response.provider_usage,
        )

    if not retrieved_docs:
        response = render_template(NO_CONTEXT_RESPONSE_TEMPLATE)
        return AgentOutput(response=response, prompt=prompt, generation_mode="template", model="none")

    top_doc = retrieved_docs[0]
    content = str(top_doc.get("content", ""))
    response = render_template(
        SUPPORT_RESPONSE_TEMPLATE,
        top_doc_title=top_doc.get("title", "Support information"),
        top_doc_snippet=_snippet(content),
        follow_up_note=_follow_up_note(content),
    )
    return AgentOutput(response=response, prompt=prompt, generation_mode="template", model="none")


def generate_response(
    ticket_text: str,
    router_result: RouterResult,
    retrieved_docs: List[Dict],
    chat_history: str = "No previous conversation.",
) -> str:
    """Backward-compatible wrapper returning only the response string."""
    return generate_response_result(ticket_text, router_result, retrieved_docs, chat_history).response
