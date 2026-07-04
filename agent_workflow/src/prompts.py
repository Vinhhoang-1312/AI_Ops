# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Prompt templates and rendering helpers for deterministic or LLM responses."""

from __future__ import annotations

from collections import UserDict
from typing import Iterable, Mapping


class _SafeFormatDict(UserDict):
    """Dictionary that leaves unknown template variables visible."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_template(template: str, **kwargs: object) -> str:
    """Render a template with safe ``str.format_map`` substitution."""
    values = _SafeFormatDict({key: "" if value is None else str(value) for key, value in kwargs.items()})
    return template.format_map(values).strip()


def format_retrieved_context(docs: Iterable[Mapping[str, object]], max_docs: int = 3, max_chars: int = 500) -> str:
    """Render retrieved documents into a compact context block."""
    rendered = []
    for index, doc in enumerate(list(docs)[:max_docs], start=1):
        title = str(doc.get("title", "Untitled"))
        source = str(doc.get("source", "unknown"))
        score = doc.get("score", "")
        content = str(doc.get("content", ""))
        if len(content) > max_chars:
            content = content[:max_chars].rsplit(" ", 1)[0] + "..."
        score_text = f", score={score:.3f}" if isinstance(score, float) else ""
        rendered.append(f"[{index}] {title} (source={source}{score_text})\n{content}")
    return "\n\n".join(rendered) if rendered else "No relevant context retrieved."


SUPPORT_RESOLUTION_PROMPT_TEMPLATE = """
System:
You are an IT support assistant. Your job is to produce a concise, practical,
grounded response to a user ticket. Use the retrieved support knowledge when it
is relevant. Do not invent policy details. If the available information is
insufficient, recommend human support or escalation.

Conversation history:
{chat_history}

Current user ticket:
{ticket_text}

Router decision:
- Selected agent: {next_agent}
- Confidence: {confidence}
- Reason: {routing_reason}
- Signals: {signals}

Retrieved support knowledge:
{retrieved_context}

Task:
Write the final support response for the user.
"""


HUMAN_ESCALATION_PROMPT_TEMPLATE = """
System:
You are preparing a ticket handoff for a human support representative. Be clear,
brief, and operational. Include why the request should be reviewed by a human.

Conversation history:
{chat_history}

Current user ticket:
{ticket_text}

Router decision:
- Selected agent: {next_agent}
- Confidence: {confidence}
- Reason: {routing_reason}
- Signals: {signals}

Retrieved support knowledge, if any:
{retrieved_context}

Task:
Write a human escalation summary with suggested next steps.
"""


FALLBACK_PROMPT_TEMPLATE = """
System:
You are handling a user message that should not trigger either automatic support
resolution or human escalation. The input is either out of scope, too unclear,
non-actionable, or unsafe for this IT support workflow.

Conversation history:
{chat_history}

Current user ticket:
{ticket_text}

Router decision:
- Selected agent: {next_agent}
- Confidence: {confidence}
- Reason: {routing_reason}
- Signals: {signals}

Task:
Reply exactly with: Sorry I might not able to process this query
"""


SUPPORT_RESPONSE_TEMPLATE = """
Relevant document: {top_doc_title}.
{top_doc_snippet}
{follow_up_note}
"""


NO_CONTEXT_RESPONSE_TEMPLATE = """
I could not find relevant support knowledge for this request.
Please contact human support for assistance or provide more details so the request can be reviewed.
"""


ESCALATION_RESPONSE_TEMPLATE = """
This request requires review by a human support agent.
Router decision: {next_agent}
Confidence: {confidence}
Reason for escalation: {routing_reason}
Signals: {signals}

User request:
{ticket_text}

Relevant context:
{retrieved_context}

Suggested next steps: Please verify the request details, business justification,
impact, required duration, affected system/device, and the applicable approval workflow before taking action.
"""
