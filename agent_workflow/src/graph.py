# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Workflow orchestrator for the agentic support system.

This module wires together the Router component, the RAG retriever, the Support
Resolution Agent and the Human Escalation Agent. It exposes ``run_workflow`` to
process one ticket end to end and ``SupportSession`` to run multiple turns with
short-term chat memory.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional
from uuid import uuid4

from agents.fallback_agent import generate_fallback_result
from agents.human_escalation_agent import generate_escalation_result
from agents.support_resolution_agent import generate_response_result
from observability.langfuse_observer import add_score, get_langfuse_client, start_ticket_trace
from rag_pipeline.src.retriever import build_retriever
from router_agent.src.router import LOW_CONFIDENCE_ESCALATION_THRESHOLD, Router
from router_agent.src.schemas import RouterResult

from .memory import ShortTermChatMemory
from .state import WorkflowState


def apply_low_confidence_escalation(router_result: RouterResult) -> RouterResult:
    """Escalate support-resolution decisions when Router confidence is below 0.5."""
    if router_result.next_agent != "support_resolution_agent":
        return router_result
    if router_result.confidence >= LOW_CONFIDENCE_ESCALATION_THRESHOLD:
        return router_result

    signals = list(router_result.signals)
    for signal in ("low router confidence", "human review preferred"):
        if signal not in signals:
            signals.append(signal)
    return RouterResult(
        next_agent="human_escalation_agent",
        confidence=router_result.confidence,
        reason=(
            "The Router confidence for automatic support resolution is below "
            f"{LOW_CONFIDENCE_ESCALATION_THRESHOLD:.2f}, so the ticket should be reviewed by human support."
        ),
        signals=signals,
    )


def run_workflow(
    ticket_text: str,
    top_k: int = 3,
    memory: Optional[ShortTermChatMemory] = None,
    retriever: Optional[Any] = None,
    router: Optional[Router] = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> WorkflowState:
    """Process a support ticket through routing, retrieval and response.

    Args:
        ticket_text: The user's support request.
        top_k: Number of documents to retrieve from the RAG pipeline.
        memory: Optional short-term chat memory. When provided, recent messages
            are rendered into the prompt and the final exchange is appended to
            memory after the response is generated.
        retriever: Optional pre-created retriever for reuse across turns.
        router: Optional pre-created router for reuse across turns.

    Returns:
        A ``WorkflowState`` with the router decision, retrieved documents,
        rendered prompt, chat history and final response or escalation summary.
    """
    with start_ticket_trace(ticket_text=ticket_text, session_id=session_id, user_id=user_id) as trace_context:
        langfuse = trace_context.client if trace_context is not None else get_langfuse_client()
        chat_history = _load_chat_history(memory=memory, langfuse=langfuse)
        state = WorkflowState(ticket_text=ticket_text, chat_history=chat_history)

        router = router or Router()
        with _langfuse_observation(langfuse, name="router-decision", as_type="span") as span:
            router_result = apply_low_confidence_escalation(router.route_ticket(ticket_text))
            _safe_span_update(
                span,
                input={
                    "ticket_text": ticket_text,
                    "chat_history_available": chat_history != "No previous conversation.",
                },
                output={
                    "next_agent": router_result.next_agent,
                    "confidence": router_result.confidence,
                    "reason": router_result.reason,
                    "signals": getattr(router_result, "signals", []),
                },
            )
        state.router_result = router_result
        state.selected_agent = router_result.next_agent

        if router_result.next_agent == "fallback_agent":
            state.retrieved_docs = []
            agent_output = generate_fallback_result(
                ticket_text=ticket_text,
                router_result=router_result,
                chat_history=chat_history,
            )
        else:
            retriever = retriever or build_retriever()
            with _langfuse_observation(langfuse, name="techqa-retrieval", as_type="retriever") as span:
                state.retrieved_docs = retriever.retrieve(ticket_text, top_k=top_k)
                _safe_span_update(
                    span,
                    input={
                        "query": ticket_text,
                        "top_k": top_k,
                        "retriever_backend": getattr(retriever, "backend", "tfidf"),
                    },
                    output=[_serialize_retrieved_doc(doc) for doc in state.retrieved_docs],
                    metadata={"retrieved_count": len(state.retrieved_docs)},
                )

        if router_result.next_agent == "support_resolution_agent":
            agent_output = generate_response_result(
                ticket_text=ticket_text,
                router_result=router_result,
                retrieved_docs=state.retrieved_docs,
                chat_history=chat_history,
            )
            agent_span_name = "support-resolution"
        elif router_result.next_agent == "human_escalation_agent":
            agent_output = generate_escalation_result(
                ticket_text=ticket_text,
                router_result=router_result,
                retrieved_docs=state.retrieved_docs,
                chat_history=chat_history,
            )
            agent_span_name = "human-escalation"
        elif router_result.next_agent == "fallback_agent":
            agent_span_name = "fallback"
        else:
            raise ValueError(f"Unsupported router next_agent: {router_result.next_agent}")

        state.final_response = agent_output.response
        state.rendered_prompt = agent_output.prompt

        with _langfuse_observation(langfuse, name="render-prompt", as_type="span") as span:
            _safe_span_update(
                span,
                input={
                    "selected_agent": state.selected_agent,
                    "template_name": _prompt_template_name(state.selected_agent),
                },
                output={"rendered_prompt": agent_output.prompt},
            )

        agent_metadata = {
            "selected_agent": state.selected_agent,
            "generation_mode": agent_output.generation_mode,
            "model": agent_output.model,
        }
        if agent_output.provider_usage:
            agent_metadata["provider_usage"] = agent_output.provider_usage

        agent_observation_kwargs: dict[str, Any] = {}
        if agent_output.generation_mode == "llm":
            agent_observation_kwargs = {
                "input": agent_output.prompt,
                "output": agent_output.response,
                "model": agent_output.model,
            }
            if agent_output.usage_details:
                agent_observation_kwargs["usage_details"] = agent_output.usage_details

        with _langfuse_observation(
            langfuse,
            name=agent_span_name,
            as_type="generation" if agent_output.generation_mode == "llm" else "span",
            **agent_observation_kwargs,
        ) as span:
            update_kwargs: dict[str, Any] = {
                "output": agent_output.response
                if agent_output.generation_mode == "llm"
                else {"final_response": agent_output.response},
                "metadata": agent_metadata,
            }
            if agent_output.usage_details:
                update_kwargs["usage_details"] = agent_output.usage_details
            _safe_span_update(span, **update_kwargs)

        if memory is not None:
            before = len(memory)
            with _langfuse_observation(langfuse, name="update-memory", as_type="span") as span:
                memory.add_user_message(ticket_text)
                memory.add_assistant_message(agent_output.response)
                _safe_span_update(
                    span,
                    output={
                        "memory_messages_before": before,
                        "memory_messages_after": len(memory),
                        "max_messages": memory.max_messages,
                    },
                )

        _record_scores(state, trace_id=trace_context.trace_id if trace_context is not None else None)
        _safe_span_update(
            trace_context.root_observation if trace_context is not None else None,
            output={
                "selected_agent": state.selected_agent,
                "final_response": state.final_response,
            },
        )
        return state


def _load_chat_history(memory: Optional[ShortTermChatMemory], langfuse: Any) -> str:
    before = len(memory) if memory is not None else 0
    with _langfuse_observation(langfuse, name="load-memory", as_type="span") as span:
        chat_history = memory.to_text() if memory is not None else "No previous conversation."
        _safe_span_update(
            span,
            output={
                "memory_messages_before": before,
                "chat_history_available": chat_history != "No previous conversation.",
                "max_messages": getattr(memory, "max_messages", 0),
            },
        )
        return chat_history


def _langfuse_observation(langfuse: Any, *, name: str, as_type: str = "span", **kwargs: Any):
    if langfuse is None:
        return nullcontext(None)
    try:
        return langfuse.start_as_current_observation(name=name, as_type=as_type, **kwargs)
    except Exception:
        return nullcontext(None)


def _safe_span_update(span: Any, **kwargs: Any) -> None:
    if span is None:
        return
    try:
        span.update(**kwargs)
    except Exception:
        return


def _record_scores(state: WorkflowState, *, trace_id: str | None = None) -> None:
    add_score(
        name="has_final_response",
        value=1.0 if state.final_response else 0.0,
        data_type="BOOLEAN",
        trace_id=trace_id,
    )
    add_score(
        name="retrieved_context_count",
        value=float(len(state.retrieved_docs)),
        data_type="NUMERIC",
        trace_id=trace_id,
    )
    add_score(
        name="routed_to_human",
        value=1.0 if state.selected_agent == "human_escalation_agent" else 0.0,
        data_type="BOOLEAN",
        trace_id=trace_id,
    )


def _serialize_retrieved_doc(doc: dict[str, Any]) -> dict[str, Any]:
    content = str(doc.get("content") or "").strip()
    snippet = ""
    if content:
        snippet = content[:180].rsplit(" ", 1)[0] if len(content) > 180 else content
    payload = {
        "doc_id": doc.get("doc_id") or doc.get("id"),
        "title": doc.get("title"),
        "score": doc.get("score"),
        "source": doc.get("source"),
        "category": doc.get("category"),
    }
    if snippet:
        payload["snippet"] = snippet
    return payload


def _prompt_template_name(selected_agent: str | None) -> str:
    if selected_agent == "support_resolution_agent":
        return "SUPPORT_RESOLUTION_PROMPT_TEMPLATE"
    if selected_agent == "human_escalation_agent":
        return "HUMAN_ESCALATION_PROMPT_TEMPLATE"
    if selected_agent == "fallback_agent":
        return "FALLBACK_PROMPT"
    return "UNKNOWN"


class SupportSession:
    """Multi-turn wrapper around ``run_workflow`` with short-term memory.

    The session keeps a bounded in-process memory and reuses Router/Retriever
    instances to avoid repeated initialization in an interactive demo.
    """

    def __init__(
        self,
        max_messages: int = 10,
        top_k: int = 3,
        retriever_backend: Optional[str] = None,
        router_backend: Optional[str] = None,
    ) -> None:
        self.memory = ShortTermChatMemory(max_messages=max_messages)
        self.top_k = top_k
        self.router = Router(backend=router_backend)
        self.retriever = build_retriever(backend=retriever_backend)
        self.session_id = f"support-session-{uuid4()}"

    def ask(self, ticket_text: str) -> WorkflowState:
        """Run one ticket through the workflow and update session memory."""
        return run_workflow(
            ticket_text,
            top_k=self.top_k,
            memory=self.memory,
            retriever=self.retriever,
            router=self.router,
            session_id=self.session_id,
        )

    def clear_memory(self) -> None:
        """Clear short-term chat memory."""
        self.memory.clear()
