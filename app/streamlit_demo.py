# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Streamlit demo for the Agentic RAG-based IT Support workflow."""

from __future__ import annotations

import html
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_workflow.src.graph import SupportSession
from agent_workflow.src.state import WorkflowState
from observability.langfuse_observer import flush_langfuse


LOGGER = logging.getLogger(__name__)

MESSAGES_KEY = "its_messages"
PENDING_TICKET_KEY = "its_pending_ticket"
SESSION_KEY = "its_support_session"
SESSION_CONFIG_KEY = "its_support_session_config"
SHOW_CONTEXT_KEY = "its_show_context"
SHOW_PROMPT_KEY = "its_show_prompt"
SHOW_DIAGNOSTICS_KEY = "its_show_diagnostics"
TOP_K_KEY = "its_top_k"
MEMORY_LIMIT_KEY = "its_memory_limit"
RETRIEVER_BACKEND_KEY = "its_retriever_backend"
ROUTER_BACKEND_KEY = "its_router_backend"

EXAMPLE_TICKETS = [
    {
        "id": "appscan_silent_install",
        "label": "AppScan silent install",
        "ticket": (
            'IBM Rational AppScan Source silent install on RHEL fails with '
            '"This is not a supported operating system" after running setup.bin -i Silent.'
        ),
        "help": "Retrieves the AppScan Source RHEL silent-install technote.",
    },
    {
        "id": "mq_data_conversion",
        "label": "WebSphere MQ errors",
        "ticket": (
            "A WebSphere MQ V7 Java client fails to connect and logs AMQ9208, "
            "AMQ6048, and AMQ9492 during CCSID 1208 to CCSID 37 conversion."
        ),
        "help": "Retrieves the WebSphere MQ APAR about Java client connection failures.",
    },
    {
        "id": "appscan_ports",
        "label": "AppScan proxy ports",
        "ticket": (
            "I need to configure firewall rules for IBM Security AppScan Standard. "
            "What ports does its desktop traffic proxy use?"
        ),
        "help": "Retrieves the AppScan Standard ports FAQ.",
    },
    {
        "id": "portal_policy_mustgather",
        "label": "Portal policy logs",
        "ticket": (
            "I need to troubleshoot a WebSphere Portal policy issue. "
            "What trace settings and XMLAccess export data should I collect?"
        ),
        "help": "Retrieves the WebSphere Portal policy MustGather article.",
    },
    {
        "id": "odm_interim_fix",
        "label": "ODM interim fix",
        "ticket": (
            "I need to install IBM Operational Decision Manager 8.6.0.0 "
            "Interim Fix 26 for APAR RS02224. Where is the download and readme?"
        ),
        "help": "Retrieves the Operational Decision Manager interim-fix record.",
    },
    {
        "id": "hardware_safety",
        "label": "Swollen battery",
        "ticket": "My laptop battery is swollen and the screen flickers when it is plugged in.",
        "help": "Routes to human escalation because of safety risk.",
    },
]


def main() -> None:
    """Render the local Streamlit demo UI."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("Install `streamlit` to run the demo app.") from exc

    st.set_page_config(
        page_title="Agentic RAG-based IT Support",
        layout="wide",
    )
    _init_session_state(st)
    controls = _sidebar(st)
    try:
        session = _support_session(st, controls)
    except Exception as exc:  # pragma: no cover - displayed in UI
        LOGGER.exception("Failed to initialize Streamlit support session")
        _render_startup_error(st, exc)
        return

    console_tab, corpus_tab = st.tabs(["Support console", "TechQA corpus"])
    with console_tab:
        _render_support_console(st, session, controls)
    with corpus_tab:
        _render_corpus_tab(st, session)


def _init_session_state(st: Any) -> None:
    st.session_state.setdefault(MESSAGES_KEY, [])
    st.session_state.setdefault(SHOW_CONTEXT_KEY, True)
    st.session_state.setdefault(SHOW_PROMPT_KEY, False)
    st.session_state.setdefault(SHOW_DIAGNOSTICS_KEY, False)
    st.session_state.setdefault(TOP_K_KEY, 3)
    st.session_state.setdefault(MEMORY_LIMIT_KEY, 10)
    default_retriever_backend = os.getenv("IT_SUPPORT_RETRIEVER_BACKEND", "auto").strip().lower()
    if default_retriever_backend not in {"auto", "tfidf", "weaviate"}:
        default_retriever_backend = "auto"
    st.session_state.setdefault(RETRIEVER_BACKEND_KEY, default_retriever_backend)
    default_router_backend = os.getenv("IT_SUPPORT_ROUTER_BACKEND", "auto").strip().lower()
    if default_router_backend not in {"auto", "openvino", "nli", "heuristic"}:
        default_router_backend = "auto"
    st.session_state.setdefault(ROUTER_BACKEND_KEY, default_router_backend)


def _sidebar(st: Any) -> dict[str, Any]:
    with st.sidebar:
        st.title("Agentic RAG-based IT Support")
        st.caption("Router + TechQA RAG + support agents")
        st.markdown("#### Ticket examples")
        for example in EXAMPLE_TICKETS:
            if st.button(
                example["label"],
                key=f"example-{example['id']}",
                help=example["help"],
                use_container_width=True,
            ):
                st.session_state[PENDING_TICKET_KEY] = example["ticket"]
                st.rerun()

        if st.button("Clear conversation", use_container_width=True):
            _clear_conversation(st)
            st.rerun()

        st.divider()
        router_backend = st.selectbox(
            "Router backend",
            options=("auto", "openvino", "nli", "heuristic"),
            key=ROUTER_BACKEND_KEY,
            help="Auto tries OpenVINO IR, then safetensors NLI, then heuristics.",
        )
        backend = st.selectbox(
            "Retrieval backend",
            options=("auto", "weaviate", "tfidf"),
            key=RETRIEVER_BACKEND_KEY,
            help="Auto tries local Weaviate with Ollama embeddings first and falls back to TF-IDF.",
        )
        top_k = st.slider(
            "Retrieved TechQA documents",
            min_value=1,
            max_value=8,
            key=TOP_K_KEY,
            help="Number of TechQA records retrieved for each ticket.",
        )
        memory_limit = st.slider(
            "Memory messages",
            min_value=2,
            max_value=20,
            step=2,
            key=MEMORY_LIMIT_KEY,
            help="Bounded chat history stored only in this Streamlit session.",
        )
        show_context = st.toggle(
            "Show retrieved context",
            key=SHOW_CONTEXT_KEY,
        )
        show_prompt = st.toggle(
            "Show rendered prompt",
            key=SHOW_PROMPT_KEY,
        )
        show_diagnostics = st.toggle(
            "Expand diagnostics",
            key=SHOW_DIAGNOSTICS_KEY,
        )

    return {
        "top_k": int(top_k),
        "retriever_backend": str(backend),
        "router_backend": str(router_backend),
        "memory_limit": int(memory_limit),
        "show_context": bool(show_context),
        "show_prompt": bool(show_prompt),
        "show_diagnostics": bool(show_diagnostics),
    }


def _support_session(st: Any, controls: dict[str, Any]) -> SupportSession:
    config = (
        controls["top_k"],
        controls["memory_limit"],
        controls["retriever_backend"],
        controls["router_backend"],
    )
    existing_config = st.session_state.get(SESSION_CONFIG_KEY)
    if SESSION_KEY not in st.session_state or existing_config != config:
        LOGGER.info(
            "Creating Streamlit SupportSession top_k=%s memory_limit=%s retriever_backend=%s router_backend=%s",
            controls["top_k"],
            controls["memory_limit"],
            controls["retriever_backend"],
            controls["router_backend"],
        )
        st.session_state[SESSION_KEY] = SupportSession(
            max_messages=controls["memory_limit"],
            top_k=controls["top_k"],
            retriever_backend=controls["retriever_backend"],
            router_backend=controls["router_backend"],
        )
        st.session_state[SESSION_CONFIG_KEY] = config
        if existing_config is not None:
            st.session_state[MESSAGES_KEY] = []
    return st.session_state[SESSION_KEY]


def _clear_conversation(st: Any) -> None:
    st.session_state[MESSAGES_KEY] = []
    session = st.session_state.get(SESSION_KEY)
    if session is not None:
        session.clear_memory()


def _render_startup_error(st: Any, exc: Exception) -> None:
    st.title("Support console is unavailable")
    st.write("The workflow could not initialize its Router or TechQA retrieval session.")
    st.error("The Streamlit demo could not initialize the support session.", icon=None)
    st.caption(str(exc))
    st.code(
        "For Weaviate mode, verify Ollama and Weaviate are running and TechQADocument is populated. For TF-IDF fallback, check that data/TechQA exists and Python dependencies are installed.",
        language="text",
    )


def _render_support_console(st: Any, session: SupportSession, controls: dict[str, Any]) -> None:
    st.title("Support triage console")
    st.write(
        "Decide whether a ticket should be handled automatically or escalated, "
        "then retrieve TechQA evidence for the selected agent."
    )

    _render_history(st, controls)
    pending = st.session_state.pop(PENDING_TICKET_KEY, None)
    submitted = st.chat_input("Describe an IT support issue or ask a technical support question.")
    prompt = pending or submitted
    if prompt:
        _run_turn(st, session, str(prompt), controls)


def _render_history(st: Any, controls: dict[str, Any]) -> None:
    for item in st.session_state[MESSAGES_KEY]:
        with st.chat_message(item["role"]):
            if item["role"] == "user":
                st.write(item["content"])
            else:
                _render_result(st, item["result"], controls)


def _run_turn(st: Any, session: SupportSession, prompt: str, controls: dict[str, Any]) -> None:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return

    started = time.perf_counter()
    st.session_state[MESSAGES_KEY].append({"role": "user", "content": cleaned_prompt})
    with st.chat_message("user"):
        st.write(cleaned_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing ticket, retrieving TechQA context, and drafting response..."):
            try:
                state = session.ask(cleaned_prompt)
            except Exception as exc:  # pragma: no cover - displayed in UI
                LOGGER.exception("Streamlit support turn failed")
                st.error("The support workflow failed to process this ticket.", icon=None)
                st.caption(str(exc))
                return
        payload = _state_to_payload(state, elapsed_ms=(time.perf_counter() - started) * 1000)
        _render_result(st, payload, controls)
        flush_langfuse()

    st.session_state[MESSAGES_KEY].append({"role": "assistant", "result": payload})


def _state_to_payload(state: WorkflowState, *, elapsed_ms: float) -> dict[str, Any]:
    router_payload: dict[str, Any] = {}
    if state.router_result is not None:
        router_payload = asdict(state.router_result) if is_dataclass(state.router_result) else dict(state.router_result)
    return {
        "ticket_text": state.ticket_text,
        "router": router_payload,
        "selected_agent": state.selected_agent,
        "retrieved_docs": list(state.retrieved_docs or []),
        "final_response": state.final_response or "No response was produced.",
        "chat_history": state.chat_history,
        "rendered_prompt": state.rendered_prompt,
        "elapsed_ms": elapsed_ms,
        "error": state.error,
    }


def _render_result(st: Any, payload: dict[str, Any], controls: dict[str, Any]) -> None:
    _render_status_strip(st, payload)
    _render_metrics(st, payload)
    st.markdown("#### Final response")
    st.markdown(str(payload.get("final_response") or "No response was produced."))
    if controls["show_context"]:
        _render_retrieved_docs(st, payload)
    if controls["show_prompt"] or controls["show_diagnostics"]:
        _render_diagnostics(st, payload, expanded=controls["show_diagnostics"])


def _render_status_strip(st: Any, payload: dict[str, Any]) -> None:
    router = dict(payload.get("router") or {})
    docs = list(payload.get("retrieved_docs") or [])
    chips = [
        f"decision: {_human_agent_name(str(payload.get('selected_agent') or router.get('next_agent') or 'unknown'))}",
        f"signals: {len(router.get('signals') or [])}",
        f"docs: {len(docs)}",
        f"backend: {_retrieval_backend_label(docs)}",
        f"latency: {float(payload.get('elapsed_ms') or 0):.0f} ms",
    ]
    st.caption(" | ".join(chips))


def _render_metrics(st: Any, payload: dict[str, Any]) -> None:
    router = dict(payload.get("router") or {})
    docs = list(payload.get("retrieved_docs") or [])
    confidence = router.get("confidence")
    confidence_text = "n/a"
    if isinstance(confidence, (int, float)):
        confidence_text = f"{confidence:.2f}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Decision", _human_agent_name(str(payload.get("selected_agent") or router.get("next_agent") or "Unknown")))
    col2.metric("Confidence", confidence_text)
    col3.metric("Signals", str(len(router.get("signals") or [])))
    col4.metric("Retrieved", str(len(docs)))


def _render_retrieved_docs(st: Any, payload: dict[str, Any]) -> None:
    docs = list(payload.get("retrieved_docs") or [])
    with st.expander(f"Retrieved TechQA context ({len(docs)})", expanded=False):
        if not docs:
            st.write("No TechQA documents were retrieved.")
            return
        for index, doc in enumerate(docs, start=1):
            title = str(doc.get("title") or doc.get("doc_id") or "TechQA document")
            score = doc.get("score")
            score_text = f"score={score:.3f}" if isinstance(score, (int, float)) else "score=n/a"
            metadata = [
                score_text,
                f"source={doc.get('source', 'TechQA')}",
                f"category={doc.get('category', 'unknown')}",
                f"doc_id={doc.get('doc_id') or doc.get('id') or 'unknown'}",
                f"backend={doc.get('retrieval_backend', 'tfidf')}",
            ]
            if doc.get("vector_model"):
                metadata.append(f"model={doc.get('vector_model')}")
            if doc.get("retrieval_fallback_reason"):
                metadata.append("fallback=tfidf")
            if doc.get("category_filter_fallback"):
                metadata.append(f"fallback_from={doc.get('category_filter')}")
            st.markdown(f"**{index}. {html.escape(title)}**")
            st.caption(" | ".join(str(item) for item in metadata))
            st.write(_snippet(str(doc.get("content") or ""), 520))
            st.divider()


def _render_diagnostics(st: Any, payload: dict[str, Any], *, expanded: bool) -> None:
    with st.expander("Router, memory, and prompt diagnostics", expanded=expanded):
        st.markdown("#### Router decision")
        st.json(payload.get("router") or {})
        st.markdown("#### Chat history used for this turn")
        st.code(str(payload.get("chat_history") or ""), language="text")
        prompt = str(payload.get("rendered_prompt") or "").strip()
        if prompt:
            st.markdown("#### Rendered agent prompt")
            st.code(prompt, language="text")
        st.markdown("#### Raw workflow payload")
        st.json(payload)


def _render_corpus_tab(st: Any, session: SupportSession) -> None:
    documents = list(getattr(session.retriever, "documents", []) or [])
    st.title("TechQA corpus")
    st.write("Inspect the normalized technical support records currently used by the retriever.")

    source_counts = Counter(str(doc.get("source") or "unknown") for doc in documents)
    category_counts = Counter(str(doc.get("category") or "unknown") for doc in documents)
    col1, col2, col3 = st.columns(3)
    col1.metric("Loaded records", str(len(documents)))
    col2.metric("Sources", str(len(source_counts)))
    col3.metric("Categories", str(len(category_counts)))
    st.caption(
        f"Data path: `{getattr(session.retriever, 'data_path', 'unknown')}` | "
        f"Retriever backend: `{getattr(session.retriever, 'backend', 'tfidf')}` | Router backend: `{getattr(session.router, 'backend', 'auto')}`"
    )

    query = st.text_input("Filter loaded TechQA records", placeholder="Search title, content, product, or document id")
    filtered = _filter_documents(documents, query)
    st.caption(f"Showing {min(len(filtered), 50)} of {len(filtered)} matching records")
    rows = [
        {
            "doc_id": doc.get("doc_id") or doc.get("id"),
            "title": doc.get("title"),
            "category": doc.get("category"),
            "source": doc.get("source"),
            "snippet": _snippet(str(doc.get("content") or ""), 180),
        }
        for doc in filtered[:50]
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)

    with st.expander("Category distribution", expanded=False):
        st.dataframe(
            [{"category": key, "records": value} for key, value in category_counts.most_common()],
            hide_index=True,
            use_container_width=True,
        )


def _filter_documents(documents: list[dict[str, Any]], query: str | None) -> list[dict[str, Any]]:
    needle = (query or "").strip().casefold()
    if not needle:
        return documents
    matches = []
    for doc in documents:
        haystack = " ".join(
            str(value)
            for value in (
                doc.get("doc_id"),
                doc.get("id"),
                doc.get("title"),
                doc.get("content"),
                doc.get("source"),
                doc.get("category"),
                doc.get("metadata"),
            )
        ).casefold()
        if needle in haystack:
            matches.append(doc)
    return matches


def _retrieval_backend_label(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "none"
    backends = {str(doc.get("retrieval_backend") or "tfidf") for doc in docs}
    return ", ".join(sorted(backends))


def _human_agent_name(agent_id: str) -> str:
    mapping = {
        "support_resolution_agent": "Support Resolution",
        "human_escalation_agent": "Human Escalation",
        "fallback_agent": "Fallback",
    }
    return mapping.get(agent_id, agent_id.replace("_", " ").title())


def _snippet(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."


__all__ = ["main"]
