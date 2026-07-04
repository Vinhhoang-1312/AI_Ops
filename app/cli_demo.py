# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Command-line demonstration of the Agentic RAG-based IT Support system.

The CLI supports both one-shot execution and an interactive session with
short-term chat memory. In interactive mode the last few exchanges are included
in the rendered prompt for the Support Resolution or Human Escalation Agent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.env import load_project_env
from agent_workflow.src.graph import SupportSession, run_workflow
from observability.langfuse_observer import flush_langfuse
from rag_pipeline.src.retriever import build_retriever
from router_agent.src.model_store import NLI_ROUTER_MODEL_DIR, download_router_model
from router_agent.src.router import Router


def _print_state(state, show_prompt: bool = False) -> None:
    rr = state.router_result
    if rr is None:
        print("An error occurred during routing.")
        return

    print("\n=== Router Decision ===")
    print(f"Next agent: {rr.next_agent}")
    print(f"Confidence: {rr.confidence:.2f}")
    print(f"Reason: {rr.reason}")
    if rr.signals:
        print("Signals:")
        for signal in rr.signals:
            print(f"  - {signal}")

    if state.retrieved_docs:
        print("\n=== Retrieved TechQA Documents ===")
        for doc in state.retrieved_docs:
            backend = doc.get("retrieval_backend", "tfidf")
            title = doc.get("title") or doc.get("doc_id") or "TechQA document"
            score = doc.get("score", 0.0)
            print(f"- {title} (score={score:.2f}, backend={backend})")
    else:
        print("\nNo documents were retrieved.")

    print("\n=== Final Response ===")
    print(state.final_response or "No response available.")

    if show_prompt:
        print("\n=== Rendered Prompt ===")
        print(state.rendered_prompt or "No prompt available.")


def main() -> None:
    load_project_env(PROJECT_ROOT)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Agentic RAG-based IT Support demo")
    parser.add_argument(
        "--ticket",
        type=str,
        default=None,
        help="Provide the ticket text as a command-line argument instead of interactive input",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive multi-turn session with short-term memory",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=10,
        help="Maximum number of chat messages to keep in short-term memory",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the rendered prompt template used by the selected agent",
    )
    parser.add_argument(
        "--router-backend",
        choices=("auto", "openvino", "nli", "heuristic"),
        default=None,
        help="Router backend. `auto` tries OpenVINO IR, then safetensors NLI, then heuristics.",
    )
    parser.add_argument(
        "--retriever-backend",
        choices=("tfidf", "auto", "weaviate"),
        default=None,
        help="Retrieval backend. `weaviate` uses Ollama embeddings and local Weaviate; `auto` falls back to TF-IDF.",
    )
    parser.add_argument(
        "--download-router-model",
        action="store_true",
        help=f"Download cross-encoder/nli-MiniLM2-L6-H768 into {NLI_ROUTER_MODEL_DIR} and exit.",
    )
    args = parser.parse_args()

    if args.download_router_model:
        path = download_router_model()
        print(f"Router model ready at {path}")
        flush_langfuse()
        return

    if args.ticket and not args.interactive:
        retriever = build_retriever(backend=args.retriever_backend)
        router = Router(backend=args.router_backend)
        state = run_workflow(
            args.ticket,
            retriever=retriever,
            router=router,
            session_id=f"cli-session-{uuid4()}",
        )
        _print_state(state, show_prompt=args.show_prompt)
        flush_langfuse()
        return

    session = SupportSession(
        max_messages=args.memory_size,
        retriever_backend=args.retriever_backend,
        router_backend=args.router_backend,
    )
    if args.ticket:
        state = session.ask(args.ticket)
        _print_state(state, show_prompt=args.show_prompt)
        flush_langfuse()

    print("\nInteractive support session. Type 'exit' to quit or 'clear' to clear memory.")
    while True:
        try:
            ticket_text = input("\nTicket> ").strip()
        except KeyboardInterrupt:
            print("\nAborted.")
            flush_langfuse()
            return

        if not ticket_text:
            continue
        if ticket_text.lower() in {"exit", "quit"}:
            flush_langfuse()
            return
        if ticket_text.lower() == "clear":
            session.clear_memory()
            print("Short-term memory cleared.")
            continue

        state = session.ask(ticket_text)
        _print_state(state, show_prompt=args.show_prompt)
        print(f"\nMemory messages retained: {len(session.memory)}")
        flush_langfuse()


if __name__ == "__main__":
    try:
        main()
    finally:
        flush_langfuse()
