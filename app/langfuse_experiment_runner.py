# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Run the IT support workflow against a Langfuse dataset.

This command is the client-side half of Phase 4 evaluation. Langfuse owns the
dataset and server-side evaluators; this script executes the local app for each
dataset item and stores the outputs as a Langfuse experiment run.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_workflow.src.graph import run_workflow
from app.env import load_project_env
from rag_pipeline.src.retriever import build_retriever
from router_agent.src.router import Router


DEFAULT_DATASET_NAME = "it-support/end-to-end"
DEFAULT_EXPERIMENT_NAME = "it-support/end-to-end"


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def extract_ticket_text(item: Any) -> str:
    """Extract ticket text from supported Langfuse dataset item shapes."""
    item_input = _item_value(item, "input")
    if isinstance(item_input, str):
        ticket_text = item_input.strip()
    elif isinstance(item_input, dict):
        ticket_text = str(
            item_input.get("ticket_text")
            or item_input.get("ticket")
            or item_input.get("query")
            or item_input.get("input")
            or ""
        ).strip()
    else:
        ticket_text = ""

    if not ticket_text:
        raise ValueError(
            "Dataset item input must be a string or contain one of: "
            "ticket_text, ticket, query, input."
        )
    return ticket_text


def _compact_doc(doc: dict[str, Any], *, max_chars: int = 500) -> dict[str, Any]:
    content = " ".join(str(doc.get("content") or "").split())
    if len(content) > max_chars:
        content = content[:max_chars].rsplit(" ", 1)[0] + "..."
    return {
        "doc_id": doc.get("doc_id") or doc.get("id"),
        "title": doc.get("title"),
        "score": doc.get("score"),
        "source": doc.get("source"),
        "category": doc.get("category"),
        "snippet": content,
    }


def workflow_output(state: Any) -> dict[str, Any]:
    router = None
    if state.router_result is not None:
        router = asdict(state.router_result) if is_dataclass(state.router_result) else dict(state.router_result)

    retrieved_docs = [_compact_doc(doc) for doc in state.retrieved_docs or []]
    return {
        "selected_agent": state.selected_agent,
        "router": router,
        "retrieved_docs": retrieved_docs,
        "retrieved_context_count": len(retrieved_docs),
        "rendered_prompt": state.rendered_prompt,
        "final_response": state.final_response,
    }


def _with_experiment_environment(args: argparse.Namespace, task):
    previous_tracing = os.environ.get("IT_SUPPORT_LANGFUSE_ENABLED")
    previous_response_backend = os.environ.get("IT_SUPPORT_RESPONSE_BACKEND")
    if not args.include_workflow_traces:
        os.environ["IT_SUPPORT_LANGFUSE_ENABLED"] = "false"
    if args.response_backend:
        os.environ["IT_SUPPORT_RESPONSE_BACKEND"] = args.response_backend
    try:
        return task()
    finally:
        if previous_tracing is None:
            os.environ.pop("IT_SUPPORT_LANGFUSE_ENABLED", None)
        else:
            os.environ["IT_SUPPORT_LANGFUSE_ENABLED"] = previous_tracing

        if previous_response_backend is None:
            os.environ.pop("IT_SUPPORT_RESPONSE_BACKEND", None)
        else:
            os.environ["IT_SUPPORT_RESPONSE_BACKEND"] = previous_response_backend


def _dataset_fetch_error(dataset_name: str, exc: Exception) -> str:
    detail = "dataset was not found" if "notfound" in type(exc).__name__.lower() else str(exc)
    return (
        f"Could not fetch Langfuse dataset '{dataset_name}': {detail}. "
        "Create it in the Langfuse UI first and add dataset items."
    )


def run_experiment(args: argparse.Namespace):
    from langfuse import get_client

    load_project_env(PROJECT_ROOT)
    langfuse = get_client()

    try:
        dataset = langfuse.get_dataset(args.dataset)
    except Exception as exc:
        raise SystemExit(_dataset_fetch_error(args.dataset, exc)) from exc

    items = list(dataset.items)
    if args.limit is not None:
        items = items[: args.limit]

    if not items:
        raise SystemExit(f"Langfuse dataset '{args.dataset}' has no items.")

    router = Router(backend=args.router_backend)
    retriever = build_retriever(backend=args.retriever_backend)
    progress_lock = Lock()
    progress = {"count": 0, "total": len(items)}

    def task(*, item, **kwargs):
        ticket_text = extract_ticket_text(item)
        with progress_lock:
            progress["count"] += 1
            print(f"[{progress['count']}/{progress['total']}] {ticket_text[:96]}", flush=True)

        def run_task():
            state = run_workflow(
                ticket_text,
                top_k=args.top_k,
                retriever=retriever,
                router=router,
                session_id=f"experiment-{args.run_name or args.experiment}",
            )
            return workflow_output(state)

        return _with_experiment_environment(args, run_task)

    result = langfuse.run_experiment(
        name=args.experiment,
        run_name=args.run_name,
        description=args.description,
        data=items,
        task=task,
        max_concurrency=args.max_concurrency,
        metadata={
            "app": "agentic-rag-based-it-support",
            "dataset": args.dataset,
            "router_backend": args.router_backend or os.getenv("IT_SUPPORT_ROUTER_BACKEND", "auto"),
            "retriever_backend": args.retriever_backend or os.getenv("IT_SUPPORT_RETRIEVER_BACKEND", "auto"),
            "response_backend": args.response_backend or os.getenv("IT_SUPPORT_RESPONSE_BACKEND", "auto"),
            "top_k": str(args.top_k),
        },
    )
    langfuse.flush()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Langfuse dataset experiment for the IT support app")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="Langfuse dataset name")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME, help="Langfuse experiment name")
    parser.add_argument("--run-name", default=None, help="Optional exact Langfuse dataset run name")
    parser.add_argument(
        "--description",
        default="Run agentic-rag-based-it-support against the Langfuse end-to-end dataset.",
        help="Experiment run description",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve per item")
    parser.add_argument(
        "--router-backend",
        choices=("auto", "openvino", "nli", "heuristic"),
        default=None,
        help="Router backend to use during the experiment",
    )
    parser.add_argument(
        "--retriever-backend",
        choices=("tfidf", "auto", "weaviate"),
        default=None,
        help="Retriever backend to use during the experiment",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent dataset items. Keep 1 for local model/retriever stability.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of dataset items to run. Useful for smoke tests.",
    )
    parser.add_argument(
        "--response-backend",
        choices=("template", "auto", "openrouter"),
        default="template",
        help="Response backend during experiments. Default template keeps local eval runs fast and repeatable.",
    )
    parser.add_argument(
        "--include-workflow-traces",
        action="store_true",
        help="Also emit the app's detailed it-support-ticket traces while the experiment runs.",
    )
    parser.add_argument(
        "--show-items",
        action="store_true",
        help="Print per-item experiment details after the run.",
    )
    return parser


def main() -> None:
    _configure_console()
    args = build_parser().parse_args()
    result = run_experiment(args)
    print(result.format(include_item_results=args.show_items))
    if result.dataset_run_url:
        print(f"\nDataset run URL: {result.dataset_run_url}")


if __name__ == "__main__":
    main()
