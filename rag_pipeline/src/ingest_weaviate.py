# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Index TechQA support documents into local Weaviate using Ollama embeddings.

Example:
    python -m rag_pipeline.src.ingest_weaviate --reset --max-docs 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.src.retriever import load_support_documents, resolve_default_data_path
from rag_pipeline.src.weaviate_retriever import WeaviateRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Index TechQA documents into Weaviate")
    parser.add_argument(
        "--data-path",
        default=None,
        help="TechQA directory or JSON/JSONL/CSV file. Defaults to data/TechQA, then rag_pipeline/data.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=1000,
        help="Maximum number of TechQA records to index. Use 0 for all records.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of objects to send to Weaviate per batch.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the Weaviate collection before indexing.",
    )
    args = parser.parse_args()

    max_docs = None if args.max_docs == 0 else args.max_docs
    data_path = args.data_path or str(resolve_default_data_path())
    documents = load_support_documents(data_path, max_docs=max_docs)

    retriever = WeaviateRetriever()
    if args.reset:
        print(f"Resetting Weaviate collection {retriever.collection} at {retriever.weaviate_url}...")
        retriever.reset_collection()

    print(
        f"Indexing {len(documents)} TechQA documents into {retriever.collection} "
        f"using Ollama model {retriever.embedder.model}..."
    )
    count = retriever.ingest_documents(documents, batch_size=args.batch_size)
    print(f"Indexed {count} documents.")


if __name__ == "__main__":
    main()
