# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Build and persist a TF-IDF vector store for TechQA support documents.

The runtime retriever can build vectors on demand. This script precomputes the
same vectorizer, sparse matrix and document metadata from the default TechQA
corpus under ``data/TechQA`` or from a path supplied with ``--data-path``.

Usage:

    python rag_pipeline/src/build_vector_store.py
    python rag_pipeline/src/build_vector_store.py --data-path data/TechQA
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.src.retriever import load_support_documents, resolve_default_data_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TechQA TF-IDF vector store")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(resolve_default_data_path()),
        help="TechQA directory or JSON/JSONL/CSV corpus path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "vector_store"),
        help="Directory where vector store artifacts will be written",
    )
    parser.add_argument("--max-docs", type=int, default=1000, help="Maximum documents to index")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = load_support_documents(args.data_path, max_docs=args.max_docs)
    corpus = [doc["title"] + "\n" + doc["content"] for doc in documents]
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(corpus)

    with (output_dir / "vectorizer.pkl").open("wb") as vf:
        pickle.dump(vectorizer, vf)

    np.savez(
        output_dir / "doc_vectors.npz",
        data=doc_vectors.data,
        indices=doc_vectors.indices,
        indptr=doc_vectors.indptr,
        shape=doc_vectors.shape,
    )

    with (output_dir / "documents.json").open("w", encoding="utf-8") as df:
        json.dump(documents, df, ensure_ascii=False, indent=2)

    print(f"Vector store built with {len(documents)} documents from {args.data_path}.")


if __name__ == "__main__":
    main()
