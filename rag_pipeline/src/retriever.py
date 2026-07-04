# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Simple TF-IDF based retriever for support documents.

The default corpus is the TechQA data stored under ``data/TechQA`` at the
project root. The loader also accepts JSONL, JSON and CSV files so tests and
future converted corpora can use the same retrieval path. CSV ticket datasets
are loaded as whole records; no chunking is applied by this module.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TECHQA_PATH = PROJECT_ROOT / "data" / "TechQA"
FALLBACK_SAMPLE_PATH = PROJECT_ROOT / "rag_pipeline" / "data"


def resolve_default_data_path() -> Path:
    """Return the TechQA RAG corpus path.

    The preferred path is ``data/TechQA`` at the project root. For compact
    demos or tests, the function falls back to ``rag_pipeline/data`` when the
    root TechQA directory is not present.
    """
    if DEFAULT_TECHQA_PATH.exists():
        return DEFAULT_TECHQA_PATH
    return FALLBACK_SAMPLE_PATH


def _candidate_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    supported_suffixes = {".json", ".jsonl", ".csv"}
    files = [file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in supported_suffixes]
    return sorted(files, key=lambda item: ("techqa" not in item.name.lower(), item.name.lower()))


def _iter_json_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        for record in payload:
            if isinstance(record, dict):
                yield record
        return

    if not isinstance(payload, dict):
        return

    for collection_key in ("records", "data", "documents", "items"):
        collection = payload.get(collection_key)
        if isinstance(collection, list):
            for record in collection:
                if isinstance(record, dict):
                    yield record
            return

    if all(isinstance(value, dict) for value in payload.values()):
        for key, record in payload.items():
            normalized_record = dict(record)
            normalized_record.setdefault("id", key)
            yield normalized_record
        return

    yield payload


def _iter_jsonl_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, dict):
                yield record


def _iter_csv_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row_index, record in enumerate(csv.DictReader(file)):
            normalized_record = dict(record)
            normalized_record.setdefault("_source_file", path.stem)
            normalized_record.setdefault("_source_row", row_index)
            yield normalized_record


def _raw_records(path: Path) -> Iterable[dict]:
    for file_path in _candidate_files(path):
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            yield from _iter_json_records(file_path)
        elif suffix == ".jsonl":
            yield from _iter_jsonl_records(file_path)
        elif suffix == ".csv":
            yield from _iter_csv_records(file_path)


def _first_text(record: dict, *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            return value
    return ""


def _short_title(text: str, max_chars: int = 96) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit(" ", 1)[0] + "..."


def _normalize_record(record: dict, index: int) -> Optional[dict]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    source_row = record.get("_source_row")

    doc_id = _first_text(record, "doc_id", "id", "document_id", "uid")
    if not doc_id:
        doc_id = _first_text(metadata, "sourceDocumentId", "id")
    if not doc_id and _first_text(record, "Document") and isinstance(source_row, int):
        doc_id = f"ticket_{source_row:06d}"
    if not doc_id:
        doc_id = f"techqa_{index:06d}"

    title = _first_text(record, "title", "question", "subject", "heading")
    question = _first_text(record, "question", "query", "problem")
    answer = _first_text(record, "answer", "solution", "resolution")
    body = _first_text(record, "text", "content", "body", "document", "Document")

    if question and answer:
        content = f"Question: {question}\nAnswer: {answer}"
    elif question and body:
        content = f"Question: {question}\n{body}"
    else:
        content = body

    if not content:
        content = title
    if not content.strip():
        return None
    if not title:
        title = question or _short_title(content) or _first_text(record, "Topic_group", "category", "topic", "label") or doc_id

    category = _first_text(record, "category", "topic", "label", "Topic_group") or "Technical QA"
    source = _first_text(record, "source", "dataset", "_source_file") or "TechQA"

    document = {
        "doc_id": doc_id,
        "id": doc_id,
        "title": title,
        "content": content,
        "source": source,
        "category": category,
    }
    if isinstance(source_row, int):
        document["source_row"] = source_row
    if metadata:
        document["metadata"] = metadata
    return document


def load_support_documents(data_path: str | Path | None = None, max_docs: Optional[int] = 1000) -> List[dict]:
    """Load and normalize support documents without splitting records into chunks."""
    path = Path(data_path) if data_path is not None else resolve_default_data_path()
    documents: List[dict] = []

    for index, record in enumerate(_raw_records(path)):
        if max_docs is not None and len(documents) >= max_docs:
            break
        document = _normalize_record(record, index)
        if document is not None:
            documents.append(document)

    if not documents:
        raise ValueError(f"No support documents loaded from {path}")
    return documents


class Retriever:
    """Retrieve relevant support documents using TF-IDF cosine similarity."""

    def __init__(self, data_path: Optional[str] = None, max_docs: int = 1000) -> None:
        """Initialize the retriever.

        Args:
            data_path: Path to a TechQA directory or a JSON/JSONL/CSV document file.
                If ``None``, ``data/TechQA`` is used when present.
            max_docs: Maximum number of documents to load.
        """
        self.data_path = str(Path(data_path) if data_path is not None else resolve_default_data_path())
        self.max_docs = max_docs
        self._load_documents()
        self._build_vector_store()

    def _load_documents(self) -> None:
        """Load normalized support documents from the configured corpus."""
        self.documents = load_support_documents(self.data_path, self.max_docs)

    def _build_vector_store(self) -> None:
        """Build the TF-IDF matrix from loaded documents."""
        corpus = [doc["title"] + "\n" + doc["content"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[dict]:
        """Retrieve the top_k documents most relevant to the query.

        Args:
            query: The query string, usually the ticket description.
            top_k: Number of documents to return.
            category_filter: If provided, documents matching this category are
                considered first. If the TechQA corpus has no matching category,
                retrieval falls back to the full corpus.

        Returns:
            A list of document dictionaries sorted by similarity. Each document
            contains normalized fields and a ``score`` field.
        """
        fallback_used = False
        if category_filter:
            requested_category = category_filter.casefold()
            indices = [
                i
                for i, doc in enumerate(self.documents)
                if str(doc.get("category", "")).casefold() == requested_category
            ]
            if not indices:
                indices = list(range(len(self.documents)))
                fallback_used = True
        else:
            indices = list(range(len(self.documents)))

        sub_matrix = self.doc_vectors[indices]
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, sub_matrix).flatten()
        if sims.size == 0:
            return []

        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc_id = indices[idx]
            doc = dict(self.documents[doc_id])
            doc["score"] = float(sims[idx])
            if category_filter:
                doc["category_filter"] = category_filter
                doc["category_filter_fallback"] = fallback_used
            results.append(doc)
        return results


def build_retriever(
    *,
    backend: Optional[str] = None,
    data_path: Optional[str] = None,
    max_docs: int = 1000,
    weaviate_url: Optional[str] = None,
    collection: Optional[str] = None,
) -> Any:
    """Create the configured retriever backend.

    ``auto`` is the default and tries Weaviate first, then falls back to the
    deterministic local TF-IDF backend when vector services are unavailable.
    ``weaviate`` requires a running Weaviate instance and Ollama embedding
    model.
    """
    selected = (backend or os.getenv("IT_SUPPORT_RETRIEVER_BACKEND") or "auto").strip().lower()
    if selected == "tfidf":
        return Retriever(data_path=data_path, max_docs=max_docs)

    from .weaviate_retriever import HybridRetriever

    return HybridRetriever(
        backend=selected,
        data_path=data_path,
        max_docs=max_docs,
        weaviate_url=weaviate_url,
        collection=collection,
    )
