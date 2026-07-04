# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Weaviate + Ollama retrieval backend for whole support documents."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional
from urllib import error, request

from .retriever import Retriever, load_support_documents, resolve_default_data_path


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "all-minilm:33m"
DEFAULT_WEAVIATE_URL = "http://localhost:8080"
DEFAULT_WEAVIATE_COLLECTION = "TechQADocument"
DEFAULT_HYBRID_ALPHA = 0.5


class ServiceRequestError(RuntimeError):
    """Raised when an external retrieval service cannot satisfy a request."""


def _json_request(method: str, url: str, payload: Optional[dict[str, Any]] = None, timeout: int = 30) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ServiceRequestError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise ServiceRequestError(f"{method} {url} failed: {exc.reason}") from exc

    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


@dataclass
class OllamaEmbedder:
    """Small REST client for Ollama's embedding API."""

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_OLLAMA_EMBED_MODEL
    timeout: int = 60

    def embed(self, text: str) -> List[float]:
        payload = {"model": self.model, "input": text}
        response = _json_request("POST", f"{self.base_url.rstrip('/')}/api/embed", payload, timeout=self.timeout)
        vector = parse_ollama_embedding_response(response)
        if not vector:
            raise ServiceRequestError("Ollama returned an empty embedding vector")
        return vector


def parse_ollama_embedding_response(response: dict[str, Any]) -> List[float]:
    """Return the first embedding vector from supported Ollama response shapes."""
    if isinstance(response.get("embeddings"), list) and response["embeddings"]:
        candidate = response["embeddings"][0]
    elif isinstance(response.get("embedding"), list):
        candidate = response["embedding"]
    else:
        raise ServiceRequestError("Ollama embedding response did not include `embeddings` or `embedding`")

    if not isinstance(candidate, list):
        raise ServiceRequestError("Ollama embedding payload is not a vector")
    return [float(value) for value in candidate]


def _resolve_hybrid_alpha(value: Optional[float]) -> float:
    raw_value = value
    if raw_value is None:
        env_value = os.getenv("WEAVIATE_HYBRID_ALPHA")
        raw_value = float(env_value) if env_value else DEFAULT_HYBRID_ALPHA
    return min(1.0, max(0.0, float(raw_value)))


def _graphql_input(value: Any) -> str:
    if isinstance(value, dict):
        pairs = [f"{key}: {_graphql_input(item)}" for key, item in value.items()]
        return "{" + ", ".join(pairs) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_graphql_input(item) for item in value) + "]"
    return json.dumps(value)


def _coerce_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


class WeaviateRetriever:
    """Retrieve TechQA documents from a Weaviate collection with self-provided vectors."""

    def __init__(
        self,
        *,
        weaviate_url: Optional[str] = None,
        collection: Optional[str] = None,
        embedder: Optional[OllamaEmbedder] = None,
        hybrid_alpha: Optional[float] = None,
        timeout: int = 30,
    ) -> None:
        self.weaviate_url = (weaviate_url or os.getenv("WEAVIATE_URL") or DEFAULT_WEAVIATE_URL).rstrip("/")
        self.collection = collection or os.getenv("WEAVIATE_COLLECTION") or DEFAULT_WEAVIATE_COLLECTION
        self.hybrid_alpha = _resolve_hybrid_alpha(hybrid_alpha)
        self.embedder = embedder or OllamaEmbedder(
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            model=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL),
        )
        self.timeout = timeout

    def is_ready(self) -> bool:
        try:
            _json_request("GET", f"{self.weaviate_url}/v1/meta", timeout=5)
            return True
        except ServiceRequestError:
            return False

    def ensure_collection(self) -> None:
        try:
            _json_request("GET", f"{self.weaviate_url}/v1/schema/{self.collection}", timeout=self.timeout)
            return
        except ServiceRequestError:
            pass

        schema = {
            "class": self.collection,
            "description": "TechQA support documents indexed with Ollama all-minilm embeddings",
            "vectorizer": "none",
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {"distance": "cosine"},
            "properties": [
                {"name": "doc_id", "dataType": ["text"]},
                {"name": "title", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
                {"name": "source", "dataType": ["text"]},
                {"name": "category", "dataType": ["text"]},
                {"name": "source_row", "dataType": ["int"]},
            ],
        }
        _json_request("POST", f"{self.weaviate_url}/v1/schema", schema, timeout=self.timeout)

    def reset_collection(self) -> None:
        try:
            _json_request("DELETE", f"{self.weaviate_url}/v1/schema/{self.collection}", timeout=self.timeout)
        except ServiceRequestError:
            pass

    def ingest_documents(
        self,
        documents: Iterable[dict[str, Any]],
        *,
        batch_size: int = 32,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> int:
        self.ensure_collection()
        total = 0
        batch: list[dict[str, Any]] = []
        seen = 0
        for doc in documents:
            text = f"{doc.get('title', '')}\n{doc.get('content', '')}".strip()
            if not text:
                continue
            seen += 1
            doc_id = str(doc.get("doc_id") or doc.get("id") or f"techqa_{total:06d}")
            properties = {
                "doc_id": doc_id,
                "title": str(doc.get("title") or doc_id),
                "content": str(doc.get("content") or ""),
                "source": str(doc.get("source") or "TechQA"),
                "category": str(doc.get("category") or "Technical QA"),
            }
            if doc.get("source_row") is not None:
                try:
                    properties["source_row"] = int(doc["source_row"])
                except (TypeError, ValueError):
                    pass
            batch.append(
                {
                    "class": self.collection,
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection}:{doc_id}")),
                    "properties": properties,
                    "vector": self.embedder.embed(text),
                }
            )
            self._emit_progress(
                progress_callback,
                {
                    "stage": "embedding",
                    "doc_id": doc_id,
                    "embedded_documents": seen,
                    "batched_documents": total + len(batch),
                    "imported_documents": total,
                    "batch_size": len(batch),
                    "collection": self.collection,
                    "model": self.embedder.model,
                },
            )
            if len(batch) >= batch_size:
                self._send_batch(batch)
                total += len(batch)
                self._emit_progress(
                    progress_callback,
                    {
                        "stage": "import",
                        "doc_id": doc_id,
                        "embedded_documents": seen,
                        "batched_documents": total,
                        "imported_documents": total,
                        "last_batch_size": len(batch),
                        "collection": self.collection,
                        "model": self.embedder.model,
                    },
                )
                batch = []

        if batch:
            self._send_batch(batch)
            total += len(batch)
            self._emit_progress(
                progress_callback,
                {
                    "stage": "import",
                    "embedded_documents": seen,
                    "batched_documents": total,
                    "imported_documents": total,
                    "last_batch_size": len(batch),
                    "collection": self.collection,
                    "model": self.embedder.model,
                },
            )
        self._emit_progress(
            progress_callback,
            {
                "stage": "complete",
                "embedded_documents": seen,
                "batched_documents": total,
                "imported_documents": total,
                "collection": self.collection,
                "model": self.embedder.model,
            },
        )
        return total

    def _emit_progress(
        self,
        callback: Optional[Callable[[dict[str, Any]], None]],
        payload: dict[str, Any],
    ) -> None:
        if callback is None:
            return
        callback(payload)

    def _send_batch(self, objects: list[dict[str, Any]]) -> None:
        response = _json_request("POST", f"{self.weaviate_url}/v1/batch/objects", {"objects": objects}, timeout=self.timeout)
        if isinstance(response, list):
            items = response
        elif isinstance(response, dict):
            items = response.get("objects", [])
        else:
            raise ServiceRequestError(f"Weaviate batch ingest returned an unexpected payload: {response!r}")

        for item in items:
            result = item.get("result") if isinstance(item, dict) else None
            errors = result.get("errors") if isinstance(result, dict) else None
            if errors:
                raise ServiceRequestError(f"Weaviate batch ingest returned errors: {errors}")

    def list_documents(self, limit: int = 1000) -> List[dict]:
        graph_query = f"""
        {{
          Get {{
            {self.collection}(limit: {int(limit)}) {{
              doc_id
              title
              content
              source
              category
              source_row
              _additional {{ id }}
            }}
          }}
        }}
        """
        response = _json_request("POST", f"{self.weaviate_url}/v1/graphql", {"query": graph_query}, timeout=self.timeout)
        if response.get("errors"):
            raise ServiceRequestError(f"Weaviate document load failed: {response['errors']}")

        rows = (((response.get("data") or {}).get("Get") or {}).get(self.collection) or [])
        documents: list[dict] = []
        for row in rows:
            doc = dict(row)
            additional = doc.pop("_additional", {}) or {}
            doc["id"] = doc.get("doc_id") or additional.get("id")
            doc["retrieval_backend"] = "weaviate"
            documents.append(doc)
        return documents

    def retrieve(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[dict]:
        vector = self.embedder.embed(query)
        hybrid = {
            "query": query,
            "vector": vector,
            "alpha": self.hybrid_alpha,
            "properties": ["doc_id", "title", "content", "category"],
        }
        arguments = f"hybrid: {_graphql_input(hybrid)}, limit: {int(top_k)}"
        if category_filter:
            arguments += f', where: {{path: ["category"], operator: Equal, valueText: {json.dumps(category_filter)}}}'
        graph_query = f"""
        {{
          Get {{
            {self.collection}({arguments}) {{
              doc_id
              title
              content
              source
              category
              source_row
              _additional {{ score explainScore id }}
            }}
          }}
        }}
        """
        response = _json_request("POST", f"{self.weaviate_url}/v1/graphql", {"query": graph_query}, timeout=self.timeout)
        if response.get("errors"):
            raise ServiceRequestError(f"Weaviate GraphQL query failed: {response['errors']}")

        rows = (((response.get("data") or {}).get("Get") or {}).get(self.collection) or [])
        results: list[dict] = []
        for row in rows:
            doc = dict(row)
            additional = doc.pop("_additional", {}) or {}
            score = _coerce_score(additional.get("score"))
            doc["score"] = score
            doc["id"] = doc.get("doc_id")
            doc["retrieval_backend"] = "weaviate"
            doc["search_strategy"] = "hybrid"
            doc["hybrid_alpha"] = self.hybrid_alpha
            doc["vector_model"] = self.embedder.model
            if additional.get("explainScore"):
                doc["hybrid_explain_score"] = additional.get("explainScore")
            if category_filter:
                doc["category_filter"] = category_filter
                doc["category_filter_fallback"] = False
            results.append(doc)
        return results


class HybridRetriever:
    """Retriever that can use Weaviate first and fall back to local TF-IDF."""

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        data_path: Optional[str] = None,
        max_docs: int = 1000,
        weaviate_url: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> None:
        self.backend = (backend or os.getenv("IT_SUPPORT_RETRIEVER_BACKEND") or "auto").strip().lower()
        if self.backend not in {"tfidf", "auto", "weaviate"}:
            raise ValueError("Retriever backend must be one of: tfidf, auto, weaviate")

        self.local: Optional[Retriever] = None
        self.documents: List[dict] = []
        self.data_path = "weaviate"
        self._local_data_path = data_path
        self._local_max_docs = max_docs
        self.weaviate_error: Optional[str] = None
        self.weaviate: Optional[WeaviateRetriever] = None
        if self.backend == "tfidf":
            self._ensure_local()
        if self.backend in {"auto", "weaviate"}:
            try:
                candidate = WeaviateRetriever(weaviate_url=weaviate_url, collection=collection)
                if not candidate.is_ready():
                    raise ServiceRequestError(f"Weaviate is not ready at {candidate.weaviate_url}")
                candidate.ensure_collection()
                self.weaviate = candidate
                self.data_path = f"{candidate.weaviate_url}/{candidate.collection}"
                self.documents = candidate.list_documents(limit=max_docs)
            except Exception as exc:
                self.weaviate_error = str(exc)
                if self.backend == "weaviate":
                    raise
                self._ensure_local()

    def _ensure_local(self) -> None:
        if self.local is None:
            self.local = Retriever(data_path=self._local_data_path, max_docs=self._local_max_docs)
            self.documents = self.local.documents
            self.data_path = self.local.data_path

    def retrieve(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[dict]:
        if self.weaviate is not None:
            try:
                docs = self.weaviate.retrieve(query, top_k=top_k, category_filter=category_filter)
                if docs or self.backend == "weaviate":
                    return docs
                self.weaviate_error = "Weaviate returned no documents"
            except Exception as exc:
                self.weaviate_error = str(exc)
                if self.backend == "weaviate":
                    raise

        self._ensure_local()
        docs = self.local.retrieve(query, top_k=top_k, category_filter=category_filter)
        for doc in docs:
            doc["retrieval_backend"] = "tfidf"
            if self.weaviate_error:
                doc["retrieval_fallback_reason"] = self.weaviate_error
        return docs


def ingest_default_techqa_to_weaviate(
    *,
    data_path: Optional[str] = None,
    max_docs: Optional[int] = 1000,
    reset: bool = False,
    batch_size: int = 32,
) -> int:
    """Load TechQA records and index them as whole documents in Weaviate."""
    documents = load_support_documents(data_path or resolve_default_data_path(), max_docs=max_docs)
    retriever = WeaviateRetriever()
    if reset:
        retriever.reset_collection()
    return retriever.ingest_documents(documents, batch_size=batch_size)
