"""SQLite persistence for realtime facial-expression recognition snapshots."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import HISTORY_DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS recognition_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    label TEXT,
    confidence REAL NOT NULL,
    face_count INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    device TEXT NOT NULL,
    top_emotions_json TEXT NOT NULL,
    faces_json TEXT NOT NULL,
    probabilities_json TEXT NOT NULL
);
"""


def init_history_db(db_path: str | Path = HISTORY_DB_PATH) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(SCHEMA)
        conn.commit()
    return path


def save_state_snapshot(state: Any, source: str = "manual", db_path: str | Path = HISTORY_DB_PATH) -> int:
    """Persist the current realtime state and return the inserted row id."""
    path = init_history_db(db_path)
    top_emotions = _top_emotions(state)
    faces = _faces(state)

    with closing(sqlite3.connect(path)) as conn:
        cursor = conn.execute(
            """
            INSERT INTO recognition_snapshots (
                created_at, source, status, label, confidence, face_count, sample_count,
                latency_ms, device, top_emotions_json, faces_json, probabilities_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                source,
                str(getattr(state, "status", "")),
                getattr(state, "label", None),
                float(getattr(state, "confidence", 0.0)),
                len(faces),
                int(getattr(state, "sample_count", 0)),
                float(getattr(state, "latency_ms", 0.0)),
                str(getattr(state, "device", "")),
                json.dumps(top_emotions, ensure_ascii=True),
                json.dumps(faces, ensure_ascii=True),
                json.dumps(getattr(state, "probabilities", {}) or {}, ensure_ascii=True),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def fetch_recent_snapshots(limit: int = 12, db_path: str | Path = HISTORY_DB_PATH) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return []

    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, source, status, label, confidence, face_count, sample_count,
                   latency_ms, device, top_emotions_json, faces_json
            FROM recognition_snapshots
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def _top_emotions(state: Any, limit: int = 2) -> list[tuple[str, float]]:
    top_k = getattr(state, "top_k", None) or []
    if top_k:
        return [(str(label), float(score)) for label, score in top_k[:limit]]

    label = getattr(state, "label", None)
    if label:
        return [(str(label), float(getattr(state, "confidence", 0.0)))]
    return []


def _faces(state: Any) -> list[dict[str, Any]]:
    faces = getattr(state, "faces", None) or []
    if not faces:
        return []

    items = []
    for idx, face in enumerate(faces, start=1):
        region = getattr(face, "face_region", None)
        items.append(
            {
                "rank": idx,
                "label": str(getattr(face, "label", "unknown")),
                "confidence": float(getattr(face, "confidence", 0.0)),
                "top_k": [(str(label), float(score)) for label, score in (getattr(face, "top_k", None) or [])],
                "face_region": {
                    "x": int(getattr(region, "x", 0)),
                    "y": int(getattr(region, "y", 0)),
                    "w": int(getattr(region, "w", 0)),
                    "h": int(getattr(region, "h", 0)),
                    "detected": bool(getattr(region, "detected", False)),
                },
            }
        )
    return items


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["top_emotions"] = json.loads(item.pop("top_emotions_json"))
    item["faces"] = json.loads(item.pop("faces_json"))
    return item
