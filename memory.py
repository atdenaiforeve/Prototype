"""Prototype long-term memory engine.

Persistent memory for knowledge, conversations, experiences, training events,
tasks, goals, corrections, skills, and relationships.

The neural model's weights are NOT stored here. This database stores explicit
information Prototype can retrieve and reason over.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

DB_PATH = Path(__file__).with_name("memory.db")
SCHEMA_VERSION = 2

MEMORY_TYPES = {
    "fact", "conversation", "experience", "training", "task", "goal",
    "preference", "concept", "correction", "skill", "event", "note",
    "self_update_intention",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "this", "that", "it", "as",
    "be", "by", "from", "at", "we", "you", "i", "my", "your", "our",
}


def _now() -> float:
    return time.time()


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


class MemoryEngine:
    """SQLite-backed long-term memory with relevance, confidence and links."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        return con

    def _initialize(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'general',
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_used_at REAL,
                    source TEXT,
                    session_id TEXT,
                    tags TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    archived INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS memory_links (
                    memory_id INTEGER NOT NULL,
                    related_memory_id INTEGER NOT NULL,
                    relationship TEXT NOT NULL DEFAULT 'related',
                    strength REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (memory_id, related_memory_id, relationship),
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY(related_memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    old_content TEXT,
                    new_content TEXT,
                    reason TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS training_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset TEXT,
                    lesson TEXT,
                    examples INTEGER NOT NULL DEFAULT 0,
                    tokens INTEGER NOT NULL DEFAULT 0,
                    epoch INTEGER,
                    loss REAL,
                    checkpoint TEXT,
                    notes TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(archived);
                CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
                CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
                CREATE INDEX IF NOT EXISTS idx_links_related ON memory_links(related_memory_id);
                """
            )
            con.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def remember(
        self,
        content: str,
        memory_type: str = "note",
        importance: float = 0.5,
        confidence: float = 0.5,
        source: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        deduplicate: bool = True,
    ) -> int:
        """Store a memory and return its ID.

        If an active memory with identical normalized content exists,
        reinforce it instead of creating a duplicate by default.
        """
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")
        if memory_type not in MEMORY_TYPES:
            memory_type = "note"

        importance = max(0.0, min(1.0, float(importance)))
        confidence = max(0.0, min(1.0, float(confidence)))
        now = _now()
        tag_list = sorted({str(x).strip().lower() for x in (tags or []) if str(x).strip()})
        meta = metadata or {}

        with self._connect() as con:
            if deduplicate:
                row = con.execute(
                    """
                    SELECT id FROM memories
                    WHERE lower(trim(content)) = lower(trim(?)) AND archived = 0
                    LIMIT 1
                    """,
                    (content,),
                ).fetchone()
                if row:
                    con.execute(
                        """
                        UPDATE memories
                        SET importance = MAX(importance, ?),
                            confidence = MAX(confidence, ?),
                            access_count = access_count + 1,
                            updated_at = ?,
                            last_used_at = ?
                        WHERE id = ?
                        """,
                        (importance, confidence, now, now, row["id"]),
                    )
                    return int(row["id"])

            cur = con.execute(
                """
                INSERT INTO memories
                (content, memory_type, importance, confidence, created_at,
                 updated_at, source, session_id, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content, memory_type, importance, confidence, now, now,
                    source, session_id, json.dumps(tag_list), json.dumps(meta),
                ),
            )
            memory_id = int(cur.lastrowid)
            con.execute(
                """
                INSERT INTO memory_history
                (memory_id, action, new_content, reason, created_at)
                VALUES (?, 'created', ?, 'new memory', ?)
                """,
                (memory_id, content, now),
            )
            return memory_id

    def remember_self_update_intention(
        self,
        content: str,
        *,
        reason: Optional[str] = None,
        importance: float = 0.95,
        confidence: float = 0.9,
        freshness_seconds: int = 24 * 60 * 60,
    ) -> int:
        """Store the current self-update intention with a 24-hour freshness window."""
        now = _now()
        freshness_seconds = max(1, int(freshness_seconds))
        metadata = {
            "status": "planned",
            "expires_at": now + freshness_seconds,
            "reason": reason or "",
        }
        return self.remember(
            content,
            memory_type="self_update_intention",
            importance=importance,
            confidence=confidence,
            source="self_model",
            tags=["self-update", "intention"],
            metadata=metadata,
            deduplicate=False,
        )

    def latest_self_update_intention(
        self,
        *,
        freshness_seconds: int = 24 * 60 * 60,
        include_completed: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Return the newest fresh self-update intention, or None if there isn't one."""
        now = _now()
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT * FROM memories
                WHERE memory_type = ? AND archived = 0
                ORDER BY created_at DESC, id DESC
                """,
                ("self_update_intention",),
            ).fetchall()

            for row in rows:
                try:
                    metadata = json.loads(row["metadata"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    metadata = {}
                status = str(metadata.get("status", "planned"))
                if not include_completed and status in {"completed", "rejected", "expired"}:
                    continue
                expires_at = float(metadata.get("expires_at", row["created_at"] + freshness_seconds))
                if expires_at <= now:
                    if status not in {"completed", "rejected", "expired"}:
                        metadata["status"] = "expired"
                        con.execute(
                            "UPDATE memories SET metadata = ?, updated_at = ? WHERE id = ?",
                            (json.dumps(metadata), now, row["id"]),
                        )
                    continue

                con.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_used_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                return self._row_to_dict(row)
        return None

    def set_self_update_intention_status(self, memory_id: int, status: str) -> bool:
        """Set the lifecycle status of a self-update intention."""
        allowed = {"planned", "testing", "completed", "rejected", "expired"}
        if status not in allowed:
            raise ValueError(f"Invalid self-update intention status: {status}")
        with self._connect() as con:
            row = con.execute(
                "SELECT metadata FROM memories WHERE id = ? AND memory_type = 'self_update_intention'",
                (int(memory_id),),
            ).fetchone()
            if not row:
                return False
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            metadata["status"] = status
            con.execute(
                "UPDATE memories SET metadata = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata), _now(), int(memory_id)),
            )
            return True

    def recall(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
        min_confidence: float = 0.0,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the most relevant memories using lexical overlap plus trust."""
        query_terms = _tokens(query)
        limit = max(1, int(limit))

        with self._connect() as con:
            sql = "SELECT * FROM memories WHERE confidence >= ?"
            params: list[Any] = [min_confidence]
            if not include_archived:
                sql += " AND archived = 0"
            if memory_type:
                sql += " AND memory_type = ?"
                params.append(memory_type)
            rows = con.execute(sql, params).fetchall()

            scored = []
            for row in rows:
                text = row["content"]
                overlap = len(query_terms & _tokens(text))
                exact_bonus = 2.0 if query.strip().lower() in text.lower() else 0.0
                trust = 0.75 * float(row["confidence"]) + 0.25 * float(row["importance"])
                usage = min(1.0, float(row["access_count"]) / 20.0)
                score = overlap * 3.0 + exact_bonus + trust + usage * 0.25
                if query_terms and overlap == 0 and exact_bonus == 0:
                    continue
                scored.append((score, row))

            scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
            selected = scored[:limit]

            now = _now()
            for _, row in selected:
                con.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1, last_used_at = ?
                    WHERE id = ?
                    """,
                    (now, row["id"]),
                )

            return [self._row_to_dict(row, score=score) for score, row in selected]

    def get(self, memory_id: int) -> Optional[dict[str, Any]]:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM memories WHERE id = ?", (int(memory_id),)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        memory_type: Optional[str] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        reason: str = "memory update",
    ) -> bool:
        with self._connect() as con:
            old = con.execute(
                "SELECT * FROM memories WHERE id = ?", (int(memory_id),)
            ).fetchone()
            if not old:
                return False

            fields, values = [], []
            if content is not None:
                fields.append("content = ?")
                values.append(content.strip())
            if memory_type is not None and memory_type in MEMORY_TYPES:
                fields.append("memory_type = ?")
                values.append(memory_type)
            if importance is not None:
                fields.append("importance = ?")
                values.append(max(0.0, min(1.0, float(importance))))
            if confidence is not None:
                fields.append("confidence = ?")
                values.append(max(0.0, min(1.0, float(confidence))))
            if source is not None:
                fields.append("source = ?")
                values.append(source)

            if not fields:
                return True

            now = _now()
            fields.extend(["updated_at = ?"])
            values.append(now)
            values.append(int(memory_id))
            con.execute(
                f"UPDATE memories SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            con.execute(
                """
                INSERT INTO memory_history
                (memory_id, action, old_content, new_content, reason, created_at)
                VALUES (?, 'updated', ?, ?, ?, ?)
                """,
                (
                    memory_id, old["content"], content if content is not None else old["content"],
                    reason, now,
                ),
            )
            return True

    def reinforce(self, memory_id: int, amount: float = 0.05, reason: str = "reinforced") -> bool:
        with self._connect() as con:
            cur = con.execute(
                """
                UPDATE memories
                SET confidence = MIN(1.0, confidence + ?),
                    importance = MIN(1.0, importance + (? * 0.25)),
                    access_count = access_count + 1,
                    updated_at = ?
                WHERE id = ? AND archived = 0
                """,
                (abs(float(amount)), abs(float(amount)), _now(), int(memory_id)),
            )
            return cur.rowcount > 0

    def weaken(self, memory_id: int, amount: float = 0.05, reason: str = "weakened") -> bool:
        with self._connect() as con:
            cur = con.execute(
                """
                UPDATE memories
                SET confidence = MAX(0.0, confidence - ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (abs(float(amount)), _now(), int(memory_id)),
            )
            if cur.rowcount:
                con.execute(
                    """
                    INSERT INTO memory_history
                    (memory_id, action, reason, created_at)
                    VALUES (?, 'weakened', ?, ?)
                    """,
                    (memory_id, reason, _now()),
                )
            return cur.rowcount > 0

    def correct(self, memory_id: int, corrected_content: str, reason: str = "correction") -> Optional[int]:
        """Preserve the old memory as history and create a corrected memory."""
        old = self.get(memory_id)
        if not old:
            return None

        self.weaken(memory_id, 0.35, reason)
        new_id = self.remember(
            corrected_content,
            memory_type="correction",
            importance=max(0.6, old["importance"]),
            confidence=0.9,
            source=old.get("source"),
            tags=["correction"],
        )
        self.link_memories(new_id, memory_id, "corrects", strength=1.0)
        return new_id

    def archive(self, memory_id: int, reason: str = "archived") -> bool:
        with self._connect() as con:
            cur = con.execute(
                "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
                (_now(), int(memory_id)),
            )
            if cur.rowcount:
                con.execute(
                    """
                    INSERT INTO memory_history
                    (memory_id, action, reason, created_at)
                    VALUES (?, 'archived', ?, ?)
                    """,
                    (memory_id, reason, _now()),
                )
            return cur.rowcount > 0

    def forget(self, memory_id: int) -> bool:
        with self._connect() as con:
            cur = con.execute("DELETE FROM memories WHERE id = ?", (int(memory_id),))
            return cur.rowcount > 0

    def link_memories(
        self,
        memory_id: int,
        related_memory_id: int,
        relationship: str = "related",
        strength: float = 0.5,
    ) -> bool:
        if int(memory_id) == int(related_memory_id):
            return False
        with self._connect() as con:
            a = con.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
            b = con.execute("SELECT id FROM memories WHERE id = ?", (related_memory_id,)).fetchone()
            if not a or not b:
                return False
            con.execute(
                """
                INSERT INTO memory_links
                (memory_id, related_memory_id, relationship, strength, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, related_memory_id, relationship)
                DO UPDATE SET strength = excluded.strength
                """,
                (memory_id, related_memory_id, relationship,
                 max(0.0, min(1.0, float(strength))), _now()),
            )
            return True

    def related_memories(self, memory_id: int, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT m.*, l.relationship, l.strength
                FROM memory_links l
                JOIN memories m ON m.id = l.related_memory_id
                WHERE l.memory_id = ? AND m.archived = 0
                ORDER BY l.strength DESC, m.importance DESC
                LIMIT ?
                """,
                (int(memory_id), max(1, int(limit))),
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def record_training(
        self,
        dataset: str,
        lesson: Optional[str] = None,
        examples: int = 0,
        tokens: int = 0,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        checkpoint: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO training_events
                (dataset, lesson, examples, tokens, epoch, loss, checkpoint, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset, lesson, int(examples), int(tokens), epoch, loss,
                    checkpoint, notes, _now(),
                ),
            )
            return int(cur.lastrowid)

    def training_history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM training_events ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            return [dict(row) for row in rows]

    def history(self, memory_id: int, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT * FROM memory_history
                WHERE memory_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(memory_id), max(1, int(limit))),
            ).fetchall()
            return [dict(row) for row in rows]

    def all_memories(self, limit: int = 100, include_archived: bool = False) -> list[dict[str, Any]]:
        with self._connect() as con:
            sql = "SELECT * FROM memories"
            if not include_archived:
                sql += " WHERE archived = 0"
            sql += " ORDER BY importance DESC, updated_at DESC LIMIT ?"
            rows = con.execute(sql, (max(1, int(limit)),)).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            active = con.execute(
                "SELECT COUNT(*) FROM memories WHERE archived = 0"
            ).fetchone()[0]
            archived = total - active
            training = con.execute("SELECT COUNT(*) FROM training_events").fetchone()[0]
            links = con.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0]
            corrections = con.execute(
                "SELECT COUNT(*) FROM memories WHERE memory_type = 'correction'"
            ).fetchone()[0]
            return {
                "schema_version": SCHEMA_VERSION,
                "total_memories": total,
                "active_memories": active,
                "archived_memories": archived,
                "relationships": links,
                "corrections": corrections,
                "training_events": training,
                "database": str(self.db_path),
            }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, score: Optional[float] = None) -> dict[str, Any]:
        result = dict(row)
        for field in ("tags", "metadata"):
            try:
                result[field] = json.loads(result[field])
            except (TypeError, json.JSONDecodeError):
                pass
        if score is not None:
            result["relevance_score"] = score
        result["archived"] = bool(result.get("archived", False))
        return result


_default_engine: Optional[MemoryEngine] = None


def get_memory() -> MemoryEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = MemoryEngine()
    return _default_engine


# Backwards-compatible convenience functions.
def remember(content: str, memory_type: str = "general", importance: float = 0.5, source: Optional[str] = None, **kwargs: Any) -> int:
    return get_memory().remember(
        content,
        memory_type=memory_type if memory_type in MEMORY_TYPES else "note",
        importance=importance,
        source=source,
        **kwargs,
    )


def remember_self_update_intention(content: str, **kwargs: Any) -> int:
    return get_memory().remember_self_update_intention(content, **kwargs)


def latest_self_update_intention(**kwargs: Any) -> Optional[dict[str, Any]]:
    return get_memory().latest_self_update_intention(**kwargs)


def set_self_update_intention_status(memory_id: int, status: str) -> bool:
    return get_memory().set_self_update_intention_status(memory_id, status)


def recall(query: str, limit: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
    return get_memory().recall(query, limit=limit, **kwargs)


def update_memory(memory_id: int, **kwargs: Any) -> bool:
    return get_memory().update_memory(memory_id, **kwargs)


def archive(memory_id: int) -> bool:
    return get_memory().archive(memory_id)


def forget(memory_id: int) -> bool:
    return get_memory().forget(memory_id)


def link_memories(memory_id: int, related_memory_id: int, relationship: str = "related") -> bool:
    return get_memory().link_memories(memory_id, related_memory_id, relationship)


def related_memories(memory_id: int, limit: int = 20) -> list[dict[str, Any]]:
    return get_memory().related_memories(memory_id, limit)


def all_memories(limit: int = 100, include_archived: bool = False) -> list[dict[str, Any]]:
    return get_memory().all_memories(limit, include_archived)


if __name__ == "__main__":
    memory = get_memory()
    print("Prototype Memory Engine")
    print(json.dumps(memory.stats(), indent=2))
