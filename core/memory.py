"""Memória persistente do sistema (SQLite).

Guarda ciclos de pesquisa, hipóteses, experimentos e um índice de conhecimento
para que execuções futuras possam reaproveitar contexto e evitar repetição de
tópicos já estudados.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.utils import ensure_dir, setup_logger, utcnow_iso

logger = setup_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    score REAL,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(topic, kind, content)
);
CREATE INDEX IF NOT EXISTS idx_cycles_topic ON research_cycles(topic);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);
"""


@dataclass
class MemorySystem:
    """Armazenamento persistente em SQLite para o ciclo científico."""

    path: Path = field(default_factory=lambda: Path("outputs/memory.sqlite3"))
    _conn: sqlite3.Connection | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        ensure_dir(self.path.parent)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---------------------------------------------------------------- cycles
    def start_cycle(self, topic: str) -> int:
        cursor = self._conn.execute(
            "INSERT INTO research_cycles (topic, status, created_at) VALUES (?, ?, ?)",
            (topic, "em_andamento", utcnow_iso()),
        )
        self._conn.commit()
        cycle_id = int(cursor.lastrowid)
        logger.info("Ciclo #%s iniciado para o tópico: %s", cycle_id, topic)
        return cycle_id

    def finish_cycle(self, cycle_id: int, status: str, payload: dict[str, Any], score: float | None = None) -> None:
        self._conn.execute(
            "UPDATE research_cycles SET status = ?, finished_at = ?, score = ?, payload = ? WHERE id = ?",
            (status, utcnow_iso(), score, json.dumps(payload, ensure_ascii=False, default=str), cycle_id),
        )
        self._conn.commit()

    def get_cycle(self, cycle_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM research_cycles WHERE id = ?", (cycle_id,)).fetchone()
        return self._row_to_dict(row)

    def list_cycles(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM research_cycles ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        if data.get("payload"):
            try:
                data["payload"] = json.loads(data["payload"])
            except json.JSONDecodeError:
                pass
        return data

    # ------------------------------------------------------------- knowledge
    def remember(self, topic: str, kind: str, content: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO knowledge (topic, kind, content, created_at) VALUES (?, ?, ?, ?)",
            (topic, kind, content, utcnow_iso()),
        )
        self._conn.commit()

    def recall(self, topic: str, kind: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if kind:
            rows = self._conn.execute(
                "SELECT * FROM knowledge WHERE topic = ? AND kind = ? ORDER BY id DESC LIMIT ?",
                (topic, kind, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM knowledge WHERE topic = ? ORDER BY id DESC LIMIT ?",
                (topic, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def known_topics(self, limit: int = 50) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT topic FROM research_cycles ORDER BY topic LIMIT ?", (limit,)
        ).fetchall()
        return [row["topic"] for row in rows]

    def is_duplicate_topic(self, topic: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM research_cycles WHERE lower(topic) = lower(?) AND status = 'concluido' LIMIT 1",
            (topic.strip(),),
        ).fetchone()
        return row is not None

    def stats(self) -> dict[str, Any]:
        cycles = self._conn.execute("SELECT COUNT(*) AS c FROM research_cycles").fetchone()["c"]
        done = self._conn.execute("SELECT COUNT(*) AS c FROM research_cycles WHERE status = 'concluido'").fetchone()["c"]
        facts = self._conn.execute("SELECT COUNT(*) AS c FROM knowledge").fetchone()["c"]
        return {"ciclos": cycles, "concluidos": done, "conhecimento": facts}

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
