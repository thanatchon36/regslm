"""Content-addressed SQLite cache of teacher outputs (resume-safe)."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from regslm.teacher.prompts import PROMPT_VERSION

DEFAULT_DIR = Path("data/local/teacher_cache")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, unit_id TEXT, variant TEXT, output TEXT, valid INTEGER,
    n_prompt_tokens INTEGER, n_gen_tokens INTEGER, created_at REAL
)"""


def cache_key(
    model: str, revision: str | None, variant: str, system: str, user: str, sampling_json: str
) -> str:
    parts = [model, revision or "", PROMPT_VERSION, variant, system, user, sampling_json]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


@dataclass(frozen=True)
class CacheRow:
    key: str
    unit_id: str
    variant: str
    output: str
    valid: bool
    n_prompt_tokens: int
    n_gen_tokens: int


class TeacherCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)
        self._db.commit()

    @classmethod
    def for_model(cls, model_slug: str, root: Path = DEFAULT_DIR) -> TeacherCache:
        return cls(root / f"{model_slug}.sqlite")

    def get(self, key: str) -> CacheRow | None:
        r = self._db.execute(
            "SELECT key, unit_id, variant, output, valid, n_prompt_tokens, n_gen_tokens "
            "FROM cache WHERE key=?",
            (key,),
        ).fetchone()
        return None if r is None else CacheRow(r[0], r[1], r[2], r[3], bool(r[4]), r[5], r[6])

    def put_many(self, rows: list[CacheRow]) -> None:
        """Insert rows in one transaction (one commit per batch)."""
        now = time.time()
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        r.key,
                        r.unit_id,
                        r.variant,
                        r.output,
                        int(r.valid),
                        r.n_prompt_tokens,
                        r.n_gen_tokens,
                        now,
                    )
                    for r in rows
                ],
            )

    def close(self) -> None:
        self._db.close()
