"""
Durable response storage in a single SQLite file.

The file is the source of truth. The HNSW index lives in RAM and is rebuilt
from this table at startup (see CacheRouter.restore).

Design:
  - Hits never touch the database. Responses are loaded into a dict at open
    and get() reads only the dict. The database is write-through: put()
    writes the row, commits, then updates the dict.
  - Safe across threads. FastAPI runs sync handlers on a thread pool, so one
    connection is shared behind a lock.
  - Vectors only mean something in the embedding model that produced them,
    so the file records the model and refuses to open under a different one.
"""

import sqlite3
import threading
import time

import numpy as np

from cache_store import CacheStore


class EmbedderMismatch(RuntimeError):
    """The database was built with a different embedding model."""


class SqliteCacheStore(CacheStore):
    def __init__(self, path: str, embedder_id: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()

        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                " id TEXT PRIMARY KEY,"
                " response TEXT NOT NULL,"
                " vector BLOB,"
                " source TEXT,"
                " created_at REAL NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'embedder'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('embedder', ?)", (embedder_id,)
                )
            elif row[0] != embedder_id:
                self._conn.close()
                raise EmbedderMismatch(
                    f"{path} was built with {row[0]!r} but this run uses {embedder_id!r}; "
                    "its vectors are meaningless here. Delete the file or point "
                    "CACHE_DB_PATH somewhere else."
                )
            self._conn.commit()

            self._responses = dict(self._conn.execute("SELECT id, response FROM entries"))

    def put(self, id, response, vector=None, source=None):
        blob = None if vector is None else np.asarray(vector, dtype=np.float32).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries (id, response, vector, source, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (id, response, blob, source, time.time()),
            )
            self._conn.commit()
            self._responses[id] = response

    def get(self, id):
        return self._responses.get(id)

    def all_items(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, vector, response FROM entries WHERE vector IS NOT NULL ORDER BY rowid"
            ).fetchall()
        for id, blob, response in rows:
            yield id, np.frombuffer(blob, dtype=np.float32), response

    def __len__(self):
        return len(self._responses)

    def clear(self):
        with self._lock:
            removed = self._conn.execute("DELETE FROM entries").rowcount
            self._conn.commit()
            self._responses.clear()
            return removed

    def close(self):
        with self._lock:
            self._conn.close()
