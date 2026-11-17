"""Durable renderer receipts. No model/framework imports; files live with outputs.

An operation names one immutable request. A missing downstream acknowledgement
is not permission to submit it again. Adapters reconcile or retain that unknown
state. Completed receipts and their output files are retained, never age-pruned.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class OperationConflict(ValueError):
    pass


def operation_id(value: Any) -> str:
    """Canonical UUIDs only, also safe as URL/path components."""
    return str(uuid.UUID(str(value))) if value else str(uuid.uuid4())


def request_sha256(payload: dict[str, Any]) -> str:
    request = {key: value for key, value in payload.items() if key != "operation_id"}
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encode(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__receipt_bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported receipt value: {type(value).__name__}")


def _decode(value: dict[str, Any]) -> Any:
    if set(value) == {"__receipt_bytes__"}:
        return base64.b64decode(value["__receipt_bytes__"], validate=True)
    return value


def _json(value: Any) -> str:
    return json.dumps(value, default=_encode, sort_keys=True, allow_nan=False)


class Receipts:
    def __init__(self, directory: Path, *, exclusive: bool = False):
        directory.mkdir(parents=True, exist_ok=True)
        self.owner = None
        if exclusive:
            self.owner = (directory / "owner.lock").open("a")
            try:
                fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self.owner.close()
                raise RuntimeError("Another renderer owns this receipt journal") from None
        self.path = directory / "receipts.sqlite3"
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY, operation_id TEXT NOT NULL UNIQUE,
                request_sha256 TEXT NOT NULL, frozen TEXT NOT NULL, state TEXT NOT NULL)""")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA synchronous=FULL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _job(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        job = json.loads(row[3], object_hook=_decode)
        job.update(json.loads(row[4], object_hook=_decode))
        job.update(id=row[0], job_id=row[0], operation_id=row[1], request_sha256=row[2])
        if job.get("status") in {"done", "error", "cancelled", "interrupted", "abandoned"}:
            job["payload"] = {}
        job.setdefault("payload", {})
        return job

    def find(self, identity: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
        identity = operation_id(identity)
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM receipts WHERE operation_id=?", (identity,)).fetchone()
        job = self._job(row)
        if job and payload is not None and job["request_sha256"] != request_sha256(payload):
            raise OperationConflict("operation_id already belongs to a different immutable request")
        return job

    def all(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [self._job(row) for row in conn.execute("SELECT * FROM receipts ORDER BY id")]

    def create(self, payload: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
        identity = operation_id(payload.get("operation_id"))
        digest = request_sha256(payload)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = self._job(conn.execute("SELECT * FROM receipts WHERE operation_id=?", (identity,)).fetchone())
            if previous:
                if previous["request_sha256"] != digest:
                    raise OperationConflict("operation_id already belongs to a different immutable request")
                return previous
            # Also avoids reusing the small process-local IDs issued before this
            # journal existed. Integers remain exact in JavaScript (< 2**53).
            maximum = conn.execute("SELECT COALESCE(MAX(id), 0) FROM receipts").fetchone()[0]
            job_id = max(time.time_ns() // 1000, maximum + 1)
            conn.execute("INSERT INTO receipts VALUES (?, ?, ?, ?, ?)",
                         (job_id, identity, digest, _json(frozen), "{}"))
        return {**frozen, "id": job_id, "job_id": job_id,
                "operation_id": identity, "request_sha256": digest}

    def close(self) -> None:
        if self.owner:
            self.owner.close()

    def save(self, job: dict[str, Any]) -> None:
        # Large immutable source media is stored once, not rewritten at every
        # checkpoint. The frozen payload survives in-memory cleanup after use.
        state = {key: value for key, value in job.items() if key != "payload"}
        with self.connect() as conn:
            cursor = conn.execute("UPDATE receipts SET state=? WHERE id=? AND operation_id=?",
                                  (_json(state), job["id"], job["operation_id"]))
            if cursor.rowcount != 1:
                raise RuntimeError("renderer receipt disappeared; refusing untracked work")
            if job.get("status") in {"done", "error", "cancelled", "interrupted", "abandoned"}:
                conn.execute("UPDATE receipts SET frozen=json_remove(frozen, '$.payload') WHERE id=?", (job["id"],))

    def acknowledge(self, identity: str, digest: str) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM receipts WHERE operation_id=?", (operation_id(identity),)).fetchone()
            job = self._job(row)
            if not job or job["request_sha256"] != digest or job.get("status") != "done":
                raise OperationConflict("Acknowledgement needs the matching completed operation digest")
            state = json.loads(row[4])
            state["acknowledged_at"] = state.get("acknowledged_at") or time.time()
            conn.execute("UPDATE receipts SET state=? WHERE id=?", (_json(state), job["id"]))
        return {**job, "acknowledged_at": state["acknowledged_at"]}

    def retained_ids(self) -> set[int]:
        """Outputs cannot be age-pruned until durable storage is acknowledged."""
        with self.connect() as conn:
            return {row[0] for row in conn.execute(
                "SELECT id FROM receipts WHERE json_extract(state, '$.acknowledged_at') IS NULL "
                "AND COALESCE(json_extract(frozen, '$.retain_outputs'), 1)=1")}


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"byte_count": path.stat().st_size, "sha256": digest.hexdigest()}
