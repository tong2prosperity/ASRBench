"""Small SQLite journal. Every completed attempt is committed independently."""

import json
from contextlib import contextmanager
import sqlite3
import time
import uuid
from pathlib import Path


def uid(prefix):
    return prefix + "_" + uuid.uuid4().hex[:16]


class Store:
    def __init__(self, path="runtime/asrbench.sqlite3"):
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, state TEXT NOT NULL, created REAL NOT NULL, spec TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, profile_id TEXT NOT NULL, sample_id TEXT NOT NULL, body TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS attempts_run ON attempts(run_id, profile_id, sample_id);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, created REAL NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover(self):
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET state='interrupted' WHERE state IN ('running','queued','stopping')"
            )

    def add_dataset(self, body):
        did = uid("ds")
        with self.connect() as db:
            db.execute(
                "INSERT INTO datasets VALUES (?,?)",
                (did, json.dumps(body, ensure_ascii=False)),
            )
        return {"id": did, **body}

    def dataset(self, did):
        with self.connect() as db:
            row = db.execute("SELECT body FROM datasets WHERE id=?", (did,)).fetchone()
        if not row:
            raise ValueError("数据集不存在")
        return {"id": did, **json.loads(row["body"])}

    def datasets(self):
        with self.connect() as db:
            return [
                {"id": r["id"], **json.loads(r["body"])}
                for r in db.execute("SELECT * FROM datasets ORDER BY rowid DESC")
            ]

    def create_run(self, spec):
        rid = uid("run")
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?)",
                (rid, "queued", time.time(), json.dumps(spec, ensure_ascii=False)),
            )
        self.event(rid, "state", {"state": "queued"})
        return self.run(rid)

    def run(self, rid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if not row:
            raise ValueError("任务不存在")
        return {
            "id": row["id"],
            "state": row["state"],
            "created": row["created"],
            "spec": json.loads(row["spec"]),
        }

    def runs(self):
        with self.connect() as db:
            ids = [
                r["id"] for r in db.execute("SELECT id FROM runs ORDER BY created DESC")
            ]
        return [self.run(rid) for rid in ids]

    def state(self, rid, state):
        with self.connect() as db:
            db.execute("UPDATE runs SET state=? WHERE id=?", (state, rid))
        self.event(rid, "state", {"state": state})

    def attempt(self, rid, pid, sid, body):
        with self.connect() as db:
            cur = db.execute(
                "INSERT INTO attempts(run_id,profile_id,sample_id,body) VALUES (?,?,?,?)",
                (rid, pid, sid, json.dumps(body, ensure_ascii=False)),
            )
            aid = cur.lastrowid
        self.event(
            rid,
            "result",
            {
                "profile_id": pid,
                "sample_id": sid,
                "status": body["status"],
                "attempt_id": aid,
            },
        )

    def attempts(self, rid, latest=True):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM attempts WHERE run_id=? ORDER BY id", (rid,)
            ).fetchall()
        results = [
            dict(
                json.loads(r["body"]),
                attempt_id=r["id"],
                profile_id=r["profile_id"],
                sample_id=r["sample_id"],
            )
            for r in rows
        ]
        if not latest:
            return results
        return list({(r["profile_id"], r["sample_id"]): r for r in results}.values())

    def event(self, rid, kind, body):
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(run_id,created,kind,body) VALUES (?,?,?,?)",
                (rid, time.time(), kind, json.dumps(body, ensure_ascii=False)),
            )

    def events(self, rid, after=0, limit=500):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT ?",
                (rid, after, limit),
            ).fetchall()
        return [
            dict(
                id=r["id"],
                created=r["created"],
                kind=r["kind"],
                **json.loads(r["body"]),
            )
            for r in rows
        ]

    def all_events(self, rid):
        output, cursor = [], 0
        while True:
            batch = self.events(rid, cursor)
            output.extend(batch)
            if len(batch) < 500:
                return output
            cursor = batch[-1]["id"]
