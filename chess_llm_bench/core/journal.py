"""Durable run files, tolerant tail reads, exclusive writer leases and stable keys."""

import fcntl
import json
import os
import tempfile
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines) - 1 and not raw.endswith(b"\n"):
                break  # Active writer or an interrupted final append.
            raise
    return rows


def append_row(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Recover only an unfinished trailing write, preserving it for inspection.
    # Complete malformed rows are errors and never silently discarded.
    if path.exists():
        with path.open("rb+") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            if size:
                stream.seek(-1, 2)
                if stream.read(1) != b"\n":
                    stream.seek(0)
                    raw = stream.read()
                    tail = raw.rsplit(b"\n", 1)[-1]
                    try:
                        json.loads(tail)
                        stream.seek(0, 2)
                        stream.write(b"\n")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        path.with_suffix(path.suffix + ".partial").write_bytes(tail)
                        stream.truncate(size - len(tail))
    with path.open("a") as stream:
        stream.write(json.dumps(record, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def latest(rows, fields):
    return list({tuple(r.get(k) for k in fields): r for r in rows}.values())


class RunLease:
    def __init__(self, directory):
        self.stream = (Path(directory) / ".writer.lock").open("a+")

    def __enter__(self):
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError("This run already has an active writer") from None
        return self

    def __exit__(self, *args):
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()
