"""Durable JSONL preparation journals. A partial final write is recoverable."""

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"binary_bytes":len(value), "sha256":hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {str(k):json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    raise TypeError(f"unsupported provenance type: {type(value).__name__}")


def recover(path):
    """Read physical LF-delimited records and truncate only an incomplete tail.

    A malformed complete line is corruption, not an interrupted append: fail.
    The caller must hold the split's exclusive preparation lock.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r+b") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.seek(position)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                break
            records.append(json.loads(line))
    return records


def append(stream, value, sync=True):
    """Append one record. ``sync=False`` batches the fsync (the caller must end
    with a synced record); a torn tail is still recoverable either way."""
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n"
    stream.write(payload)
    stream.flush()
    if sync:
        os.fsync(stream.fileno())


@contextlib.contextmanager
def exclusive_lock(path):
    """Non-blocking exclusive lock on a file, on POSIX and on Windows.

    POSIX keeps the historical ``fcntl.flock`` behaviour. Windows uses a
    one-byte region lock via ``msvcrt``; the lock file is otherwise empty and
    has no semantic meaning. The lock is released in a ``finally`` block, so an
    exception in the caller never leaves it held.
    """
    with Path(path).open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.seek(0)
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, value):
    """Write a JSON record atomically, with a per-process temporary file.

    Several ``prepare --split`` jobs can share one output directory. A fixed
    ``.tmp`` name makes them fight over the same file on Windows (WinError 32),
    so the suffix must include the writer's pid. The final ``os.replace`` is
    atomic and all writers of a given file write the same content.
    """
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        append(stream, value)
    os.replace(temporary, path)
