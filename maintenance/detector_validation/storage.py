"""Fixed-capacity, single-writer evidence storage; not a filesystem sandbox.

Workers never receive this file descriptor or its path. Every payload and index
write is range checked before pwrite. A crash leaves a fixed-size file and the last
valid checkpoint. No extraction, rollover, temporary file or growth is supported.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import stat

CAPACITY = 2 * 1024**3
RESERVE = 16 * 1024**2
HEADER = 4096
MAX_MEMBER = 4 * 1024**2
MAX_ENTRIES = 8192
MAGIC = b"DTVST001"
SLOT_MAGIC = b"DTVIX001"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_name(name):
    if not isinstance(name, str) or len(name) > 240:
        raise ValueError("invalid evidence name")
    parts = name.split("/")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p in (".", "..") for p in parts):
        raise ValueError("unsafe evidence name")
    return name


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate index key")
        result[key] = value
    return result


def exact_read(fd, size, offset):
    result = bytearray()
    while len(result) < size:
        block = os.pread(fd, size - len(result), offset + len(result))
        if not block:
            raise ValueError("truncated evidence container")
        result.extend(block)
    return bytes(result)


def checked_write(fd, data, offset, capacity):
    if offset < 0 or offset + len(data) > capacity:
        raise ValueError("aggregate evidence capacity exhausted before write")
    done = 0
    while done < len(data):
        n = os.pwrite(fd, data[done:], offset + done)
        if n <= 0:
            raise OSError("short evidence write")
        done += n


def regular_path(path):
    path = Path(path).absolute()
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            raise ValueError("symlink evidence path")
    return path


class StoreWriter:
    """One descriptor, fixed file size. Existing files are never reused."""

    def __init__(self, path, capacity=CAPACITY, reserve=RESERVE):
        if type(capacity) is not int or type(reserve) is not int:
            raise ValueError("integer storage bounds required")
        if reserve < 8192 or reserve % 2 or capacity < HEADER + reserve + 1024:
            raise ValueError("invalid storage bounds")
        if capacity > CAPACITY or reserve > RESERVE:
            raise ValueError("storage bounds exceed reviewed ceiling")
        self.path = regular_path(path)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.capacity, self.reserve = capacity, reserve
        self.position, self.generation, self.entries = HEADER, 0, {}
        try:
            # Size is bounded even when allocation fails; retain the failed file.
            os.ftruncate(self.fd, capacity)
            os.posix_fallocate(self.fd, 0, capacity)
            header = struct.pack(">8sQQ", MAGIC, capacity, reserve)
            checked_write(self.fd, header, 0, capacity)
            self.checkpoint(
                {"status": "started", "storage_guarantee": "bounded_application_evidence"}
            )
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def put(self, name, data):
        valid_name(name)
        if not isinstance(data, bytes) or len(data) > MAX_MEMBER:
            raise ValueError("member exceeds byte/type bound")
        if name in self.entries and name != "collection.json":
            raise ValueError("duplicate evidence member")
        if name not in self.entries and len(self.entries) >= MAX_ENTRIES:
            raise ValueError("member-count bound")
        end = self.position + len(data)
        if end > self.capacity - self.reserve:
            raise ValueError("aggregate evidence payload capacity exhausted before write")
        checked_write(self.fd, data, self.position, self.capacity - self.reserve)
        self.entries[name] = {"offset": self.position, "bytes": len(data), "sha256": sha(data)}
        self.position = end

    def checkpoint(self, state):
        generation = self.generation + 1
        payload = encoded(
            {
                "generation": generation,
                "position": self.position,
                "files": self.entries,
                "state": state,
            }
        )
        slot_size = self.reserve // 2
        if len(payload) > slot_size - 64:
            raise ValueError("reserved checkpoint capacity exhausted")
        slot = self.capacity - self.reserve + (generation % 2) * slot_size
        # Invalidate slot first; the other checkpoint remains valid on any crash.
        checked_write(self.fd, b"\0" * 64, slot, self.capacity)
        checked_write(self.fd, payload, slot + 64, self.capacity)
        os.fsync(self.fd)
        header = struct.pack(
            ">8sQQ32s", SLOT_MAGIC, generation, len(payload), hashlib.sha256(payload).digest()
        )
        checked_write(self.fd, header.ljust(64, b"\0"), slot, self.capacity)
        os.fsync(self.fd)
        self.generation = generation

    def inventory(self):
        return {
            name: {key: entry[key] for key in ("bytes", "sha256")}
            for name, entry in self.entries.items()
        }

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class StoreReader:
    """Read latest valid checkpoint; verify payload hashes on every bounded read."""

    def __init__(self, path):
        self.path = regular_path(path)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(self.fd).st_mode):
                raise ValueError("container must be a regular file")
            magic, self.capacity, self.reserve = struct.unpack(">8sQQ", exact_read(self.fd, 24, 0))
            if (
                magic != MAGIC
                or self.capacity > CAPACITY
                or self.reserve > RESERVE
                or self.reserve < 8192
                or self.reserve % 2
                or self.capacity < HEADER + self.reserve + 1024
                or os.fstat(self.fd).st_size != self.capacity
            ):
                raise ValueError("invalid fixed container header")
            candidates = []
            slot_size = self.reserve // 2
            for i in range(2):
                slot = self.capacity - self.reserve + i * slot_size
                try:
                    magic, generation, size, digest = struct.unpack(
                        ">8sQQ32s", exact_read(self.fd, 56, slot)
                    )
                    if magic != SLOT_MAGIC or size > slot_size - 64:
                        continue
                    payload = exact_read(self.fd, size, slot + 64)
                    if hashlib.sha256(payload).digest() != digest:
                        continue
                    index = json.loads(payload, object_pairs_hook=no_duplicates)
                    if index["generation"] != generation:
                        continue
                    candidates.append(index)
                except (ValueError, KeyError, TypeError, UnicodeError):
                    continue
            if not candidates:
                raise ValueError("no committed container checkpoint")
            index = max(candidates, key=lambda x: x["generation"])
            self.state, self.entries = index["state"], index["files"]
            if len(self.entries) > MAX_ENTRIES:
                raise ValueError("too many container members")
            ranges = []
            for name, entry in self.entries.items():
                valid_name(name)
                offset, size = entry["offset"], entry["bytes"]
                if (
                    type(offset) is not int
                    or type(size) is not int
                    or size < 0
                    or size > MAX_MEMBER
                    or offset < HEADER
                    or offset + size > self.capacity - self.reserve
                    or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                ):
                    raise ValueError("invalid member bounds/hash")
                ranges.append((offset, offset + size))
            ranges.sort()
            if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
                raise ValueError("overlapping container members")
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def read_bytes(self, name, max_size=MAX_MEMBER):
        entry = self.entries[valid_name(name)]
        if entry["bytes"] > max_size:
            raise ValueError("requested member exceeds read bound")
        data = exact_read(self.fd, entry["bytes"], entry["offset"])
        if sha(data) != entry["sha256"]:
            raise ValueError("container member digest mismatch")
        return data

    def inventory(self):
        return {
            name: {key: entry[key] for key in ("bytes", "sha256")}
            for name, entry in self.entries.items()
        }

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
