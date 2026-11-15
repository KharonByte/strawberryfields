"""Bounded evidence framing; child supplies no filesystem destination."""

import hashlib
import os
import struct
import stat

from .output_bounds import START_BYTES

CAPS = {1: START_BYTES, 2: 160128, 3: 160128, 4: 480128, 5: 262144}
HEADER = struct.Struct(">4sBI32s")


class EvidenceStream:
    """One cooperative child writer to a supervisor-created pipe."""

    def __init__(self, fd):
        if type(fd) is not int or fd < 3:
            raise ValueError("Evidence FD must be distinct from standard streams")
        if not stat.S_ISFIFO(os.fstat(fd).st_mode):
            raise ValueError("Evidence FD must be a pipe, never a filesystem file")
        self.fd = fd
        self.sent = set()

    def emit(self, kind, payload):
        if kind not in CAPS or kind in self.sent or len(self.sent) >= 5:
            raise ValueError("Unknown or repeated evidence frame")
        if not isinstance(payload, bytes) or len(payload) > CAPS[kind]:
            raise ValueError("Evidence payload exceeds fixed bound")
        if (not self.sent and kind != 1) or 5 in self.sent:
            raise ValueError("Evidence frame sequence violation")
        self.sent.add(kind)
        header = HEADER.pack(b"DTV1", kind, len(payload), hashlib.sha256(payload).digest())
        for part in (header, payload):
            remaining = memoryview(part)
            while remaining:
                written = os.write(self.fd, remaining)
                if written <= 0:
                    raise BrokenPipeError("Evidence pipe failed")
                remaining = remaining[written:]
        return {
            "frame_kind": kind,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
