"""Binary Unix-socket protocol for the colocated official SSD drafter.

The protocol intentionally transports only request metadata and token ids.
Draft KV, outcome logits, and the outcome tree remain in the draft process.
"""

from __future__ import annotations

import socket
import struct
from enum import IntEnum
from typing import Iterable


MAGIC = b"KSSD"
VERSION = 1
MAX_FRAME_BYTES = 64 * 1024 * 1024

FLAG_RESPONSE = 1 << 0
FLAG_ERROR = 1 << 1

_FRAME_HEADER = struct.Struct("<4sBBHI")


class OfficialSSDOp(IntEnum):
    PING = 1
    INIT = 2
    BUILD = 3
    SELECT = 4
    JIT = 5
    RESET = 6
    SHUTDOWN = 7


class OfficialSSDProtocolError(RuntimeError):
    pass


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("Official SSD socket closed while receiving a frame.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(
    sock: socket.socket,
    op: OfficialSSDOp | int,
    payload: bytes = b"",
    *,
    flags: int = 0,
) -> None:
    if len(payload) > MAX_FRAME_BYTES:
        raise OfficialSSDProtocolError(
            f"Official SSD frame is too large: {len(payload)} bytes."
        )
    header = _FRAME_HEADER.pack(MAGIC, VERSION, int(op), flags, len(payload))
    sock.sendall(header)
    if payload:
        sock.sendall(payload)


def recv_frame(sock: socket.socket) -> tuple[OfficialSSDOp, int, bytes]:
    header = recv_exact(sock, _FRAME_HEADER.size)
    magic, version, raw_op, flags, payload_size = _FRAME_HEADER.unpack(header)
    if magic != MAGIC:
        raise OfficialSSDProtocolError(f"Bad Official SSD magic: {magic!r}.")
    if version != VERSION:
        raise OfficialSSDProtocolError(
            f"Unsupported Official SSD protocol version {version}."
        )
    if payload_size > MAX_FRAME_BYTES:
        raise OfficialSSDProtocolError(
            f"Official SSD frame declares {payload_size} bytes."
        )
    try:
        op = OfficialSSDOp(raw_op)
    except ValueError as exc:
        raise OfficialSSDProtocolError(
            f"Unknown Official SSD operation {raw_op}."
        ) from exc
    return op, flags, recv_exact(sock, payload_size)


class BufferWriter:
    def __init__(self) -> None:
        self._data = bytearray()

    def u8(self, value: int | bool) -> "BufferWriter":
        self._data.extend(struct.pack("<B", int(value)))
        return self

    def i32(self, value: int) -> "BufferWriter":
        self._data.extend(struct.pack("<i", int(value)))
        return self

    def u32(self, value: int) -> "BufferWriter":
        self._data.extend(struct.pack("<I", int(value)))
        return self

    def i64(self, value: int) -> "BufferWriter":
        self._data.extend(struct.pack("<q", int(value)))
        return self

    def f64(self, value: float) -> "BufferWriter":
        self._data.extend(struct.pack("<d", float(value)))
        return self

    def text(self, value: str) -> "BufferWriter":
        encoded = value.encode("utf-8")
        self.u32(len(encoded))
        self._data.extend(encoded)
        return self

    def int_list(self, values: Iterable[int]) -> "BufferWriter":
        normalized = [int(value) for value in values]
        self.u32(len(normalized))
        if normalized:
            self._data.extend(struct.pack(f"<{len(normalized)}i", *normalized))
        return self

    def finish(self) -> bytes:
        return bytes(self._data)


class BufferReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def _take(self, size: int) -> bytes:
        end = self._offset + size
        if end > len(self._payload):
            raise OfficialSSDProtocolError(
                "Truncated Official SSD protocol payload."
            )
        value = self._payload[self._offset:end]
        self._offset = end
        return value

    def u8(self) -> int:
        return struct.unpack("<B", self._take(1))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._take(4))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self._take(8))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self._take(8))[0]

    def text(self) -> str:
        return self._take(self.u32()).decode("utf-8")

    def int_list(self) -> list[int]:
        count = self.u32()
        if not count:
            return []
        return list(struct.unpack(f"<{count}i", self._take(4 * count)))

    def finish(self) -> None:
        if self._offset != len(self._payload):
            raise OfficialSSDProtocolError(
                f"Official SSD payload has {len(self._payload) - self._offset} "
                "unexpected trailing bytes."
            )


def error_payload(message: str) -> bytes:
    return BufferWriter().text(message).finish()


def parse_error(payload: bytes) -> str:
    reader = BufferReader(payload)
    message = reader.text()
    reader.finish()
    return message
