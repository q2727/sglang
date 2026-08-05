"""Target-side client for the official SSD DraftRunner service."""

from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass
from typing import Sequence
from urllib.parse import unquote, urlsplit

from sglang.srt.speculative.ssd_draft_client import (
    DraftCandidate,
    FanOutSpec,
    OutcomeKey,
    SSDDraftClient,
    SSDDraftClientError,
)
from sglang.srt.speculative.ssd_official_protocol import (
    FLAG_ERROR,
    FLAG_RESPONSE,
    BufferReader,
    BufferWriter,
    OfficialSSDOp,
    parse_error,
    recv_frame,
    send_frame,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfficialOutcomeReady:
    """Proof that the draft process has finished the next outcome tree."""

    rid: str
    branches: int
    glue_ms: float
    tree_ms: float
    populate_ms: float
    total_ms: float


class OfficialSSDDraftClient:
    """Synchronous binary client for one GPU-resident official SSD drafter.

    ``SSDWorker`` invokes BUILD on its background executor, so this synchronous
    client still overlaps the official glue/tree kernels with target verify.
    """

    def __init__(self, server_url: str, timeout: float = 600.0):
        if timeout <= 0:
            raise ValueError("The SSD draft request timeout must be positive.")
        parsed = urlsplit(server_url)
        if parsed.scheme != "unix" or not parsed.path:
            raise ValueError(
                "The official SSD server URL must use unix:///absolute/path."
            )
        self.socket_path = unquote(parsed.path)
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()

    @staticmethod
    def is_official_url(server_url: str | None) -> bool:
        return bool(server_url and urlsplit(server_url).scheme == "unix")

    def _close_unlocked(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _connect_unlocked(self) -> socket.socket:
        if self._socket is None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)
            self._socket = sock
        return self._socket

    def _request(self, op: OfficialSSDOp, payload: bytes = b"") -> bytes:
        with self._lock:
            # A reconnect-and-retry is safe for these deterministic B=1
            # operations: repeated writes target the same KV slots and BUILD
            # replaces, rather than appends to, the one outcome tree.
            for attempt in range(2):
                try:
                    sock = self._connect_unlocked()
                    send_frame(sock, op, payload)
                    response_op, flags, response = recv_frame(sock)
                    if response_op != op or not flags & FLAG_RESPONSE:
                        raise SSDDraftClientError(
                            f"Malformed official SSD response for {op.name}."
                        )
                    if flags & FLAG_ERROR:
                        raise SSDDraftClientError(parse_error(response))
                    return response
                except SSDDraftClientError:
                    raise
                except (EOFError, OSError) as exc:
                    self._close_unlocked()
                    if attempt:
                        raise SSDDraftClientError(
                            f"Official SSD {op.name} request failed: {exc}"
                        ) from exc
            raise AssertionError("unreachable")

    @staticmethod
    def _placeholder_recoveries(
        draft_length: int, fan_out: FanOutSpec
    ) -> tuple[tuple[int, ...], ...]:
        # Recovery alternatives stay in the official drafter's tensor cache.
        # Preserve K+1 endpoint shape for target-side metrics/validation.
        SSDDraftClient.normalize_fan_outs(draft_length, fan_out)
        return tuple(() for _ in range(draft_length + 1))

    def _candidate_from_response(
        self, payload: bytes, draft_length: int, fan_out: FanOutSpec
    ) -> DraftCandidate:
        reader = BufferReader(payload)
        cache_hit = bool(reader.u8())
        _service_ms = reader.f64()
        tokens = reader.int_list()
        reader.finish()
        if len(tokens) != draft_length:
            raise SSDDraftClientError(
                f"Official SSD returned {len(tokens)} tokens; "
                f"expected {draft_length}."
            )
        return DraftCandidate(
            tokens=tokens,
            recovery_tokens=self._placeholder_recoveries(draft_length, fan_out),
            cache_hit=cache_hit,
        )

    def ping(self) -> None:
        reader = BufferReader(self._request(OfficialSSDOp.PING))
        if reader.text() != "pong":
            raise SSDDraftClientError("Official SSD service returned a bad ping.")
        reader.finish()

    def reset(self) -> None:
        reader = BufferReader(self._request(OfficialSSDOp.RESET))
        reader.finish()

    def init_draft(
        self,
        rid: str,
        prefix: Sequence[int],
        draft_length: int,
        fan_out: FanOutSpec,
    ) -> DraftCandidate:
        if not prefix:
            raise ValueError("SSD cannot initialize from an empty prefix.")
        payload = BufferWriter().text(rid).int_list(prefix).finish()
        return self._candidate_from_response(
            self._request(OfficialSSDOp.INIT, payload), draft_length, fan_out
        )

    def jit_draft(
        self,
        rid: str,
        prefix: Sequence[int],
        draft_length: int,
        fan_out: FanOutSpec,
    ) -> DraftCandidate:
        if not prefix:
            raise ValueError("SSD cannot draft from an empty prefix.")
        payload = BufferWriter().text(rid).int_list(prefix).finish()
        return self._candidate_from_response(
            self._request(OfficialSSDOp.JIT, payload), draft_length, fan_out
        )

    def build_outcome_cache(
        self,
        rid: str,
        canonical_prefix: Sequence[int],
        candidate: DraftCandidate,
        draft_length: int,
        fan_out: FanOutSpec,
    ) -> OfficialOutcomeReady:
        if not canonical_prefix:
            raise ValueError("SSD cannot build from an empty prefix.")
        if len(candidate.tokens) != draft_length:
            raise ValueError(
                f"SSD expected {draft_length} candidate tokens, "
                f"got {len(candidate.tokens)}."
            )
        branches = sum(SSDDraftClient.normalize_fan_outs(draft_length, fan_out))
        payload = (
            BufferWriter()
            .text(rid)
            .i64(len(canonical_prefix))
            .i32(int(canonical_prefix[-1]))
            .u8(candidate.cache_hit is True)
            .int_list(candidate.tokens)
            .finish()
        )
        reader = BufferReader(self._request(OfficialSSDOp.BUILD, payload))
        returned_branches = reader.u32()
        glue_ms = reader.f64()
        tree_ms = reader.f64()
        populate_ms = reader.f64()
        total_ms = reader.f64()
        reader.finish()
        if returned_branches != branches:
            raise SSDDraftClientError(
                f"Official SSD built {returned_branches} branches; "
                f"expected {branches}."
            )
        return OfficialOutcomeReady(
            rid=rid,
            branches=returned_branches,
            glue_ms=glue_ms,
            tree_ms=tree_ms,
            populate_ms=populate_ms,
            total_ms=total_ms,
        )

    def select_outcome(
        self,
        handle: OfficialOutcomeReady,
        rid: str,
        canonical_prefix: Sequence[int],
        outcome_key: OutcomeKey,
        draft_length: int,
        fan_out: FanOutSpec,
    ) -> DraftCandidate:
        if handle.rid != rid:
            raise SSDDraftClientError(
                f"Official SSD outcome handle belongs to {handle.rid}, not {rid}."
            )
        payload = (
            BufferWriter()
            .text(rid)
            .i64(len(canonical_prefix))
            .i32(int(outcome_key[0]))
            .i32(int(outcome_key[1]))
            .finish()
        )
        return self._candidate_from_response(
            self._request(OfficialSSDOp.SELECT, payload), draft_length, fan_out
        )
