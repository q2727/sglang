"""Device-side launch gate (doorbell) for the decoupled verifier's C6 wait.

EXPERIMENTAL -- verified UNSAFE on driver 580.126.09; keep default-off.
While a ``cuStreamWaitValue32`` is parked, ordinary same-process stream ops
(event records, other-stream kernel launches, allocator calls) can block the
HOST inside the driver: reproduced twice in production (process-wide freeze
until the 300s watchdog kill; py-spy showed the scheduler in a stream sync,
the IPC thread inside ``land``'s event sync / ``cuMemAlloc`` on a driver
rwlock) and deterministically in microbenchmarks (an ``Event.record`` behind
a parked wait wedges), while near-identical sequences -- the same ops with a
graph replay interleaved -- pass. The working/wedging boundary is not
predictable enough to ship. See the pitfalls log for the full forensics.

Original design, kept for a future driver or a push-model rework: the C6
gate is a bounded HOST wait for the enumeration block a verify round's
select is about to read. The doorbell turns it into a device-side wait: the
scheduler thread enqueues one ``cuStreamWaitValue32`` (GEQ, per gated seat)
on its compute stream and launches the whole round immediately; the GPU sits
at the wait until the landing stream's scatter writes the seat's stamp into
``DecoupledEnumBuffer.doorbell_flags``. Measured on B200: enqueue ~3us,
release-to-run ~0.02 ms when it works.

Correctness never depends on the doorbell: the select re-checks the real
stamps and falls back on mismatch, so a spurious or forced release only
costs a fallback round. That makes the timeout path trivial -- a watchdog
force-writes the expected value (host-side release) and the round degrades
exactly like today's gate timeout.

Deadlock topology rule: the landing scatter must run on a stream DIFFERENT
from the waiting compute stream (a landing kernel enqueued behind the wait
on the same stream could never satisfy it). The IPC landing paths run under
their own dedicated stream when the doorbell is armed.
"""

from __future__ import annotations

import ctypes
import logging

import torch

logger = logging.getLogger(__name__)

_CU_STREAM_WAIT_VALUE_GEQ = 0x0

_libcuda = None


def _cuda() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so")
    return _libcuda


class Doorbell:
    """Per-seat wait/release plumbing over the enum buffer's flag tensor."""

    def __init__(self, *, flags: torch.Tensor) -> None:
        assert flags.dtype == torch.int32 and flags.is_cuda
        self.flags = flags
        self._base_ptr = flags.data_ptr()
        # Watchdog releases ride their own stream: the compute stream is
        # parked at the wait, and the landing stream must stay independent.
        self._release_stream = torch.cuda.Stream(device=flags.device)

    def enqueue_wait(
        self, *, stream: torch.cuda.Stream, pool_idx: int, stamp: int
    ) -> bool:
        """Enqueue a GEQ wait for one seat's stamp on ``stream``. Returns
        False when the driver refuses (caller falls back to the host gate)."""
        rc = _cuda().cuStreamWaitValue32_v2(
            ctypes.c_void_p(stream.cuda_stream),
            ctypes.c_void_p(self._base_ptr + 4 * pool_idx),
            ctypes.c_uint32(stamp & 0xFFFFFFFF),
            ctypes.c_uint32(_CU_STREAM_WAIT_VALUE_GEQ),
        )
        if rc != 0:
            logger.warning("cuStreamWaitValue32 rc=%d; doorbell wait skipped", rc)
            return False
        return True

    def force_release(self, *, pool_idx: int, stamp: int) -> None:
        """Watchdog path: host-release a seat's wait by writing the expected
        value from the side stream. The select still falls back naturally
        (the real stamps never arrived); this only unparks the GPU."""
        rc = _cuda().cuStreamWriteValue32_v2(
            ctypes.c_void_p(self._release_stream.cuda_stream),
            ctypes.c_void_p(self._base_ptr + 4 * pool_idx),
            ctypes.c_uint32(stamp & 0xFFFFFFFF),
            ctypes.c_uint32(0),
        )
        if rc != 0:
            logger.error(
                "cuStreamWriteValue32 rc=%d; doorbell force-release failed "
                "(seat %d may stay parked until its block lands)",
                rc,
                pool_idx,
            )
