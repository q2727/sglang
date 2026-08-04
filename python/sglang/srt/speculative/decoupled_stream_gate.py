"""Host-func stream gate: park the STREAM, never the launching thread.

The C6 gate's host form parks the scheduler thread until the drafter's block
lands (or the budget runs out). That serializes the launch path: nothing
behind the gate -- select graph, verify graph, the acceptance tail -- can
even be ENQUEUED while the wait runs, and result processing of earlier
rounds is pushed behind the park (one of the wrong-phase basin's sustaining
mechanisms).

This gate is a ``cudaLaunchHostFunc`` node on the forward stream instead:
the callback blocks on the arrival board's condvar (notified by the IPC
thread's landing), the launch thread returns immediately, and the round's
launches queue up on the stream BEHIND the gate. Unlike the memop doorbell
(judged unshippable: a pending ``cuStreamWaitValue32`` froze same-process
host CUDA calls in the driver on 580.x), a blocked host func holds no
driver-level parking state -- the landing scatter, event records and
allocations all keep working while it waits, which is exactly what the
release path needs (validated by the standalone six-act probe).

Callback discipline: no CUDA API inside (spec requirement); exceptions must
not escape the ctypes trampoline (undefined behavior) -- they are logged and
swallowed, which degrades that round to a natural select fallback.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
from collections import deque
from typing import Callable

import torch

logger = logging.getLogger(__name__)

_HOSTFN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

_CUDART_CANDIDATES = ("libcudart.so.12", "libcudart.so.13", "libcudart.so")


def _load_cudart() -> ctypes.CDLL:
    last_err: Exception | None = None
    for name in _CUDART_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError as err:
            last_err = err
    found = ctypes.util.find_library("cudart")
    if found:
        return ctypes.CDLL(found)
    raise RuntimeError(f"libcudart not loadable: {last_err}")


class StreamGate:
    """Enqueue blocking host callbacks onto CUDA streams.

    Keeps every trampoline alive until its callback has RUN (a collected
    trampoline is a use-after-free inside the driver); completed entries are
    pruned lazily on the next enqueue, so steady state holds O(outstanding)
    references.
    """

    def __init__(self) -> None:
        lib = _load_cudart()
        lib.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, _HOSTFN, ctypes.c_void_p]
        lib.cudaLaunchHostFunc.restype = ctypes.c_int
        self._lib = lib
        self._live: deque[tuple[_HOSTFN, list]] = deque()

    def enqueue(self, stream: torch.cuda.Stream, fn: Callable[[], None]) -> bool:
        """Add ``fn`` as a host node on ``stream``. Returns False when the
        driver refused the node (caller falls back to the host gate)."""
        done: list = []

        def _trampoline(_userdata: object) -> None:
            try:
                fn()
            except BaseException:
                # An exception escaping a ctypes callback is undefined
                # behavior; a swallowed gate error just means this round's
                # select reads whatever stamp is there and falls back.
                logger.exception("stream-gate callback failed; round falls back")
            finally:
                done.append(True)

        cb = _HOSTFN(_trampoline)
        entry = (cb, done)
        self._live.append(entry)
        rc = self._lib.cudaLaunchHostFunc(ctypes.c_void_p(stream.cuda_stream), cb, None)
        if rc != 0:
            self._live.remove(entry)
            logger.warning("cudaLaunchHostFunc refused (rc=%d); host gate", rc)
            return False
        while self._live and self._live[0][1]:
            self._live.popleft()
        return True
