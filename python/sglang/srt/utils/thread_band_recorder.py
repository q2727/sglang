"""Side-band trace recorder for threads kineto cannot see.

The torch/kineto profiler registers its callbacks THREAD-LOCALLY on the
thread that starts it: a long-lived worker thread (the decoupled IPC
threads) emits no host events at all -- ``record_function`` is a no-op
there and even raw aten ops are dropped (verified empirically; the
exported trace contains no track for such threads).

This module gives those threads a way into the SAME chrome trace: they
record ``(name, t0, dur)`` bands against ``time.monotonic()`` -- the very
clock kineto stamps its ``ts`` fields with (CLOCK_MONOTONIC microseconds)
-- and the profiler manager injects the bands into the exported
``*.trace.json.gz`` right after kineto writes it, on the recording
thread's own tid track, with a proper ``thread_name`` metadata row.

Usage (worker thread):

    _bands = register_thread_band_recorder("sgl-draft-ipc")
    ...
    with _bands.band("drafter_ipc.land_commit"):
        ...

The recorder is dormant (one bool check per band) until the profiler
manager flips ``set_recording(True)`` for the profile window.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# One profile window's worth of bands per thread; beyond this the recorder
# drops (and counts) rather than growing unboundedly.
_MAX_EVENTS_PER_THREAD = 200_000

_registry_lock = threading.Lock()
_recorders: list[ThreadBandRecorder] = []
_recording = False
# Clock-sync pair (see mark_clock_sync): monotonic_ns sampled inside a
# kineto-visible record_function marker on the profiled thread. Kineto's
# ``ts`` is NOT raw CLOCK_MONOTONIC (it carries its own base), so the
# injection derives the offset from this pair instead of assuming a clock.
_CLOCK_SYNC_MARKER = "sglang_band_clock_sync"
_clock_sync_monotonic_ns: Optional[int] = None


class _Band:
    __slots__ = ("_recorder", "_name", "_t0")

    def __init__(self, recorder: ThreadBandRecorder, name: str) -> None:
        self._recorder = recorder
        self._name = name
        self._t0 = 0

    def __enter__(self):
        self._t0 = time.monotonic_ns()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._recorder._append(self._name, self._t0, time.monotonic_ns())
        return False


class _NullBand:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


_NULL_BAND = _NullBand()


class ThreadBandRecorder:
    """Band recorder for ONE worker thread (register on that thread)."""

    def __init__(self, thread_label: str) -> None:
        self.thread_label = thread_label
        # Stamped on the first recorded band: construction may happen on the
        # owner's thread (e.g. the scheduler building its IPC controller)
        # while the bands come from the worker thread the track must carry.
        self.tid: Optional[int] = None
        self.pid = os.getpid()
        self._events: list[tuple[str, int, int]] = []
        self._dropped = 0
        self._lock = threading.Lock()

    def band(self, name: str):
        """A with-block band; free (one global bool) when not recording."""
        if not _recording:
            return _NULL_BAND
        return _Band(self, name)

    def _append(self, name: str, t0_ns: int, t1_ns: int) -> None:
        if self.tid is None:
            self.tid = threading.get_native_id()
        with self._lock:
            if len(self._events) >= _MAX_EVENTS_PER_THREAD:
                self._dropped += 1
                return
            self._events.append((name, t0_ns, t1_ns))

    def _drain(self) -> tuple[list[tuple[str, int, int]], int]:
        with self._lock:
            events, self._events = self._events, []
            dropped, self._dropped = self._dropped, 0
            return events, dropped


def register_thread_band_recorder(thread_label: str) -> ThreadBandRecorder:
    """Create + register a recorder for the CALLING thread."""
    recorder = ThreadBandRecorder(thread_label)
    with _registry_lock:
        _recorders.append(recorder)
    return recorder


def set_recording(enabled: bool) -> None:
    """Flip recording for every registered thread (profiler start/stop)."""
    global _recording
    _recording = enabled


def mark_clock_sync() -> None:
    """Emit the kineto/monotonic sync pair (call on the PROFILED thread,
    right after the torch profiler starts): a named record_function marker
    kineto stamps with ITS clock, paired with the monotonic reading the
    band recorders use. The injector aligns the two timelines from this."""
    global _clock_sync_monotonic_ns
    import torch

    with torch.profiler.record_function(_CLOCK_SYNC_MARKER):
        _clock_sync_monotonic_ns = time.monotonic_ns()


def inject_into_chrome_trace(trace_path: str) -> None:
    """Append every registered thread's bands into an exported kineto trace.

    Clock compatibility: kineto's ``ts`` is CLOCK_MONOTONIC in microseconds,
    the same clock ``time.monotonic_ns`` reads, so the bands land in the
    right place without any re-basing. No-op when nothing was recorded.
    """
    with _registry_lock:
        recorders = list(_recorders)
    collected: list[tuple[ThreadBandRecorder, list[tuple[str, int, int]]]] = []
    for recorder in recorders:
        events, dropped = recorder._drain()
        if dropped:
            logger.warning(
                "thread-band recorder %s dropped %d bands (ring full)",
                recorder.thread_label,
                dropped,
            )
        if events:
            collected.append((recorder, events))
    if not collected:
        return
    if _clock_sync_monotonic_ns is None:
        logger.warning(
            "side-band injection skipped: no clock-sync marker was taken "
            "(mark_clock_sync not called at profiler start)"
        )
        return
    try:
        opener = gzip.open if trace_path.endswith(".gz") else open
        with opener(trace_path, "rt") as f:
            trace = json.load(f)
        marker_ts = next(
            (
                e["ts"]
                for e in trace["traceEvents"]
                if e.get("name") == _CLOCK_SYNC_MARKER and e.get("ph") == "X"
            ),
            None,
        )
        if marker_ts is None:
            logger.warning(
                "side-band injection skipped: clock-sync marker not found in %s",
                trace_path,
            )
            return
        # kineto_ts(us) = monotonic_us + offset, derived from the sync pair.
        offset_us = float(marker_ts) - _clock_sync_monotonic_ns / 1_000
        payload: list[dict] = []
        for recorder, events in collected:
            payload.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": recorder.pid,
                    "tid": recorder.tid,
                    "args": {"name": recorder.thread_label},
                }
            )
            for name, t0_ns, t1_ns in events:
                payload.append(
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": name,
                        "pid": recorder.pid,
                        "tid": recorder.tid,
                        "ts": t0_ns / 1_000 + offset_us,
                        "dur": max(t1_ns - t0_ns, 1_000) / 1_000,
                    }
                )
        trace["traceEvents"].extend(payload)
        with opener(trace_path, "wt") as f:
            json.dump(trace, f)
        logger.info(
            "injected %d side-band events into %s (clock offset %.3f s)",
            len(payload),
            trace_path,
            offset_us / 1e6,
        )
    except Exception:
        logger.exception("side-band trace injection failed for %s", trace_path)
