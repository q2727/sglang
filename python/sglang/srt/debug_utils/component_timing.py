"""One-shot component timing for the batch-8 Qwen3.5 KT pilot.

The profiler is dormant unless both environment variables are present:

``SGLANG_COMPONENT_TIMING_DIR``
    Directory receiving one JSONL file per TP rank.

``SGLANG_COMPONENT_TIMING_CONTROL``
    Text control file.  ``record:<session>:natural`` records the first eligible
    batch for a session.  ``record:<session>:cpu_isolate`` additionally asks the
    KT wrapper to synchronize its CPU branch immediately after submission.

This is diagnostic instrumentation, not a production telemetry path.
"""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist


_state: Optional[dict[str, Any]] = None
_recorded_sessions: set[str] = set()


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(torch.cuda.current_device()) if torch.cuda.is_available() else 0


def _read_control() -> Optional[tuple[str, str]]:
    control_path = os.environ.get("SGLANG_COMPONENT_TIMING_CONTROL")
    output_dir = os.environ.get("SGLANG_COMPONENT_TIMING_DIR")
    if not control_path or not output_dir:
        return None
    try:
        value = Path(control_path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    fields = value.split(":", 2)
    if len(fields) != 3 or fields[0] != "record" or not fields[1]:
        return None
    if fields[2] not in ("natural", "cpu_isolate"):
        raise RuntimeError(f"Unknown component timing mode: {fields[2]!r}")
    return fields[1], fields[2]


def begin_cycle(
    *,
    batch_size: int,
    request_ids: Iterable[str],
    forward_mode: str,
) -> bool:
    """Begin the first eligible batch for the current control-file session."""
    global _state
    if _state is not None:
        raise RuntimeError("Nested component timing cycles are not supported")
    control = _read_control()
    required_batch = int(os.environ.get("SGLANG_COMPONENT_TIMING_BATCH_SIZE", "8"))
    if control is None or batch_size != required_batch:
        return False
    session, mode = control
    if session in _recorded_sessions:
        return False
    torch.cuda.synchronize()
    _state = {
        "schema_version": 2,
        "session": session,
        "timing_mode": mode,
        "rank": _rank(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "batch_size": int(batch_size),
        "request_ids": [str(value) for value in request_ids],
        "forward_mode_at_entry": str(forward_mode),
        "started_at_unix_seconds": time.time(),
        "cycle_started_perf": time.perf_counter(),
        "current_phase": None,
        "phase_started_perf": None,
        "phases": [],
        "attention_events": [],
        "shared_dense_events": [],
        "cpu_moe_calls": [],
    }
    return True


def is_active() -> bool:
    return _state is not None


def cpu_isolation_enabled() -> bool:
    return bool(
        _state is not None
        and _state["timing_mode"] == "cpu_isolate"
        and _state["current_phase"] == "target_verify"
    )


@contextmanager
def phase(name: str):
    """Synchronize at boundaries and record critical-path wall duration."""
    if _state is None:
        yield
        return
    if _state["current_phase"] is not None:
        raise RuntimeError("Nested component timing phases are not supported")
    torch.cuda.synchronize()
    _state["current_phase"] = str(name)
    _state["phase_started_perf"] = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        ended = time.perf_counter()
        _state["phases"].append(
            {
                "name": str(name),
                "wall_ms": (ended - _state["phase_started_perf"]) * 1000.0,
            }
        )
        _state["current_phase"] = None
        _state["phase_started_perf"] = None


@contextmanager
def attention_region(
    *, layer: int, kind: str, num_tokens: int, forward_mode: str
):
    """Record main-stream CUDA event duration for one attention module call."""
    if _state is None or _state["current_phase"] is None:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _state["attention_events"].append(
            {
                "phase": _state["current_phase"],
                "layer": int(layer),
                "kind": str(kind),
                "num_tokens": int(num_tokens),
                "forward_mode": str(forward_mode),
                "start": start,
                "end": end,
            }
        )


@contextmanager
def shared_dense_region(
    *, layer: int, num_tokens: int, forward_mode: str
):
    """Record the GPU shared-expert dense branch (the experiment's SD stage)."""
    if _state is None or _state["current_phase"] is None:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _state["shared_dense_events"].append(
            {
                "phase": _state["current_phase"],
                "layer": int(layer),
                "kind": "shared_dense",
                "num_tokens": int(num_tokens),
                "forward_mode": str(forward_mode),
                "start": start,
                "end": end,
            }
        )


def record_cpu_moe(
    *,
    layer: int,
    num_tokens: int,
    submit_python_ms: float,
    isolated: bool,
    cpu_service_start: torch.cuda.Event,
    cpu_service_end: torch.cuda.Event,
    residual_wait_start: torch.cuda.Event,
    residual_wait_end: torch.cuda.Event,
    hybrid_critical_start: torch.cuda.Event,
    hybrid_critical_end: torch.cuda.Event,
) -> None:
    if _state is None or _state["current_phase"] != "target_verify":
        return
    _state["cpu_moe_calls"].append(
        {
            "phase": _state["current_phase"],
            "layer": int(layer),
            "num_tokens": int(num_tokens),
            "submit_python_ms": float(submit_python_ms),
            "isolated": bool(isolated),
            "cpu_service_start": cpu_service_start,
            "cpu_service_end": cpu_service_end,
            "residual_wait_start": residual_wait_start,
            "residual_wait_end": residual_wait_end,
            "hybrid_critical_start": hybrid_critical_start,
            "hybrid_critical_end": hybrid_critical_end,
        }
    )


def _aggregate_attention(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        groups.setdefault((row["phase"], row["kind"]), []).append(row["cuda_ms"])
    return [
        {
            "phase": phase_name,
            "kind": kind,
            "calls": len(values),
            "sum_cuda_ms": sum(values),
            "mean_cuda_ms": sum(values) / len(values),
            "max_cuda_ms": max(values),
        }
        for (phase_name, kind), values in sorted(groups.items())
    ]


def _resolve_cuda_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in ("start", "end")
        }
        | {"cuda_ms": float(row["start"].elapsed_time(row["end"]))}
        for row in rows
    ]


def _resolve_cpu_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_fields = {
        "cpu_service_start",
        "cpu_service_end",
        "residual_wait_start",
        "residual_wait_end",
        "hybrid_critical_start",
        "hybrid_critical_end",
    }
    return [
        {key: value for key, value in row.items() if key not in event_fields}
        | {
            "cpu_service_cuda_ms": float(
                row["cpu_service_start"].elapsed_time(row["cpu_service_end"])
            ),
            "residual_wait_cuda_ms": float(
                row["residual_wait_start"].elapsed_time(row["residual_wait_end"])
            ),
            "hybrid_critical_cuda_ms": float(
                row["hybrid_critical_start"].elapsed_time(
                    row["hybrid_critical_end"]
                )
            ),
        }
        for row in rows
    ]


def end_cycle(*, accepted_draft_lengths: Iterable[int]) -> Optional[Path]:
    global _state
    if _state is None:
        return None
    torch.cuda.synchronize()
    ended_perf = time.perf_counter()
    attention_rows = _resolve_cuda_events(_state["attention_events"])
    shared_dense_rows = _resolve_cuda_events(_state["shared_dense_events"])
    cpu_moe_rows = _resolve_cpu_events(_state["cpu_moe_calls"])
    accepted = [int(value) for value in accepted_draft_lengths]
    payload = {
        key: value
        for key, value in _state.items()
        if key
        not in (
            "cycle_started_perf",
            "current_phase",
            "phase_started_perf",
            "attention_events",
            "shared_dense_events",
            "cpu_moe_calls",
        )
    }
    payload.update(
        {
            "completed_at_unix_seconds": time.time(),
            "cycle_wall_ms": (
                ended_perf - _state["cycle_started_perf"]
            )
            * 1000.0,
            "accepted_draft_lengths": accepted,
            "accepted_output_tokens": sum(accepted) + len(accepted),
            "attention_calls": attention_rows,
            "attention_aggregates": _aggregate_attention(attention_rows),
            "shared_dense_calls": shared_dense_rows,
            "shared_dense_aggregates": _aggregate_attention(shared_dense_rows),
            "cpu_moe_calls": cpu_moe_rows,
            "cpu_moe_aggregates": {
                field: sum(float(row[field]) for row in cpu_moe_rows)
                for field in (
                    "submit_python_ms",
                    "cpu_service_cuda_ms",
                    "residual_wait_cuda_ms",
                    "hybrid_critical_cuda_ms",
                )
            },
        }
    )
    output_dir = Path(os.environ["SGLANG_COMPONENT_TIMING_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"component_timing_rank{_state['rank']}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _recorded_sessions.add(str(_state["session"]))
    _state = None
    return path


def abort_cycle() -> None:
    global _state
    _state = None
