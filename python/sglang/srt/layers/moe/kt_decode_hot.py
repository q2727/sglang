# SPDX-License-Identifier: Apache-2.0
"""Online hot-expert replacement for KT speculative target verification."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from safetensors import safe_open

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod


logger = logging.getLogger(__name__)

_REGISTERED_LAYERS: dict[int, "KTEPWrapperMethod"] = {}
_INITIAL_RESIDENT: dict[int, torch.Tensor] = {}
_INITIAL_WEIGHT_BACKUP: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_LAYER_STATES: dict[int, "_LayerState"] = {}
_EXPERT_SOURCE: Optional["_PackedBF16ExpertSource"] = None
_RAW_SOURCE_VALIDATED = False
_UPDATE_STEP = 0


@dataclass
class _LayerState:
    ema: torch.Tensor
    resident: torch.Tensor
    ages: torch.Tensor


@dataclass
class _Promotion:
    layer_idx: int
    slot: int
    victim: int
    candidate: int
    current_tokens: int
    candidate_score: float
    victim_score: float


def register_kt_decode_hot_layer(method: "KTEPWrapperMethod") -> None:
    layer_idx = int(method.kt_config.layer_idx)
    previous = _REGISTERED_LAYERS.get(layer_idx)
    if previous is not None and previous is not method:
        raise RuntimeError(f"KT decode-hot layer {layer_idx} was registered twice")
    _REGISTERED_LAYERS[layer_idx] = method
    _INITIAL_RESIDENT[layer_idx] = method.gpu_index_to_logical.clone()


def _state_for(method: "KTEPWrapperMethod") -> _LayerState:
    layer_idx = int(method.kt_config.layer_idx)
    resident = method.gpu_index_to_logical.to(dtype=torch.int64, device="cpu")
    state = _LAYER_STATES.get(layer_idx)
    if state is None or not torch.equal(state.resident, resident):
        state = _LayerState(
            ema=torch.zeros(method.global_num_experts, dtype=torch.float32),
            resident=resident.clone(),
            ages=torch.full(
                (method.num_gpu_experts,),
                int(method.kt_config.kt_decode_hot_min_residency),
                dtype=torch.int64,
            ),
        )
        _LAYER_STATES[layer_idx] = state
    return state


def _select_layer_promotions(
    method: "KTEPWrapperMethod",
    counts: torch.Tensor,
    *,
    allow_promotion: bool = True,
) -> tuple[list[_Promotion], torch.Tensor]:
    config = method.kt_config
    state = _state_for(method)
    decay = float(config.kt_decode_hot_ema_decay)
    state.ema.mul_(decay).add_(counts.to(torch.float32), alpha=1.0 - decay)
    state.ages.add_(1)

    resident = state.resident.clone()
    ages = state.ages.clone()
    promotions: list[_Promotion] = []
    if not allow_promotion:
        return promotions, resident
    for _ in range(int(config.kt_decode_hot_max_promotions)):
        resident_set = set(int(value) for value in resident.tolist())
        candidates = [
            expert_id
            for expert_id in range(method.global_num_experts)
            if expert_id not in resident_set
            and int(counts[expert_id]) >= int(config.kt_decode_hot_min_tokens)
        ]
        eligible_slots = [
            slot
            for slot in range(method.num_gpu_experts)
            if int(ages[slot]) >= int(config.kt_decode_hot_min_residency)
        ]
        if not candidates or not eligible_slots:
            break

        candidate = min(
            candidates,
            key=lambda expert_id: (
                -float(state.ema[expert_id]),
                -int(counts[expert_id]),
                expert_id,
            ),
        )
        slot = min(
            eligible_slots,
            key=lambda slot_id: (
                float(state.ema[int(resident[slot_id])]),
                -int(ages[slot_id]),
                slot_id,
            ),
        )
        victim = int(resident[slot])
        candidate_score = float(state.ema[candidate])
        victim_score = float(state.ema[victim])
        if candidate_score <= victim_score * float(config.kt_decode_hot_hysteresis):
            break

        promotions.append(
            _Promotion(
                layer_idx=int(config.layer_idx),
                slot=slot,
                victim=victim,
                candidate=candidate,
                current_tokens=int(counts[candidate]),
                candidate_score=candidate_score,
                victim_score=victim_score,
            )
        )
        resident[slot] = candidate
        ages[slot] = 0

    state.resident.copy_(resident)
    state.ages.copy_(ages)
    return promotions, resident


def _is_promotion_step(
    step: int,
    num_gpu_experts: int,
    max_promotions: int,
    refresh_interval: int,
) -> bool:
    initial_fill_steps = (num_gpu_experts + max_promotions - 1) // max_promotions
    return step <= initial_fill_steps or step % refresh_interval == 0


class _PackedBF16ExpertSource:
    """Lazily mmap packed Qwen BF16 experts and stage one local TP shard."""

    def __init__(self, method: "KTEPWrapperMethod") -> None:
        self.model_dir = Path(method.kt_config.weight_path)
        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                "KT decode hot replacement requires model.safetensors.index.json "
                f"under {self.model_dir}"
            )
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self.handles: dict[Path, object] = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        layer = method._decode_hot_layer
        if layer is None or not hasattr(layer, "w13_weight") or not hasattr(
            layer, "w2_weight"
        ):
            raise ValueError("KT decode hot replacement requires BF16 w13/w2 weights")
        self.device = layer.w13_weight.device
        self.w13_shape = tuple(layer.w13_weight.shape[1:])
        self.w2_shape = tuple(layer.w2_weight.shape[1:])
        if layer.w13_weight.dtype != torch.bfloat16 or layer.w2_weight.dtype != torch.bfloat16:
            raise ValueError("KT decode hot replacement requires BF16 w13/w2 weights")

        self.host_buffers = [
            (
                torch.empty(self.w13_shape, dtype=torch.bfloat16, pin_memory=True),
                torch.empty(self.w2_shape, dtype=torch.bfloat16, pin_memory=True),
            )
            for _ in range(2)
        ]
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.buffer_in_flight = [False, False]
        self.next_buffer = 0

    def _get_tensor(self, key: str) -> torch.Tensor:
        tensor = self.tensors.get(key)
        if tensor is not None:
            return tensor
        try:
            filename = self.weight_map[key]
        except KeyError as exc:
            raise KeyError(
                f"Packed BF16 checkpoint does not contain required tensor {key}"
            ) from exc
        path = self.model_dir / filename
        handle = self.handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
            self.handles[path] = handle
        tensor = handle.get_tensor(key)
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 tensor for {key}, got {tensor.dtype}")
        self.tensors[key] = tensor
        return tensor

    def _stage_into(
        self,
        method: "KTEPWrapperMethod",
        promotion: _Promotion,
        w13_host: torch.Tensor,
        w2_host: torch.Tensor,
    ) -> float:
        prefix = (
            f"model.language_model.layers.{promotion.layer_idx}.mlp.experts"
        )
        gate_up = self._get_tensor(f"{prefix}.gate_up_proj")
        down = self._get_tensor(f"{prefix}.down_proj")

        intermediate_per_rank = self.w2_shape[1]
        intermediate_size = intermediate_per_rank * self.tp_size
        hidden_size = self.w2_shape[0]
        expected_gate_shape = (
            method.global_num_experts,
            2 * intermediate_size,
            hidden_size,
        )
        expected_down_shape = (
            method.global_num_experts,
            hidden_size,
            intermediate_size,
        )
        if tuple(gate_up.shape) != expected_gate_shape or tuple(down.shape) != expected_down_shape:
            raise ValueError(
                f"Unexpected packed expert shapes at layer {promotion.layer_idx}: "
                f"gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}"
            )

        start = self.tp_rank * intermediate_per_rank
        end = start + intermediate_per_rank
        started = time.perf_counter()
        expert_id = promotion.candidate
        w13_host[:intermediate_per_rank].copy_(gate_up[expert_id, start:end])
        w13_host[intermediate_per_rank:].copy_(
            gate_up[expert_id, intermediate_size + start : intermediate_size + end]
        )
        w2_host.copy_(down[expert_id, :, start:end])
        return (time.perf_counter() - started) * 1000.0

    def stage_and_enqueue(
        self, method: "KTEPWrapperMethod", promotion: _Promotion
    ) -> float:
        buffer_idx = self.next_buffer
        self.next_buffer = (self.next_buffer + 1) % len(self.host_buffers)
        if self.buffer_in_flight[buffer_idx]:
            self.copy_events[buffer_idx].synchronize()

        w13_host, w2_host = self.host_buffers[buffer_idx]
        stage_ms = self._stage_into(method, promotion, w13_host, w2_host)

        layer = method._decode_hot_layer
        with torch.cuda.stream(self.copy_stream), torch.no_grad():
            layer.w13_weight.data[promotion.slot].copy_(w13_host, non_blocking=True)
            layer.w2_weight.data[promotion.slot].copy_(w2_host, non_blocking=True)
            self.copy_events[buffer_idx].record(self.copy_stream)
        self.buffer_in_flight[buffer_idx] = True
        return stage_ms

    def validate_initial_slot(
        self,
        method: "KTEPWrapperMethod",
        slot: int,
        backup: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        buffer_idx = 0
        if self.buffer_in_flight[buffer_idx]:
            self.copy_events[buffer_idx].synchronize()
        candidate = int(_INITIAL_RESIDENT[int(method.kt_config.layer_idx)][slot])
        promotion = _Promotion(
            layer_idx=int(method.kt_config.layer_idx),
            slot=slot,
            victim=candidate,
            candidate=candidate,
            current_tokens=0,
            candidate_score=0.0,
            victim_score=0.0,
        )
        w13_host, w2_host = self.host_buffers[buffer_idx]
        self._stage_into(method, promotion, w13_host, w2_host)
        w13_reference, w2_reference = backup
        w13_exact = torch.equal(w13_host, w13_reference[slot])
        w2_exact = torch.equal(w2_host, w2_reference[slot])
        if not w13_exact or not w2_exact:
            w13_error = float(
                (w13_host.float() - w13_reference[slot].float()).abs().max()
            )
            w2_error = float(
                (w2_host.float() - w2_reference[slot].float()).abs().max()
            )
            raise RuntimeError(
                "Packed checkpoint staging does not match SGLang GPU weights: "
                f"layer={promotion.layer_idx}, rank={self.tp_rank}, "
                f"w13_exact={w13_exact}, w2_exact={w2_exact}, "
                f"max_errors=({w13_error}, {w2_error})"
            )

    @property
    def bytes_per_promotion(self) -> int:
        w13, w2 = self.host_buffers[0]
        return (
            w13.numel() * w13.element_size()
            + w2.numel() * w2.element_size()
        )


def _get_expert_source(method: "KTEPWrapperMethod") -> _PackedBF16ExpertSource:
    global _EXPERT_SOURCE
    if _EXPERT_SOURCE is None:
        _EXPERT_SOURCE = _PackedBF16ExpertSource(method)
    elif _EXPERT_SOURCE.model_dir != Path(method.kt_config.weight_path):
        raise RuntimeError("All KT decode-hot layers must use the same checkpoint")
    return _EXPERT_SOURCE


def _ensure_initial_weight_backup(methods: list["KTEPWrapperMethod"]) -> None:
    if _INITIAL_WEIGHT_BACKUP:
        return
    rank = get_tensor_model_parallel_rank()
    started = time.perf_counter()
    total_bytes = 0
    for method in methods:
        layer = method._decode_hot_layer
        with torch.no_grad():
            w13 = layer.w13_weight.detach().cpu().contiguous()
            w2 = layer.w2_weight.detach().cpu().contiguous()
        _INITIAL_WEIGHT_BACKUP[int(method.kt_config.layer_idx)] = (w13, w2)
        total_bytes += (
            w13.numel() * w13.element_size() + w2.numel() * w2.element_size()
        )
    if rank == 0:
        logger.info(
            "[kt-decode-hot] backed up initial GPU residency: %.2f GiB in %.2fms",
            total_bytes / 1024**3,
            (time.perf_counter() - started) * 1000.0,
        )


def _resident_slot_bytes(
    backup: tuple[torch.Tensor, torch.Tensor],
) -> int:
    return sum(tensor[0].numel() * tensor.element_size() for tensor in backup)


def _validate_raw_source(
    source: _PackedBF16ExpertSource, methods: list["KTEPWrapperMethod"]
) -> None:
    global _RAW_SOURCE_VALIDATED
    if _RAW_SOURCE_VALIDATED:
        return
    sample_positions = sorted({0, len(methods) // 2, len(methods) - 1})
    for position in sample_positions:
        method = methods[position]
        layer_idx = int(method.kt_config.layer_idx)
        source.validate_initial_slot(
            method, slot=0, backup=_INITIAL_WEIGHT_BACKUP[layer_idx]
        )
    _RAW_SOURCE_VALIDATED = True
    logger.info(
        "[kt-decode-hot] rank=%d raw checkpoint staging exactly matched "
        "initial GPU weights for layers %s",
        get_tensor_model_parallel_rank(),
        [int(methods[position].kt_config.layer_idx) for position in sample_positions],
    )


def _update_mapping(
    method: "KTEPWrapperMethod", slot: int, candidate: int
) -> int:
    victim = int(method.gpu_index_to_logical[slot])
    method.gpu_experts_mask[victim] = False
    method.gpu_experts_mask[candidate] = True
    method.logical_to_gpu_index[victim] = -1
    method.logical_to_gpu_index[candidate] = slot
    method.gpu_index_to_logical[slot] = candidate
    method.gpu_experts_mask_cuda.copy_(method.gpu_experts_mask)
    method.logical_to_gpu_index_cuda.copy_(method.logical_to_gpu_index)
    if method.wrapper is not None:
        method.wrapper.gpu_experts_mask.copy_(method.gpu_experts_mask)
    return victim


def _write_profile(record: dict) -> None:
    profile_path = os.environ.get("SGLANG_KT_DECODE_HOT_PROFILE_PATH")
    if not profile_path:
        return
    path = Path(profile_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _read_control_mode() -> str:
    control_path = os.environ.get("SGLANG_KT_DECODE_HOT_CONTROL_PATH")
    if not control_path:
        return "dynamic"
    try:
        return Path(control_path).read_text(encoding="utf-8").strip().lower()
    except FileNotFoundError:
        return "dynamic"


def _synchronized_control_mode(device: torch.device) -> str:
    rank = get_tensor_model_parallel_rank()
    mode_names = ("static", "dynamic", "invalid")
    mode_code = torch.zeros((), dtype=torch.int32, device=device)
    if rank == 0:
        mode = _read_control_mode()
        code = mode_names.index(mode) if mode in mode_names[:2] else 2
        mode_code.fill_(code)
    if dist.is_initialized():
        dist.broadcast(mode_code, src=0, group=get_tp_group().device_group)
    return mode_names[int(mode_code.cpu())]


def _validate_restored_residency(
    methods: list["KTEPWrapperMethod"],
    changes: list[tuple["KTEPWrapperMethod", int, int]],
) -> None:
    for method in methods:
        initial = _INITIAL_RESIDENT[int(method.kt_config.layer_idx)]
        expected_mask = torch.zeros_like(method.gpu_experts_mask)
        expected_mask[initial.to(torch.int64)] = True
        expected_logical_to_gpu = torch.full_like(method.logical_to_gpu_index, -1)
        expected_logical_to_gpu[initial.to(torch.int64)] = torch.arange(
            len(initial),
            dtype=expected_logical_to_gpu.dtype,
            device=expected_logical_to_gpu.device,
        )
        metadata = (
            ("gpu_index_to_logical", method.gpu_index_to_logical, initial),
            ("gpu_experts_mask", method.gpu_experts_mask, expected_mask),
            (
                "logical_to_gpu_index",
                method.logical_to_gpu_index,
                expected_logical_to_gpu,
            ),
            (
                "gpu_experts_mask_cuda",
                method.gpu_experts_mask_cuda.cpu(),
                expected_mask,
            ),
            (
                "logical_to_gpu_index_cuda",
                method.logical_to_gpu_index_cuda.cpu(),
                expected_logical_to_gpu,
            ),
        )
        for name, actual, expected in metadata:
            if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
                raise RuntimeError(
                    "KT decode-hot static restore metadata mismatch: "
                    f"layer={method.kt_config.layer_idx}, field={name}"
                )
        if method.wrapper is not None and not torch.equal(
            method.wrapper.gpu_experts_mask.detach().cpu(), expected_mask.cpu()
        ):
            raise RuntimeError(
                "KT decode-hot static restore wrapper mask mismatch: "
                f"layer={method.kt_config.layer_idx}"
            )

    for method, slot, _ in changes:
        layer_idx = int(method.kt_config.layer_idx)
        w13_backup, w2_backup = _INITIAL_WEIGHT_BACKUP[layer_idx]
        layer = method._decode_hot_layer
        if not torch.equal(layer.w13_weight.data[slot].cpu(), w13_backup[slot]):
            raise RuntimeError(
                "KT decode-hot static restore w13 mismatch: "
                f"layer={layer_idx}, slot={slot}"
            )
        if not torch.equal(layer.w2_weight.data[slot].cpu(), w2_backup[slot]):
            raise RuntimeError(
                "KT decode-hot static restore w2 mismatch: "
                f"layer={layer_idx}, slot={slot}"
            )


def _restore_initial_residency(methods: list["KTEPWrapperMethod"]) -> None:
    """Restore startup expert placement before a same-process static A/B run."""
    global _UPDATE_STEP
    rank = get_tensor_model_parallel_rank()
    device = methods[0]._decode_hot_expert_counts.device
    changes: list[tuple["KTEPWrapperMethod", int, int]] = []
    for method in methods:
        initial = _INITIAL_RESIDENT[int(method.kt_config.layer_idx)]
        for slot, candidate_tensor in enumerate(initial):
            candidate = int(candidate_tensor)
            if int(method.gpu_index_to_logical[slot]) != candidate:
                changes.append((method, slot, candidate))
    if not changes:
        _LAYER_STATES.clear()
        _UPDATE_STEP = 0
        return

    started = time.perf_counter()
    torch.cuda.synchronize(device)
    with torch.no_grad():
        for method, slot, candidate in changes:
            layer_idx = int(method.kt_config.layer_idx)
            w13_backup, w2_backup = _INITIAL_WEIGHT_BACKUP[layer_idx]
            layer = method._decode_hot_layer
            layer.w13_weight.data[slot].copy_(w13_backup[slot])
            layer.w2_weight.data[slot].copy_(w2_backup[slot])
            _update_mapping(method, slot, candidate)
    torch.cuda.synchronize(device)
    _validate_restored_residency(methods, changes)
    _LAYER_STATES.clear()
    _UPDATE_STEP = 0
    update_ms = (time.perf_counter() - started) * 1000.0
    if rank == 0:
        logger.info(
            "[kt-decode-hot] restored and exactly validated static residency: "
            "slots=%d stage=%.2fms total=%.2fms",
            len(changes),
            0.0,
            update_ms,
        )
        _write_profile(
            {
                "mode": "static-reset",
                "slots": len(changes),
                "stage_ms_rank0": 0.0,
                "update_ms": update_ms,
                "h2d_bytes_per_rank": len(changes)
                * _resident_slot_bytes(
                    _INITIAL_WEIGHT_BACKUP[
                        int(changes[0][0].kt_config.layer_idx)
                    ]
                ),
            }
        )


def maybe_update_kt_decode_hot_experts() -> None:
    """Promote hot experts after one speculative target verification."""
    global _UPDATE_STEP
    if not _REGISTERED_LAYERS:
        return

    methods = [method for _, method in sorted(_REGISTERED_LAYERS.items())]
    if any(method._decode_hot_expert_counts is None for method in methods):
        return
    rank = get_tensor_model_parallel_rank()
    device = methods[0]._decode_hot_expert_counts.device
    mode = _synchronized_control_mode(device)
    _ensure_initial_weight_backup(methods)
    if mode == "static":
        _restore_initial_residency(methods)
        return
    if mode != "dynamic":
        if rank == 0:
            logger.warning(
                "[kt-decode-hot] unknown control mode %r; expected static or dynamic",
                mode,
            )
        return
    max_promotions = int(methods[0].kt_config.kt_decode_hot_max_promotions)
    _UPDATE_STEP += 1
    started = time.perf_counter()

    decisions = torch.full(
        (len(methods), max_promotions, 2),
        -1,
        dtype=torch.int64,
        device=device,
    )
    promotion_metadata: list[_Promotion] = []
    total_assignments = 0
    resident_assignments_before = 0
    resident_assignments_after = 0
    if rank == 0:
        refresh_interval = int(methods[0].kt_config.kt_decode_hot_refresh_interval)
        allow_promotion = _is_promotion_step(
            _UPDATE_STEP,
            methods[0].num_gpu_experts,
            max_promotions,
            refresh_interval,
        )
        counts_by_layer = torch.stack(
            [method._decode_hot_expert_counts for method in methods]
        ).cpu()
        for layer_position, (method, counts) in enumerate(
            zip(methods, counts_by_layer)
        ):
            before_resident = method.gpu_index_to_logical.to(torch.int64)
            total_assignments += int(counts.sum())
            resident_assignments_before += int(counts[before_resident].sum())
            promotions, after_resident = _select_layer_promotions(
                method, counts, allow_promotion=allow_promotion
            )
            resident_assignments_after += int(counts[after_resident].sum())
            for promotion_position, promotion in enumerate(promotions):
                decisions[layer_position, promotion_position, 0] = promotion.slot
                decisions[layer_position, promotion_position, 1] = promotion.candidate
                promotion_metadata.append(promotion)

    if dist.is_initialized():
        dist.broadcast(decisions, src=0, group=get_tp_group().device_group)
    decisions_cpu = decisions.cpu()

    source = None
    stage_ms = 0.0
    applied: list[tuple["KTEPWrapperMethod", int, int]] = []
    for layer_position, method in enumerate(methods):
        for promotion_position in range(max_promotions):
            slot = int(decisions_cpu[layer_position, promotion_position, 0])
            candidate = int(decisions_cpu[layer_position, promotion_position, 1])
            if slot < 0:
                continue
            victim = int(method.gpu_index_to_logical[slot])
            promotion = _Promotion(
                layer_idx=int(method.kt_config.layer_idx),
                slot=slot,
                victim=victim,
                candidate=candidate,
                current_tokens=0,
                candidate_score=0.0,
                victim_score=0.0,
            )
            source = source or _get_expert_source(method)
            _validate_raw_source(source, methods)
            stage_ms += source.stage_and_enqueue(method, promotion)
            applied.append((method, slot, candidate))

    if source is not None:
        torch.cuda.current_stream(device).wait_stream(source.copy_stream)
        for method, slot, candidate in applied:
            _update_mapping(method, slot, candidate)
        torch.cuda.synchronize(device)

    update_ms = (time.perf_counter() - started) * 1000.0
    if rank == 0:
        before_coverage = (
            resident_assignments_before / total_assignments
            if total_assignments
            else 0.0
        )
        after_coverage = (
            resident_assignments_after / total_assignments
            if total_assignments
            else 0.0
        )
        record = {
            "mode": "dynamic",
            "step": _UPDATE_STEP,
            "layers": len(methods),
            "assignments": total_assignments,
            "resident_coverage_before": before_coverage,
            "resident_coverage_after": after_coverage,
            "promotions": len(promotion_metadata),
            "stage_ms_rank0": stage_ms,
            "update_ms": update_ms,
            "h2d_bytes_per_rank": (
                len(promotion_metadata) * source.bytes_per_promotion
                if source is not None
                else 0
            ),
            "changes": [promotion.__dict__ for promotion in promotion_metadata],
        }
        _write_profile(record)
        if promotion_metadata or _UPDATE_STEP <= 4 or _UPDATE_STEP % 16 == 0:
            changes = ",".join(
                f"L{item.layer_idx}:{item.victim}->{item.candidate}({item.current_tokens})"
                for item in promotion_metadata[:8]
            )
            logger.info(
                "[kt-decode-hot] step=%d promotions=%d coverage=%.3f->%.3f "
                "stage=%.2fms total=%.2fms changes=%s",
                _UPDATE_STEP,
                len(promotion_metadata),
                before_coverage,
                after_coverage,
                stage_ms,
                update_ms,
                changes,
            )
