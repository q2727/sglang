"""Equivalence check for GDN DRAFT_EXTEND_V2 per-node recurrent-state capture.

What it proves
--------------
``GDNAttnBackend.alloc_draft_extend_v2_node_states`` lets ONE fixed-width extend
row report the recurrent state after each of its last ``num_nodes`` tokens. The
thing it replaces is the "glue triangle": ``num_nodes`` separate extend rows,
row ``g`` scanning only the first ``delta + g`` tokens from a forked copy of the
seat state, so that row ``g``'s ROW-END state is node ``g``'s state.

So the check is: run both, compare.

* reference -- ``num_nodes`` rows, true lengths ``delta .. delta + K``, each
  from an identical fork of one real mid-sequence state, capture OFF (today's
  code path, byte for byte). Node ``g`` = the state row ``g`` left in its slot.
* under test -- ONE row of true length ``delta + K`` from the same fork,
  capture ON. Node ``g`` = node slot ``g`` of the capture buffers.
* also compared: the per-position GDN layer OUTPUT. Row ``g``'s real positions
  are a prefix of the single row's, and the drafter reads node ``g``'s logits at
  row ``g``'s last real position, so equality there is the (whole) reason one
  row can stand in for the triangle on the logits side too.

Everything is real: real weights, real GDN layer count / head dims, and inputs
captured from a real prefill of a real prompt (the per-layer ``mixed_qkv / a /
b`` the model actually produced, plus the mamba state that prefill left behind).
Both sides are fed the SAME per-position values, so any nonzero diff is the
capability's own error and nothing else -- both sides run the same recurrent
kernel over the same fp32 op order, so the expected answer is EXACTLY zero, not
"within bf16 tolerance".

Run (single GPU, ~1 min):

    PYTHONPATH=python CUDA_VISIBLE_DEVICES=5 HF_HOME=/cluster-storage \
        python3 -m sglang.test.gdn_node_state_equivalence \
        --model-path Qwen/Qwen3.5-0.8B
"""

import argparse
import logging
from array import array

import torch

from sglang.benchmark.one_batch import extend, load_model
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


def _slot_layout(*, src_slot: int, num_nodes: int, num_pool_slots: int) -> tuple:
    """Physical state slots this check owns: the reference rows' forks, the row
    under test, and the junk slot every pad half-row scans. Taken next to the
    prefill's own slot -- nothing else runs in the process once the prefill is
    done, so raw ids keep the check independent of allocator bookkeeping."""
    ref_slots = [src_slot + 1 + node for node in range(num_nodes)]
    test_slot = src_slot + 1 + num_nodes
    junk_slot = src_slot + 2 + num_nodes
    assert junk_slot < num_pool_slots, (
        f"mamba pool has {num_pool_slots} slots, need {junk_slot + 1} past "
        f"{src_slot=} -- raise --mem-fraction-static"
    )
    return ref_slots, test_slot, junk_slot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt-len", type=int, default=96)
    parser.add_argument("--mem-fraction-static", type=float, default=0.6)
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[3, 5],
        help="K values (chain length); num_nodes = K + 1, W = 2K + 1.",
    )
    parser.add_argument(
        "--delta",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Committed-token counts prepended to the chain.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Max allowed abs diff; 0 demands bit-exactness (the expectation).",
    )
    parser.add_argument("--per-layer", action="store_true", help="Print every layer.")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Additionally capture the capture-on forward in a CUDA graph and "
        "replay it at every delta (the graph is captured at the first one).",
    )
    parser.add_argument(
        "--node-shift",
        type=int,
        default=0,
        help="Negative control: compare node j against reference node j+shift, "
        "which must FAIL (proves the comparison discriminates).",
    )
    return parser.parse_args()


def _capture_prefill_activations(*, model_runner, prompt_len: int, vocab_size: int):
    """Prefill one real prompt and return, per GDN layer, the ``(mixed_qkv, a,
    b)`` it saw, plus the layer objects and the request's mamba slot."""
    gdn_layers = sorted(
        (
            module
            for module in model_runner.model.modules()
            if isinstance(module, RadixLinearAttention)
        ),
        key=lambda module: module.layer_id,
    )
    assert gdn_layers, "no RadixLinearAttention layers found -- not a hybrid model?"

    activations: dict[int, tuple] = {}
    handles = []
    for layer in gdn_layers:

        def pre_hook(module, args, kwargs):
            # RadixLinearAttention.forward(forward_batch, mixed_qkv=, a=, b=)
            activations[module.layer_id] = (
                kwargs["mixed_qkv"].detach().clone(),
                kwargs["a"].detach().clone(),
                kwargs["b"].detach().clone(),
            )

        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))

    generator = torch.Generator().manual_seed(1234)
    tokens = torch.randint(
        0, min(vocab_size, 30000), (prompt_len,), generator=generator
    ).tolist()
    req = Req(
        rid="gdn-node-state-check",
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    req.full_untruncated_fill_ids = req.origin_input_ids
    req.logprob_start_len = -1
    req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))
    _next_tokens, _logits, batch = extend([req], model_runner)
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()

    assert len(activations) == len(gdn_layers), (
        f"captured {len(activations)} of {len(gdn_layers)} GDN layers -- the "
        "prefill did not run every linear layer"
    )
    # req.mamba_pool_idx is a VIRTUAL slot id; the kernels (and this check's
    # direct slot reads) work in physical ids.
    src_slot = model_runner.req_to_token_pool.translate_mamba_indices(
        torch.tensor([batch.reqs[0].mamba_pool_idx], device=model_runner.device)
    )
    return gdn_layers, activations, int(src_slot[0].item())


def _row_gather(source: torch.Tensor, *, true_lens: list[int], width: int):
    """Pack ``[L, ...]`` per-position values into ``[rows * width, ...]``: row
    ``g`` is ``source[:true_lens[g]]`` then repeats of its last real position --
    the same padded-window layout the drafter's fused extend feeds."""
    index: list[int] = []
    for true_len in true_lens:
        index.extend(list(range(true_len)) + [true_len - 1] * (width - true_len))
    return source[torch.tensor(index, dtype=torch.int64, device=source.device)]


def _metadata(
    *,
    true_lens: list[int],
    width: int,
    slots: list[int],
    junk_slot: int,
    device: str,
) -> ForwardMetadata:
    """The doubled half-row DRAFT_EXTEND_V2 metadata: real half of row ``r``
    scans ``[r * W, r * W + true_len)`` from ``slots[r]``, its pad half scans
    the junk slot. Mirrors ``_draft_extend_v2_eager_metadata``."""
    rows = len(true_lens)
    row_starts = torch.arange(
        0, (rows + 1) * width, width, dtype=torch.int32, device=device
    )
    cu_seqlens = torch.empty((2 * rows + 1,), dtype=torch.int32, device=device)
    cu_seqlens[0::2] = row_starts
    cu_seqlens[1::2] = row_starts[:rows] + torch.tensor(
        true_lens, dtype=torch.int32, device=device
    )
    state_indices = torch.full((2 * rows,), junk_slot, dtype=torch.int32, device=device)
    state_indices[0::2] = torch.tensor(slots, dtype=torch.int32, device=device)
    has_initial_states = torch.zeros((2 * rows,), dtype=torch.bool, device=device)
    has_initial_states[0::2] = True
    return ForwardMetadata(
        query_start_loc=cu_seqlens,
        mamba_cache_indices=state_indices,
        has_initial_states=has_initial_states,
    )


def _diff(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float, float]:
    """(max abs diff, max abs diff / max abs reference, max abs reference).

    The third value is the signal magnitude: a diff of 0 only means something
    if the reference is not itself 0.
    """
    lhs = lhs.to(torch.float32)
    rhs = rhs.to(torch.float32)
    abs_diff = (lhs - rhs).abs().max().item()
    scale = rhs.abs().max().item()
    return abs_diff, (abs_diff / scale if scale > 0 else 0.0), scale


def _triangle_reference(
    *,
    backend,
    layer,
    layer_cache,
    row_activations: tuple,
    fork: tuple,
    ref_slots: list[int],
    ref_true_lens: list[int],
    width: int,
) -> tuple:
    """Today's glue triangle for one layer: ``len(ref_slots)`` rows of the given
    true lengths, each from an identical fork of the seat state, capture OFF.
    Returns ``(row-end conv windows, row-end ssm states, per-position output)``.
    """
    conv_states = layer_cache.conv[0]
    ssm_states = layer_cache.temporal
    fork_conv, fork_ssm = fork
    for slot in ref_slots:
        conv_states[slot] = fork_conv
        ssm_states[slot] = fork_ssm
    out = backend._forward_draft_extend_v2(
        layer=layer,
        **{
            name: _row_gather(source, true_lens=ref_true_lens, width=width)
            for name, source in zip(("mixed_qkv", "a", "b"), row_activations)
        },
    )
    return conv_states[ref_slots].clone(), ssm_states[ref_slots].clone(), out


def _run_case(
    *,
    model_runner,
    gdn_layers,
    activations,
    src_slot: int,
    num_steps: int,
    delta: int,
    per_layer: bool,
    node_shift: int = 0,
) -> dict:
    """One (K, delta): triangle vs single row, reported per layer per node."""
    device = model_runner.device
    pool = model_runner.req_to_token_pool
    num_nodes = num_steps + 1
    width = 2 * num_steps + 1
    row_len = delta + num_steps
    assert row_len <= width, f"{row_len=} exceeds the padded window {width=}"

    # Reference rows: true lengths delta, delta+1, ..., delta+K (node g's state
    # is row g's row-end state). Under test: one row of the full length.
    ref_true_lens = [delta + g for g in range(num_nodes)]
    ref_slots, test_slot, junk_slot = _slot_layout(
        src_slot=src_slot,
        num_nodes=num_nodes,
        num_pool_slots=pool.mamba_pool.mamba_cache.temporal.shape[1],
    )

    ref_backend = GDNAttnBackend(model_runner)
    ref_backend.draft_extend_v2_num_tokens_per_req = width
    ref_backend.draft_extend_v2_pad_state_slot = junk_slot
    ref_backend.forward_metadata = _metadata(
        true_lens=ref_true_lens,
        width=width,
        slots=ref_slots,
        junk_slot=junk_slot,
        device=device,
    )

    cap_backend = GDNAttnBackend(model_runner)
    cap_backend.draft_extend_v2_num_tokens_per_req = width
    cap_backend.draft_extend_v2_pad_state_slot = junk_slot
    cap_backend.alloc_draft_extend_v2_node_states(max_rows=1, num_nodes=num_nodes)
    cap_backend.forward_metadata = _metadata(
        true_lens=[row_len],
        width=width,
        slots=[test_slot],
        junk_slot=junk_slot,
        device=device,
    )
    cap_backend._refresh_draft_extend_v2_node_rows(bs=1, num_real_rows=1)

    worst = {name: (0.0, 0.0, 0.0, "") for name in ("conv", "ssm", "out")}
    for layer in gdn_layers:
        layer_id = layer.layer_id
        mamba_idx = pool.mamba_map[layer_id]
        layer_cache = pool.mamba2_layer_cache(layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal

        # The row activations: the prefill's LAST row_len positions, so the
        # values are real magnitudes at real head layouts.
        mixed_qkv, a, b = (t[-row_len:] for t in activations[layer_id])
        fork_conv = conv_states[src_slot].clone()
        fork_ssm = ssm_states[src_slot].clone()

        ref_conv, ref_ssm, ref_out = _triangle_reference(
            backend=ref_backend,
            layer=layer,
            layer_cache=layer_cache,
            row_activations=(mixed_qkv, a, b),
            fork=(fork_conv, fork_ssm),
            ref_slots=ref_slots,
            ref_true_lens=ref_true_lens,
            width=width,
        )

        conv_states[test_slot] = fork_conv
        ssm_states[test_slot] = fork_ssm
        cap_out = cap_backend._forward_draft_extend_v2(
            layer=layer,
            mixed_qkv=_row_gather(mixed_qkv, true_lens=[row_len], width=width),
            a=_row_gather(a, true_lens=[row_len], width=width),
            b=_row_gather(b, true_lens=[row_len], width=width),
        )
        node_conv = cap_backend.draft_extend_v2_node_conv_windows[mamba_idx, 0]
        node_ssm = cap_backend.draft_extend_v2_node_ssm_states[mamba_idx, 0]

        # Anchor check, independent of the reference: the LAST node slot must be
        # the row-end state, i.e. exactly what the row wrote back to its own
        # state slot. Pins the tail-alignment anchor even if the whole reference
        # construction were wrong.
        anchor = {
            "ssm": _diff(node_ssm[-1], ssm_states[test_slot]),
            "conv": _diff(node_conv[-1], conv_states[test_slot]),
        }
        for name, (abs_diff, _rel, scale) in anchor.items():
            assert abs_diff == 0.0 and scale > 0.0, (
                f"last node slot is not the row-end {name} state: layer="
                f"{layer_id} max_abs={abs_diff:.3e} scale={scale:.3e}"
            )

        for node in range(num_nodes):
            end = ref_true_lens[node]
            ref_node = node + node_shift
            if not 0 <= ref_node < num_nodes:
                continue  # negative control: no neighbour to compare against
            checks = {
                "conv": (node_conv[node], ref_conv[ref_node]),
                "ssm": (node_ssm[node], ref_ssm[ref_node]),
                # Row `node`'s real positions are a prefix of the single row's;
                # its LAST one is where the drafter reads node logits.
                # (node_shift also shifts the POSITION here: row g's outputs at
                # SHARED positions are equal for every g by causality, so only a
                # position shift can give this comparison discriminating power.)
                "out": (
                    cap_out[0, node_shift : end + node_shift],
                    ref_out[0, ref_node * width : ref_node * width + end],
                ),
            }
            for name, (lhs, rhs) in checks.items():
                abs_diff, rel_diff, scale = _diff(lhs, rhs)
                previous = worst[name]
                worst[name] = (
                    max(abs_diff, previous[0]),
                    max(rel_diff, previous[1]),
                    max(scale, previous[2]),
                    (
                        f"layer={layer_id} node={node}"
                        if abs_diff > previous[0]
                        else previous[3]
                    ),
                )
                if per_layer:
                    logger.info(
                        "  K=%d delta=%d layer=%2d node=%d %-4s "
                        "max_abs=%.3e max_rel=%.3e ref_scale=%.3e",
                        num_steps,
                        delta,
                        layer_id,
                        node,
                        name,
                        abs_diff,
                        rel_diff,
                        scale,
                    )
    # Also make sure the capture wrote something at all (a silently skipped
    # store would otherwise "match" a zeroed reference), and that every
    # comparison ran on a non-trivial signal.
    assert cap_backend.draft_extend_v2_node_ssm_states.abs().sum().item() > 0
    for name, (_abs, _rel, scale, _where) in worst.items():
        assert scale > 0.0, f"{name} comparison ran on an all-zero reference"
    return worst


def _check_padding_row(
    *,
    model_runner,
    gdn_layers,
    activations,
    src_slot: int,
    num_steps: int,
    delta: int,
) -> None:
    """Pin the determinism invariant for rows OUTSIDE the batch's real rows.

    A replay whose bucket is wider than the real row count carries padding
    rows: an empty real half and a ``-1`` node row. Their buffer rows must stay
    at their zero fill on the ssm plane, must stay FINITE on the conv plane
    (which is written for every row, from written tensors -- never
    uninitialized graph memory), and must not perturb the real row.
    """
    device = model_runner.device
    pool = model_runner.req_to_token_pool
    num_nodes = num_steps + 1
    width = 2 * num_steps + 1
    row_len = delta + num_steps
    _ref_slots, test_slot, junk_slot = _slot_layout(
        src_slot=src_slot,
        num_nodes=num_nodes,
        num_pool_slots=pool.mamba_pool.mamba_cache.temporal.shape[1],
    )

    results = {}
    for rows, num_real_rows in ((1, 1), (2, 1)):
        backend = GDNAttnBackend(model_runner)
        backend.draft_extend_v2_num_tokens_per_req = width
        backend.draft_extend_v2_pad_state_slot = junk_slot
        backend.alloc_draft_extend_v2_node_states(max_rows=rows, num_nodes=num_nodes)
        # A padding row's real half is EMPTY (its odd cu_seqlens boundary equals
        # its row start), exactly what _draft_extend_v2_metadata builds.
        backend.forward_metadata = _metadata(
            true_lens=[row_len] + [0] * (rows - num_real_rows),
            width=width,
            slots=[test_slot] + [junk_slot] * (rows - num_real_rows),
            junk_slot=junk_slot,
            device=device,
        )
        backend._refresh_draft_extend_v2_node_rows(bs=rows, num_real_rows=num_real_rows)
        for layer in gdn_layers:
            layer_cache = pool.mamba2_layer_cache(layer.layer_id)
            layer_cache.conv[0][test_slot] = layer_cache.conv[0][src_slot]
            layer_cache.temporal[test_slot] = layer_cache.temporal[src_slot]
            # The runner zero-fills padding rows' input ids, so zeros here.
            row = [
                torch.cat(
                    [
                        _row_gather(
                            source[-row_len:], true_lens=[row_len], width=width
                        ),
                        torch.zeros(
                            ((rows - 1) * width, *source.shape[1:]),
                            dtype=source.dtype,
                            device=device,
                        ),
                    ]
                )
                for source in activations[layer.layer_id]
            ]
            backend._forward_draft_extend_v2(
                layer=layer, mixed_qkv=row[0], a=row[1], b=row[2]
            )
        results[rows] = (
            backend.draft_extend_v2_node_conv_windows.clone(),
            backend.draft_extend_v2_node_ssm_states.clone(),
        )

    padded_conv, padded_ssm = results[2]
    single_conv, single_ssm = results[1]
    assert padded_ssm[:, 1].abs().sum().item() == 0.0, (
        "a padding row's ssm node slots were written -- the recurrent kernel "
        "must skip them (node row == -1)"
    )
    assert torch.isfinite(padded_conv[:, 1]).all().item(), (
        "a padding row's conv node windows are not finite -- the gather read "
        "something nobody wrote"
    )
    for name, (lhs, rhs) in (
        ("ssm", (padded_ssm[:, 0], single_ssm[:, 0])),
        ("conv", (padded_conv[:, 0], single_conv[:, 0])),
    ):
        abs_diff, _rel, scale = _diff(lhs, rhs)
        assert abs_diff == 0.0 and scale > 0.0, (
            f"a padding row perturbed the real row's {name} node states: "
            f"max_abs={abs_diff:.3e} scale={scale:.3e}"
        )
    logger.info(
        "PADDING ROW K=%d delta=%d: padding ssm rows untouched (exact zero), "
        "padding conv rows finite, real row identical to the 1-row batch",
        num_steps,
        delta,
    )


def _run_graph_case(
    *,
    model_runner,
    gdn_layers,
    activations,
    src_slot: int,
    num_steps: int,
    deltas: list[int],
    tolerance: float,
) -> list[str]:
    """Capture the node-capture forward in ONE CUDA graph, then replay it for
    several deltas and check every replay against the eager triangle.

    This is the claim that matters for the drafter: the buffers and the write
    positions are shape constants, and the only data-dependent input (each row's
    true scan length) travels through a device tensor -- so a graph captured at
    one delta must replay correctly at another. Capture happens at ``deltas[0]``
    and at least one replay uses a DIFFERENT delta.
    """
    device = model_runner.device
    pool = model_runner.req_to_token_pool
    num_nodes = num_steps + 1
    width = 2 * num_steps + 1
    ref_slots, test_slot, junk_slot = _slot_layout(
        src_slot=src_slot,
        num_nodes=num_nodes,
        num_pool_slots=pool.mamba_pool.mamba_cache.temporal.shape[1],
    )

    ref_backend = GDNAttnBackend(model_runner)
    ref_backend.draft_extend_v2_num_tokens_per_req = width
    ref_backend.draft_extend_v2_pad_state_slot = junk_slot

    cap_backend = GDNAttnBackend(model_runner)
    cap_backend.draft_extend_v2_num_tokens_per_req = width
    cap_backend.draft_extend_v2_pad_state_slot = junk_slot
    cap_backend.alloc_draft_extend_v2_node_states(max_rows=1, num_nodes=num_nodes)
    # Static metadata, refreshed IN PLACE per replay -- exactly what the
    # backend's own replay prep does.
    cap_backend.forward_metadata = _metadata(
        true_lens=[deltas[0] + num_steps],
        width=width,
        slots=[test_slot],
        junk_slot=junk_slot,
        device=device,
    )
    cap_backend._refresh_draft_extend_v2_node_rows(bs=1, num_real_rows=1)
    cu_seqlens = cap_backend.forward_metadata.query_start_loc

    # Static per-layer input buffers (the graph binds these by pointer).
    static_inputs = {
        layer.layer_id: tuple(
            torch.zeros((width, *source.shape[1:]), dtype=source.dtype, device=device)
            for source in activations[layer.layer_id]
        )
        for layer in gdn_layers
    }
    forks = {}
    for layer in gdn_layers:
        layer_cache = pool.mamba2_layer_cache(layer.layer_id)
        forks[layer.layer_id] = (
            layer_cache.conv[0][src_slot].clone(),
            layer_cache.temporal[src_slot].clone(),
        )

    def load_row(delta: int) -> None:
        """Refresh every static input + the device true length for one delta."""
        row_len = delta + num_steps
        cu_seqlens[1] = row_len
        for layer in gdn_layers:
            row = [
                _row_gather(source[-row_len:], true_lens=[row_len], width=width)
                for source in activations[layer.layer_id]
            ]
            for buffer, value in zip(static_inputs[layer.layer_id], row):
                buffer.copy_(value)

    def reset_states() -> None:
        for layer in gdn_layers:
            layer_cache = pool.mamba2_layer_cache(layer.layer_id)
            fork_conv, fork_ssm = forks[layer.layer_id]
            layer_cache.conv[0][test_slot] = fork_conv
            layer_cache.temporal[test_slot] = fork_ssm

    def run_layers() -> dict:
        return {
            layer.layer_id: cap_backend._forward_draft_extend_v2(
                layer=layer,
                mixed_qkv=static_inputs[layer.layer_id][0],
                a=static_inputs[layer.layer_id][1],
                b=static_inputs[layer.layer_id][2],
            )
            for layer in gdn_layers
        }

    load_row(deltas[0])
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            reset_states()
            run_layers()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    reset_states()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run_layers()

    failures = []
    for delta in deltas:
        row_len = delta + num_steps
        ref_true_lens = [delta + node for node in range(num_nodes)]
        load_row(delta)
        reset_states()
        graph.replay()
        torch.cuda.synchronize()
        replay = {
            layer.layer_id: (
                cap_backend.draft_extend_v2_node_conv_windows[
                    pool.mamba_map[layer.layer_id], 0
                ].clone(),
                cap_backend.draft_extend_v2_node_ssm_states[
                    pool.mamba_map[layer.layer_id], 0
                ].clone(),
                graph_out[layer.layer_id].clone(),
            )
            for layer in gdn_layers
        }
        worst = {name: (0.0, 0.0, 0.0, "") for name in ("conv", "ssm", "out")}
        ref_backend.forward_metadata = _metadata(
            true_lens=ref_true_lens,
            width=width,
            slots=ref_slots,
            junk_slot=junk_slot,
            device=device,
        )
        for layer in gdn_layers:
            layer_id = layer.layer_id
            layer_cache = pool.mamba2_layer_cache(layer_id)
            row_activations = tuple(t[-row_len:] for t in activations[layer_id])
            ref_conv, ref_ssm, ref_out = _triangle_reference(
                backend=ref_backend,
                layer=layer,
                layer_cache=layer_cache,
                row_activations=row_activations,
                fork=forks[layer_id],
                ref_slots=ref_slots,
                ref_true_lens=ref_true_lens,
                width=width,
            )
            node_conv, node_ssm, cap_out = replay[layer_id]
            for node in range(num_nodes):
                end = ref_true_lens[node]
                checks = {
                    "conv": (node_conv[node], ref_conv[node]),
                    "ssm": (node_ssm[node], ref_ssm[node]),
                    "out": (
                        cap_out[0, :end],
                        ref_out[0, node * width : node * width + end],
                    ),
                }
                for name, (lhs, rhs) in checks.items():
                    abs_diff, rel_diff, scale = _diff(lhs, rhs)
                    previous = worst[name]
                    worst[name] = (
                        max(abs_diff, previous[0]),
                        max(rel_diff, previous[1]),
                        max(scale, previous[2]),
                        (
                            f"layer={layer_id} node={node}"
                            if abs_diff > previous[0]
                            else previous[3]
                        ),
                    )
        logger.info(
            "GRAPH K=%d captured_delta=%d replay_delta=%d | %s",
            num_steps,
            deltas[0],
            delta,
            "  ".join(
                f"{name}: max_abs={abs_diff:.3e} max_rel={rel_diff:.3e} "
                f"ref_scale={scale:.3e} @{where}"
                for name, (abs_diff, rel_diff, scale, where) in worst.items()
            ),
        )
        for name, (abs_diff, _rel, scale, where) in worst.items():
            assert scale > 0.0, f"{name} compared against an all-zero reference"
            if abs_diff > tolerance:
                failures.append(
                    f"GRAPH K={num_steps} replay_delta={delta} {name} "
                    f"max_abs={abs_diff:.3e} @{where}"
                )
    del graph
    return failures


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    server_args = ServerArgs(
        model_path=args.model_path,
        mem_fraction_static=args.mem_fraction_static,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        mamba_radix_cache_strategy="no_buffer",
        page_size=1,
    )
    port_args = PortArgs.init_new(server_args)
    bench_runner, _tokenizer = load_model(server_args, port_args, 0, 0)
    model_runner = bench_runner.torch_runner

    gdn_layers, activations, src_slot = _capture_prefill_activations(
        model_runner=model_runner,
        prompt_len=args.prompt_len,
        vocab_size=model_runner.model_config.vocab_size,
    )
    conv = model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
    temporal = model_runner.req_to_token_pool.mamba_pool.mamba_cache.temporal
    logger.info(
        "model=%s gdn_layers=%d conv=%s ssm=%s src_slot=%d",
        args.model_path,
        len(gdn_layers),
        tuple(conv.shape),
        tuple(temporal.shape),
        src_slot,
    )

    failures = []
    for num_steps in args.num_steps:
        for delta in args.delta:
            worst = _run_case(
                model_runner=model_runner,
                gdn_layers=gdn_layers,
                activations=activations,
                src_slot=src_slot,
                num_steps=num_steps,
                delta=delta,
                per_layer=args.per_layer,
                node_shift=args.node_shift,
            )
            logger.info(
                "K=%d delta=%d nodes=%d W=%d | %s",
                num_steps,
                delta,
                num_steps + 1,
                2 * num_steps + 1,
                "  ".join(
                    f"{name}: max_abs={abs_diff:.3e} max_rel={rel_diff:.3e} "
                    f"ref_scale={scale:.3e} @{where}"
                    for name, (abs_diff, rel_diff, scale, where) in worst.items()
                ),
            )
            for name, (abs_diff, _rel, _scale, where) in worst.items():
                if abs_diff > args.tolerance:
                    failures.append(
                        f"K={num_steps} delta={delta} {name} "
                        f"max_abs={abs_diff:.3e} @{where}"
                    )

    if not args.node_shift:
        for num_steps in args.num_steps:
            _check_padding_row(
                model_runner=model_runner,
                gdn_layers=gdn_layers,
                activations=activations,
                src_slot=src_slot,
                num_steps=num_steps,
                delta=args.delta[0],
            )

    if args.cuda_graph and not args.node_shift:
        for num_steps in args.num_steps:
            failures.extend(
                _run_graph_case(
                    model_runner=model_runner,
                    gdn_layers=gdn_layers,
                    activations=activations,
                    src_slot=src_slot,
                    num_steps=num_steps,
                    deltas=list(args.delta),
                    tolerance=args.tolerance,
                )
            )

    if args.node_shift:
        # Negative control: comparing node j against the NEIGHBOURING reference
        # node must blow up. Without it, an all-zero diff table could just mean
        # the check is comparing something to itself.
        if failures:
            logger.info(
                "NEGATIVE CONTROL OK: shifting the node index by %d breaks "
                "%d of the comparisons",
                args.node_shift,
                len(failures),
            )
            return
        raise SystemExit(
            f"NEGATIVE CONTROL FAILED: node_shift={args.node_shift} still "
            "matched -- the comparison has no discriminating power"
        )

    if failures:
        raise SystemExit("FAIL\n" + "\n".join(failures))
    logger.info("PASS: every layer / node matches the per-row reference exactly")


if __name__ == "__main__":
    main()
