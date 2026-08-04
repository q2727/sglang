"""Register Qwen3.5 CPU-MoE operations with SGLang two-batch overlap.

Upstream TBO strategies split GPU expert-parallel communication into stages.
KTransformers has a different asynchronous boundary: CPU routed-expert work is
submitted from ``apply_tbo_begin`` and collected by ``apply_tbo_finish``.  The
single yield between those operations lets the other microbatch run attention
on the GPU while the first microbatch consumes CPU memory bandwidth.
"""

from __future__ import annotations

import os
from typing import Type

from sglang.srt.batch_overlap.operations import YieldOperation
from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy


_REGISTERED = False


def prepare_qwen35_kt_tbo_forward_batch(forward_batch) -> None:
    """Build local TBO children without entering the DP/EP MLP-sync path.

    Upstream TBO obtains its split metadata while gathering DP-attention ranks.
    KT CPU offload uses ordinary tensor parallel attention, so that gather would
    duplicate/pad requests across TP ranks.  For eager decode, the local batch
    already contains everything needed to construct the two children.
    """

    if os.environ.get("SGLANG_KT_CPU_TBO") != "1":
        return
    if forward_batch.tbo_children is not None:
        return
    if not forward_batch.forward_mode.is_decode() or forward_batch.batch_size < 2:
        return

    from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
    from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
    from sglang.srt.layers.dp_attention import DpPaddingMode

    if not isinstance(forward_batch.attn_backend, TboAttnBackend):
        return

    # OperationsExecutor uses this field to configure optional DP buffers even
    # when no DP-attention gather is active.  SUM_LEN expresses the unpadded
    # local layout used by KT.
    if forward_batch.dp_padding_mode is None:
        forward_batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    forward_batch.tbo_split_seq_index = forward_batch.batch_size // 2
    forward_batch.global_forward_mode = forward_batch.forward_mode
    TboForwardBatchPreparer.prepare(batch=forward_batch, is_draft_worker=False)

    # ModelRunner initialized only the primary backend before the local split.
    # Reinitialize it once so both child HybridLinearAttnBackends receive their
    # own request-pool and Mamba-cache indices.
    forward_batch.attn_backend.init_forward_metadata(forward_batch)


def register_qwen35_kt_tbo(
    linear_layer_type: Type,
    attention_layer_type: Type,
) -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    original = OperationsStrategy.init_new_tbo
    supported_types = (linear_layer_type, attention_layer_type)

    # TboAttnBackend inherits the q/k/v-only base ``forward`` signature.  A
    # hybrid GDN model can also reach the primary backend without splitting
    # (for example batch-size-one warmup), in which case linear-attention
    # keyword arguments must be forwarded unchanged.
    from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

    if "forward" not in TboAttnBackend.__dict__:
        def forward(self, *args, **kwargs):
            return self.primary.forward(*args, **kwargs)

        TboAttnBackend.forward = forward

    from sglang.srt.batch_overlap import two_batch_overlap

    if not getattr(two_batch_overlap, "_kt_cpu_tbo_compat", False):
        original_compute_split = two_batch_overlap.compute_split_seq_index
        original_model_filter_inputs = two_batch_overlap._model_forward_filter_inputs

        def compute_split_seq_index(
            forward_mode,
            num_tokens,
            extend_lens,
            token_num_per_seq,
        ):
            # This reproduction targets the paper's steady-state decode path.
            # Qwen3.5's GDN extend metadata carries chunk state that cannot be
            # split by the generic TBO ForwardBatch helper.  Keep prefill and
            # speculative verification on the ordinary unsplit path.
            if not forward_mode.is_decode():
                return None
            num_sequences = num_tokens // token_num_per_seq
            if num_sequences < 2:
                return None
            return original_compute_split(
                forward_mode,
                num_tokens,
                extend_lens,
                token_num_per_seq,
            )

        two_batch_overlap.compute_split_seq_index = compute_split_seq_index

        def model_forward_filter_inputs(
            hidden_states,
            residual,
            positions,
            output_forward_batch,
            tbo_subbatch_index,
        ):
            if positions.ndim != 2:
                return original_model_filter_inputs(
                    hidden_states=hidden_states,
                    residual=residual,
                    positions=positions,
                    output_forward_batch=output_forward_batch,
                    tbo_subbatch_index=tbo_subbatch_index,
                )

            # Qwen3.5 multimodal RoPE positions are [3, num_tokens].  The
            # generic TBO helper assumes [num_tokens] and otherwise slices the
            # three RoPE axes instead of the token axis, causing the Triton
            # MRoPE kernel to read beyond the child tensor.
            output = original_model_filter_inputs(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions[0],
                output_forward_batch=output_forward_batch,
                tbo_subbatch_index=tbo_subbatch_index,
            )
            token_slice = slice(*output_forward_batch.tbo_parent_token_range)
            child_positions = positions[:, token_slice]
            assert child_positions.shape[1] == output_forward_batch.tbo_padded_len
            output["positions"] = child_positions
            return output

        two_batch_overlap._model_forward_filter_inputs = model_forward_filter_inputs

        preparer = two_batch_overlap.TboForwardBatchPreparer
        original_filter_batch = preparer.filter_batch.__func__

        def filter_batch(cls, batch, **kwargs):
            rids = getattr(batch, "rids", None)
            if rids is None:
                return original_filter_batch(cls, batch, **kwargs)
            mrope_positions = getattr(batch, "mrope_positions", None)

            # The fork carries stable request IDs in ForwardBatch; upstream TBO
            # predates this field and rejects all unknown non-None fields.
            batch.rids = None
            try:
                child = original_filter_batch(cls, batch, **kwargs)
            finally:
                batch.rids = rids
            child.rids = rids[
                kwargs["start_seq_index"] : kwargs["end_seq_index"]
            ]
            if mrope_positions is not None:
                child.mrope_positions = mrope_positions[
                    :,
                    kwargs["start_token_index"] : kwargs["end_token_index"],
                ]
            # Upstream pads each TBO child to attention-TP divisibility for
            # DP-attention reduce-scatter.  KT keeps ordinary TP attention;
            # padding a one-request child to two tokens breaks GDN's exact
            # ``[time, batch, heads, dim]`` reshape.
            child.tbo_padded_len = (
                kwargs["end_token_index"] - kwargs["start_token_index"]
            )
            return child

        preparer.filter_batch = classmethod(filter_batch)
        two_batch_overlap._kt_cpu_tbo_compat = True

    def init_new_tbo(layers, forward_mode):
        if layers and all(isinstance(layer, supported_types) for layer in layers):
            for layer in layers:
                quant_method = getattr(
                    getattr(getattr(layer, "mlp", None), "experts", None),
                    "quant_method",
                    None,
                )
                if not hasattr(quant_method, "apply_tbo_begin"):
                    return original(layers, forward_mode)

            return OperationsStrategy.concat(
                [
                    OperationsStrategy(
                        operations=[
                            layer.op_tbo_attention,
                            YieldOperation(),
                            layer.op_tbo_moe_begin,
                            YieldOperation(),
                            layer.op_tbo_moe_finish,
                            YieldOperation(),
                        ],
                        deep_gemm_num_sms=None,
                        tbo_delta_stages=1,
                    )
                    for layer in layers
                ]
            )
        return original(layers, forward_mode)

    OperationsStrategy.init_new_tbo = staticmethod(init_new_tbo)
    _REGISTERED = True
