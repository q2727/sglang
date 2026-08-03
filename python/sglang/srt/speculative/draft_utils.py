from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import (
    cpu_has_amx_support,
    is_blackwell,
    is_cpu,
    is_hip,
    is_musa,
    is_npu,
)


class DraftBackendFactory:
    def __init__(
        self,
        server_args: ServerArgs,
        draft_model_runner,
        topk: int,
        speculative_num_steps: int,
        seed_dsa_topk_from_draft_extend: bool = False,
    ):
        self.server_args = server_args
        self.draft_model_runner = draft_model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.seed_dsa_topk_from_draft_extend = seed_dsa_topk_from_draft_extend
        self.draft_attn_backend = server_args.speculative_draft_attention_backend

    def _create_backend(
        self, backend_name: str, backend_map: dict, error_template: str
    ):
        backend_type = (
            self.draft_attn_backend
            if self.draft_attn_backend
            else getattr(self.server_args, backend_name)
        )
        if backend_type is None:
            backend_type = self.server_args.attention_backend

        if backend_type not in backend_map:
            raise ValueError(error_template.format(backend_type=backend_type))

        return backend_map[backend_type]()

    def create_decode_backend(self):
        # No multi-step draft backend for steps=0 (nospec) or steps=1.
        if self.speculative_num_steps <= 1:
            return None

        backend_map = {
            "flashinfer": self._create_flashinfer_decode_backend,
            "triton": self._create_triton_decode_backend,
            "intel_amx": self._create_intel_amx_decode_backend,
            "aiter": self._create_aiter_decode_backend,
            "fa3": self._create_fa3_decode_backend,
            "hybrid_linear_attn": self._create_hybrid_linear_attn_decode_backend,
            "flashmla": self._create_flashmla_decode_backend,
            "trtllm_mha": self._create_trtllm_mha_decode_backend,
            "trtllm_mla": self._create_trtllm_mla_decode_backend,
            "cutedsl_mla": self._create_cutedsl_mla_decode_backend,
            "tokenspeed_mla": self._create_tokenspeed_mla_decode_backend,
            "dsa": self._create_dsa_decode_backend,
            "nsa": self._create_dsa_decode_backend,  # Deprecated alias for "dsa"
            "ascend": self._create_ascend_decode_backend,
            "fa4": self._create_fa4_decode_backend,
            "dsv4": self._create_dsv4_decode_backend,
        }

        return self._create_backend(
            "decode_attention_backend",
            backend_map,
            "EAGLE is not supported in decode attention backend {backend_type}",
        )

    def create_draft_extend_backend(self):
        backend_map = {
            "flashinfer": self._create_flashinfer_prefill_backend,
            "triton": self._create_triton_prefill_backend,
            "intel_amx": self._create_intel_amx_prefill_backend,
            "aiter": self._create_aiter_prefill_backend,
            "fa3": self._create_fa3_prefill_backend,
            "hybrid_linear_attn": self._create_hybrid_linear_attn_prefill_backend,
            "flashmla": self._create_flashmla_prefill_backend,
            "trtllm_mha": self._create_trtllm_mha_prefill_backend,
            "trtllm_mla": self._create_trtllm_mla_prefill_backend,
            # cute-dsl MLA only supports decode; draft-extend falls back to trtllm-gen.
            "cutedsl_mla": self._create_trtllm_mla_prefill_backend,
            "tokenspeed_mla": self._create_tokenspeed_mla_prefill_backend,
            "dsa": self._create_dsa_prefill_backend,
            "nsa": self._create_dsa_prefill_backend,  # Deprecated alias for "dsa"
            "ascend": self._create_ascend_prefill_backend,
            "fa4": self._create_fa4_prefill_backend,
            "dsv4": self._create_dsv4_prefill_backend,
        }
        backend_name = (
            "decode_attention_backend"
            if self.server_args.speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        return self._create_backend(
            backend_name,
            backend_map,
            "EAGLE is not supported in attention backend {backend_type}",
        )

    def _create_dsa_decode_backend(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnMultiStepBackend,
        )

        return DeepseekSparseAttnMultiStepBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
        )

    def _create_dsa_prefill_backend(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        return DeepseekSparseAttnBackend(
            self.draft_model_runner,
            skip_prefill=False,
            seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
        )

    def _create_flashinfer_decode_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            return FlashInferMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            return FlashInferMLAMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )

    def _create_triton_decode_backend(self):
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )

        return TritonMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_intel_amx_decode_backend(self):
        from sglang.srt.layers.attention.intel_amx_backend import (
            IntelAMXMultiStepDraftBackend,
        )

        return IntelAMXMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_hybrid_linear_attn_decode_backend(self):
        if is_cpu() and cpu_has_amx_support():
            return self._create_intel_amx_decode_backend()
        if is_blackwell():
            return self._create_triton_decode_backend()
        return self._create_fa3_decode_backend()

    def _create_hybrid_linear_attn_prefill_backend(self):
        if is_cpu() and cpu_has_amx_support():
            return self._create_intel_amx_prefill_backend()
        if is_blackwell():
            return self._create_triton_prefill_backend()
        return self._create_fa3_prefill_backend()

    def _create_aiter_decode_backend(self):
        from sglang.srt.layers.attention.aiter_backend import AiterMultiStepDraftBackend

        return AiterMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_fa_decode_backend(self, fa_impl_ver: int = 3):
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionMultiStepBackend,
            )
        else:
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionMultiStepBackend as FlashAttentionMultiStepBackend,
            )

        return FlashAttentionMultiStepBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            fa_impl_ver=fa_impl_ver,
        )

    def _create_fa3_decode_backend(self):
        return self._create_fa_decode_backend(fa_impl_ver=3)

    def _create_fa4_decode_backend(self):
        return self._create_fa_decode_backend(fa_impl_ver=4)

    def _create_flashmla_decode_backend(self):
        from sglang.srt.layers.attention.flashmla_backend import (
            FlashMLAMultiStepDraftBackend,
        )

        return FlashMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mha_decode_backend(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )

        return TRTLLMHAAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mla_decode_backend(self, backend: str = "trtllm-gen"):
        if not self.draft_model_runner.use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLAMultiStepDraftBackend,
        )

        return TRTLLMMLAMultiStepDraftBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            backend=backend,
        )

    def _create_cutedsl_mla_decode_backend(self):
        return self._create_trtllm_mla_decode_backend(backend="cute-dsl")

    def _create_tokenspeed_mla_decode_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLAMultiStepDraftBackend,
        )

        return TokenspeedMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_ascend_decode_backend(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnMultiStepDraftBackend,
        )

        return AscendAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_dsv4_decode_backend(self):
        # Decode here is the EAGLE multi-step draft decode path.
        if is_npu():
            from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
                DeepseekV4AscendMultiStepDraftBackend,
            )

            return DeepseekV4AscendMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        elif is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4MultiStepBackend,
            )
        else:
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4MultiStepBackend,
            )

        return DeepseekV4MultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_flashinfer_prefill_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            return FlashInferAttnBackend(self.draft_model_runner, skip_prefill=False)
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            return FlashInferMLAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_triton_prefill_backend(self):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return TritonAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_intel_amx_prefill_backend(self):
        from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend

        return IntelAMXAttnBackend(self.draft_model_runner)

    def _create_aiter_prefill_backend(self):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        return AiterAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_fa_prefill_backend(self, fa_impl_ver: int = 3):
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )
        else:
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionBackend as FlashAttentionBackend,
            )
        return FlashAttentionBackend(
            self.draft_model_runner, skip_prefill=False, fa_impl_ver=fa_impl_ver
        )

    def _create_fa3_prefill_backend(self):
        return self._create_fa_prefill_backend(fa_impl_ver=3)

    def _create_fa4_prefill_backend(self):
        return self._create_fa_prefill_backend(fa_impl_ver=4)

    def _create_trtllm_mha_prefill_backend(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        return TRTLLMHAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_trtllm_mla_prefill_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        return TRTLLMMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_tokenspeed_mla_prefill_backend(self):
        if not self.draft_model_runner.use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        return TokenspeedMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_ascend_prefill_backend(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        return AscendAttnBackend(self.draft_model_runner)

    def _create_flashmla_prefill_backend(self):
        from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

        return FlashMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_dsv4_prefill_backend(self):
        # On NPU the "dsv4" backend resolves to the Ascend V4 subclass; its
        # draft-extend path uses the registered DSV4 prefill backend.
        if is_npu():
            from sglang.srt.layers.attention.attention_registry import (
                ATTENTION_BACKENDS,
            )

            return ATTENTION_BACKENDS["dsv4"](self.draft_model_runner)
        elif is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4HipRadixBackend,
            )

            return DeepseekV4HipRadixBackend(
                self.draft_model_runner, skip_prefill=False
            )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        return DeepseekV4AttnBackend(self.draft_model_runner, skip_prefill=False)


class HybridMultiStepDraftBackend:
    """Multi-step draft-decode backend for a draft model that owns
    linear-attention layers (a hybrid STANDALONE draft).

    ``EagleDraftWorker.draft_forward`` runs draft step ``i`` under a
    ForwardContext carrying ``attn_backends[i]``, and a hybrid model resolves
    EVERY layer -- recurrent ones included -- through that one context
    backend. So each per-step entry must be a complete
    :class:`HybridLinearAttnBackend`, not the pure-attention step backend.

    This composite leaves the pure-attention multi-step backend untouched
    (several backend families index into their own ``attn_backends`` list for
    kv-split / metadata internals, so its entries must stay pure) and exposes
    a parallel list of per-step wrappers. All steps share ONE recurrent
    sidecar: a chain step is a single-token decode that advances the row's
    state in place, so the recurrent metadata (per-row state slots) is
    step-invariant.
    """

    def __init__(
        self,
        *,
        full_multi_step_backend,
        linear_attn_backend,
        full_attn_layers: list[int],
    ):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        self.full_multi_step_backend = full_multi_step_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backends = [
            HybridLinearAttnBackend(step_backend, linear_attn_backend, full_attn_layers)
            for step_backend in full_multi_step_backend.attn_backends
        ]
        self.needs_cpu_seq_lens = (
            full_multi_step_backend.needs_cpu_seq_lens
            or linear_attn_backend.needs_cpu_seq_lens
        )
        self.max_context_len = full_multi_step_backend.max_context_len

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.full_multi_step_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        self.linear_attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.full_multi_step_backend.init_forward_metadata(forward_batch)
        self.linear_attn_backend.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.full_multi_step_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )
        self.linear_attn_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.full_multi_step_backend.init_forward_metadata_in_graph(forward_batch)
        self.linear_attn_backend.init_forward_metadata_in_graph(forward_batch)

    # No on_after_cuda_graph_warmup: the multi-step draft backends don't define
    # one (the draft graph runner probes for it), and the recurrent sidecar
    # inherits the base no-op -- it keeps no warmup-dirtied state to undo.


def resolve_hybrid_draft_full_attn_layers(draft_model_runner) -> Optional[list[int]]:
    """Full-attention layer ids of a draft model that ALSO has recurrent
    (linear-attention) layers; ``None`` when the draft has no recurrent plane.

    ``None`` is the historical case and every historical path stays on its old
    code: pure-attention draft models, and MTP / NextN heads. The head case
    needs BOTH tests -- a head grafted onto a hybrid trunk still gets a hybrid
    wrapper from the backend registry, but with the single grafted block
    declared full-attention, so its recurrent plane is never reached.
    """
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.speculative.spec_info import draft_worker_runs_complete_model

    if not draft_worker_runs_complete_model(
        is_draft_worker=draft_model_runner.is_draft_worker,
        spec_algorithm=draft_model_runner.spec_algorithm,
    ):
        return None
    main_backend = draft_model_runner.attn_backend
    if not isinstance(main_backend, HybridLinearAttnBackend):
        return None
    return main_backend.full_attn_layers


def build_hybrid_draft_decode_backend(
    *,
    full_multi_step_backend,
    draft_model_runner,
    full_attn_layers: list[int],
):
    """Wrap the pure-attention multi-step draft-decode backend for a hybrid draft."""
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

    return HybridMultiStepDraftBackend(
        full_multi_step_backend=full_multi_step_backend,
        # A fresh sidecar: the draft-extend plane sizes its own metadata for
        # the W-token verify window, this one for single-token chain steps.
        linear_attn_backend=GDNAttnBackend(draft_model_runner),
        full_attn_layers=full_attn_layers,
    )


def build_hybrid_draft_extend_backend(
    *,
    full_attn_backend,
    draft_model_runner,
    full_attn_layers: list[int],
    num_draft_tokens: int,
    pad_state_slot: torch.Tensor,
):
    """Wrap the pure-attention draft-extend backend for a hybrid draft.

    ``pad_state_slot`` is a PHYSICAL state slot the sidecar scans every pad
    segment of the fixed-width extend window through; see
    ``GDNAttnBackend.draft_extend_v2_pad_state_slot``. Setting it is also what
    turns the recurrent plane on for DRAFT_EXTEND_V2 forwards
    (``serves_draft_extend_v2``).
    """
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

    gdn_sidecar = GDNAttnBackend(draft_model_runner)
    gdn_sidecar.draft_extend_v2_num_tokens_per_req = num_draft_tokens
    gdn_sidecar.draft_extend_v2_pad_state_slot = int(pad_state_slot.item())
    return HybridLinearAttnBackend(full_attn_backend, gdn_sidecar, full_attn_layers)
