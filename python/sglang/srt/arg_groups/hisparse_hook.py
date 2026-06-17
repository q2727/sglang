from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# Backend/dtype pairing: flashmla_sparse only takes BF16 KV;
# flashmla_kv only supports FP8 (it always reads KV as FP8 via
# is_fp8_kvcache=True, inline-quantizing BF16 would defeat HiSparse).
_HISPARSE_ALLOWED_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv"},
}


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def apply_hisparse_dsa_backend_defaults(
    server_args: ServerArgs,
    user_set_prefill: bool,
    user_set_decode: bool,
    kv_cache_dtype: str,
) -> bool:
    """Pick DSA backends for --enable-hisparse based on KV dtype.

    BF16 KV -> flashmla_sparse, FP8 KV -> flashmla_kv. Returns True if hisparse
    handled backend selection (caller should skip its own default logic).

    MiniMax-M3 models use their own --attention-backend fa4 and
    MiniMaxSparseAttnBackend, so this function returns False for M3.
    """
    if not server_args.enable_hisparse:
        return False

    from sglang.srt.configs.model_config import is_minimax_sparse

    hf_config = server_args.get_model_config().hf_config
    if is_minimax_sparse(hf_config):
        # MiniMax-M3 uses its own attention backend (fa4) and
        # MiniMaxSparseAttnBackend; do not set DSA backends.
        return False

    backend = _hisparse_default_backend(kv_cache_dtype)
    if not user_set_prefill:
        server_args.dsa_prefill_backend = backend
    if not user_set_decode:
        server_args.dsa_decode_backend = backend
    logger.warning(
        f"HiSparse enabled ({kv_cache_dtype}): using DSA backends "
        f"prefill={server_args.dsa_prefill_backend}, decode={server_args.dsa_decode_backend}."
    )
    return True


def _validate_hisparse_minimax_m3(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints specific to MiniMax-M3.

    First-phase M3 HiSparse restrictions:
    - BF16 only (no FP8).
    - Text-only (no multimodal).
    - No speculative decoding.
    - No PD disaggregation.
    - No MXFP8.
    - CUDA graph is warned about and should be disabled for now.
    """
    hf_config = server_args.get_model_config().hf_config

    # BF16 KV cache only in first phase
    if server_args.kv_cache_dtype not in ("bfloat16", "auto"):
        raise ValueError(
            "MiniMax-M3 HiSparse first phase only supports BF16 KV cache. "
            f"Got --kv-cache-dtype={server_args.kv_cache_dtype}. "
            "Please use --kv-cache-dtype=bfloat16 (or auto)."
        )

    # Text-only
    if getattr(hf_config, "model_type", None) != "text":
        logger.warning(
            "MiniMax-M3 HiSparse first phase is designed for text-only models. "
            "Multimodal support is not implemented yet."
        )

    # No speculative decoding
    if server_args.speculative_algorithm is not None:
        raise ValueError(
            "MiniMax-M3 HiSparse does not support speculative decoding in the "
            "first phase. Please remove --speculative-algorithm."
        )

    # No PD disaggregation
    if server_args.disaggregation_mode != "null":
        raise ValueError(
            "MiniMax-M3 HiSparse does not support PD disaggregation in the "
            "first phase. Please remove --disaggregation-mode=null."
        )

    # No CUDA graph (graph-safe swap-in not implemented yet)
    if not server_args.disable_cuda_graph:
        logger.warning(
            "MiniMax-M3 HiSparse: CUDA graph is not yet graph-safe for the "
            "host→hot block swap-in path. Forcing --disable-cuda-graph. "
            "Set --disable-cuda-graph explicitly to suppress this warning."
        )
        server_args.disable_cuda_graph = True


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""
    if not server_args.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
        is_minimax_sparse,
    )

    hf_config = server_args.get_model_config().hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_m3_hisparse = is_minimax_sparse(hf_config)
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse or is_m3_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5), DeepSeek V4, and MiniMax-M3 "
        "sparse models. "
    )

    assert (
        server_args.disable_radix_cache
    ), "Hierarchical sparse attention currently requires --disable-radix-cache."

    # MiniMax-M3 has its own validation path (BF16 only, no spec, no PD, etc.)
    if is_m3_hisparse:
        _validate_hisparse_minimax_m3(server_args)
        return

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_v4_hisparse:
        return

    if server_args.kv_cache_dtype not in ("bfloat16", "auto", "fp8_e4m3"):
        raise ValueError(
            f"HiSparse requires bfloat16 or fp8_e4m3 KV cache, "
            f"but got --kv-cache-dtype={server_args.kv_cache_dtype}. "
            f"Please use --kv-cache-dtype=bfloat16 or fp8_e4m3."
        )

    allowed_backends = _HISPARSE_ALLOWED_BACKENDS_BY_DTYPE.get(
        server_args.kv_cache_dtype, {"flashmla_sparse", "flashmla_kv"}
    )
    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        backend = getattr(server_args, attr)
        if backend is not None and backend not in allowed_backends:
            raise ValueError(
                f"HiSparse with --kv-cache-dtype={server_args.kv_cache_dtype} requires "
                f"--dsa-{label}-backend in {sorted(allowed_backends)}, "
                f"but got {backend}."
            )
