"""Microbenchmark for the decoupled drafter's enumeration round.

Drives ``EnumDraftEngine.draft_round`` directly against a real draft model --
no verifier process, no transport: each benchmark round applies a synthetic
commit delta (as the verifier would) and runs one enumeration round, which is
exactly the drafter's steady-state loop body.

Example (drafter defaults from the e2e setup)::

    python -m sglang.benchmark.decoupled_drafter_round \
        --model-path Qwen/Qwen3-0.6B --num-steps 5 --fanout 4 \
        --batch-size 1 --prompt-len 1024 --rounds 60 --profile
"""

import argparse
import logging
import os
import random
import time
from typing import Optional

import torch

from sglang.benchmark.one_batch import load_model
from sglang.srt.environ import envs
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.decoupled_draft_engine import EnumDraftEngine
from sglang.srt.speculative.decoupled_spec_io import DraftReqKey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--warmup-rounds", type=int, default=8)
    parser.add_argument(
        "--accept-len",
        type=int,
        default=2,
        help="Synthetic per-round accept length; commit delta = accept-len + 1.",
    )
    parser.add_argument(
        "--miss-every",
        type=int,
        default=0,
        help="Every Nth round commits a bonus outside the guesses (forces the "
        "slow path); 0 = only the unavoidable first-round miss.",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Forwarded to ServerArgs when set (else the server default).",
    )
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Forwarded to ServerArgs (bounds fa3's static page-table width).",
    )
    parser.add_argument(
        "--cuda-graph-backend-prefill",
        type=str,
        default=None,
        help="Forwarded to ServerArgs (e.g. 'full' / 'tc_piecewise').",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Per-phase breakdown via SGLANG_DEBUG_DECOUPLED_DRAFT_PROFILE "
        "(adds phase-boundary device syncs; wall time will be inflated).",
    )
    parser.add_argument(
        "--torch-profile-trace",
        type=str,
        default=None,
        help="Wrap 3 post-warmup rounds in torch.profiler and export a chrome "
        "trace to this path; also prints the top ops by CPU time.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Correctness mode: drive a fast-path engine and a slow-path "
        "engine with identical commits and diff their blocks each round.",
    )
    parser.add_argument(
        "--compare-graph",
        action="store_true",
        help="Variant of --compare where BOTH engines run the glue fast "
        "path and only the extend-graph differs (fast=graph, slow=eager "
        "fused): isolates graph-vs-eager divergence with matched semantics.",
    )
    parser.add_argument(
        "--effective-fanout",
        type=int,
        default=None,
        help="Pin the engine's effective fanout below --fanout (exercises "
        "the adaptive-width row subsetting; unused columns ship poisoned).",
    )
    return parser.parse_args()


class _VerifierSim:
    """Per-key emulation of the verifier's select/verify/commit protocol.

    Tracks the chain the verifier is "verifying" (== the engine's backbone
    next round) so hit commits exercise the engine's glue fast path exactly
    like the real mesh does.
    """

    def __init__(
        self, *, accept_len: int, fanout: int, rng: random.Random, vocab_size: int
    ) -> None:
        self.accept_len = accept_len
        self.fanout = fanout
        self.rng = rng
        self.vocab_size = vocab_size
        self.chain: Optional[list[int]] = None  # None => last round missed

    @staticmethod
    def _pick_live_f(row: list) -> int:
        # Prefer a non-top guess (exercises the f-index matching) among the
        # live columns; dead cells (adaptive fanout, dead-guess exclusion at
        # width 1) are poisoned with -1 and can never be a real bonus.
        live = [f for f, unit in enumerate(row) if unit[0] != -1]
        return live[1] if len(live) > 1 else live[0]

    def next_delta(self, units: list, *, force_miss: bool) -> list[int]:
        """units = last block's [K+1][F][K+1] host list; returns the commit."""
        if self.chain is None:
            # Last commit missed, so the verifier's select missed the same
            # way (mirror symmetry) and fell back to a plain decode: this
            # commit is a single case-0 bonus. (The engine's miss-round block
            # only carries case 0 anyway -- its dead cells are poisoned.)
            case = 0
            prefix: list[int] = []
        else:
            case = min(self.accept_len, len(self.chain))
            prefix = self.chain[:case]
        if force_miss:
            bonus = self.rng.randrange(self.vocab_size)
            self.chain = None
            return prefix + [bonus]
        pick_f = self._pick_live_f(units[case])
        bonus = units[case][pick_f][0]
        self.chain = list(units[case][pick_f][1:])
        return prefix + [bonus]


def _run_rounds(
    *,
    engine: EnumDraftEngine,
    keys: list[DraftReqKey],
    sims: dict[DraftReqKey, _VerifierSim],
    last_units: dict[DraftReqKey, list],
    rounds: int,
    miss_every: int,
    round_idx0: int = 0,
) -> list[float]:
    round_ms = []
    for r in range(rounds):
        for i, key in enumerate(keys):
            if key in last_units:
                # Stagger misses across seats so bs>1 rounds exercise MIXED
                # fast/slow subrounds, not just all-miss rounds.
                force_miss = miss_every > 0 and (round_idx0 + r + i) % miss_every == 0
                delta = sims[key].next_delta(last_units[key], force_miss=force_miss)
                engine.apply_commit(key, delta)
            # else: first round; the opened prompt is the pending delta.
        start = time.monotonic()
        packed = engine.draft_round(keys)
        torch.cuda.synchronize()
        round_ms.append(1000.0 * (time.monotonic() - start))
        assert packed is not None and packed["units_device"].shape[0] == len(keys)
        units_host = packed["units_device"].cpu().tolist()
        # Mixed rounds pack seats in subround partition order (hit, case-0,
        # bootstrap), not keys order -- route each block by its echoed seat,
        # exactly like the verifier's pool-indexed landing.
        for pool_idx, units in zip(packed["pool_indices"], units_host):
            last_units[keys[pool_idx]] = units
    return round_ms


def _compare_engines(
    *,
    model_runner,
    vocab_size: int,
    args: argparse.Namespace,
) -> None:
    """Drive a fast-path and a slow-path engine with identical commits;
    report the first block divergence per round (unit-level diff)."""
    # Dead-guess exclusion rank-shifts the fast engine's guess rows against
    # the never-excluding slow reference, so the near-tie comparison runs with
    # it off on both engines (exclusion itself is validated by e2e accept).
    engines = {
        "fast": EnumDraftEngine(
            model_runner=model_runner,
            num_steps=args.num_steps,
            fanout=args.fanout,
            exclude_dead_guess=False,
        ),
        "slow": EnumDraftEngine(
            model_runner=model_runner,
            num_steps=args.num_steps,
            fanout=args.fanout,
            # --compare-graph: the reference runs the SAME glue fast path,
            # only eager (no extend graph) -- matched fused semantics.
            enable_glue_fast_path=args.compare_graph,
            enable_extend_graph=False,
            exclude_dead_guess=False,
        ),
    }
    engines["fast"].effective_fanout = args.effective_fanout or args.fanout
    rng = random.Random(42)
    prompt = [rng.randrange(vocab_size) for _ in range(args.prompt_len)]
    key = DraftReqKey(src_verifier_rank=0, request_id="cmp-0")
    for engine in engines.values():
        engine.open(
            key, req_pool_idx=0, prompt_tokens=list(prompt), committed_outputs=[]
        )
    sim = _VerifierSim(
        accept_len=args.accept_len, fanout=args.fanout, rng=rng, vocab_size=vocab_size
    )
    delta: Optional[list[int]] = None
    mismatch_rounds = 0
    for r in range(args.rounds):
        units = {}
        paths = {}
        lens = {}
        for name, engine in engines.items():
            if delta is not None:
                engine.apply_commit(key, delta)
            hits_before = engine.hit_ct
            packed = engine.draft_round([key])
            paths[name] = "fast" if engine.hit_ct > hits_before else "slow"
            lens[name] = packed["base_committed_lens"][0]
            units[name] = packed["units_device"][0].cpu().tolist()
        # The fast engine poisons dead cells (case-0 collapsed miss blocks,
        # adaptive-fanout columns) while the slow reference engine always
        # enumerates the full grid: only live fast units are comparable.
        diffs = []
        live_units = 0
        for case, (uf_row, us_row) in enumerate(zip(units["fast"], units["slow"])):
            for f, (uf, us) in enumerate(zip(uf_row, us_row)):
                if uf[0] == -1:
                    continue
                live_units += 1
                if uf != us:
                    first_pos = next(
                        p for p, (a, b) in enumerate(zip(uf, us)) if a != b
                    )
                    diffs.append((case, f, first_pos))
        if diffs:
            mismatch_rounds += 1
            logger.info(
                "round %d MISMATCH paths=%s diff_units=%d/%d at(case,f,pos)=%s",
                r,
                paths,
                len(diffs),
                live_units,
                diffs[:8],
            )
            case, f, _ = diffs[0]
            logger.info(
                "round %d first diff case=%d f=%d fast=%s slow=%s",
                r,
                case,
                f,
                units["fast"][case][f],
                units["slow"][case][f],
            )
        force_miss = args.miss_every > 0 and r % args.miss_every == 0
        # Drive both engines from the SLOW engine's block (ground truth).
        delta = sim.next_delta(units["slow"], force_miss=force_miss)
    logger.info(
        "compare done: rounds=%d mismatch_rounds=%d fast(hit=%d,miss=%d) "
        "slow(hit=%d,miss=%d)",
        args.rounds,
        mismatch_rounds,
        engines["fast"].hit_ct,
        engines["fast"].miss_ct,
        engines["slow"].hit_ct,
        engines["slow"].miss_ct,
    )


def main() -> None:
    args = _parse_args()
    if args.profile:
        envs.SGLANG_DEBUG_DECOUPLED_DRAFT_PROFILE.set(True)

    # The drafter's engine sizes its decode batches at bs (backbone), bs*F
    # (case-0 miss rounds), and bs*(K+1)*F rows (branch chains); make sure
    # each has a captured decode graph.
    branch_rows = args.batch_size * (args.num_steps + 1) * args.fanout
    server_args_kwargs = dict(
        model_path=args.model_path,
        mem_fraction_static=args.mem_fraction_static,
        cuda_graph_bs_decode=sorted(
            {args.batch_size, args.batch_size * args.fanout, branch_rows}
        ),
        disable_cuda_graph=args.disable_cuda_graph,
        # The engine owns row lifecycle (ReqToTokenPool.free) and forks mamba
        # state with copy_from -- both incompatible with radix tracking; the
        # real drafter launches with the same two settings.
        disable_radix_cache=True,
        mamba_radix_cache_strategy="no_buffer",
        page_size=int(os.environ.get("BENCH_PAGE_SIZE", "1")),
    )
    if args.attention_backend is not None:
        server_args_kwargs["attention_backend"] = args.attention_backend
    if args.context_length is not None:
        server_args_kwargs["context_length"] = args.context_length
    if args.cuda_graph_backend_prefill is not None:
        server_args_kwargs["cuda_graph_backend_prefill"] = (
            args.cuda_graph_backend_prefill
        )
    server_args = ServerArgs(**server_args_kwargs)
    port_args = PortArgs.init_new(server_args)
    bench_runner, _tokenizer = load_model(server_args, port_args, 0, 0)
    model_runner = bench_runner.torch_runner
    vocab_size = model_runner.model_config.vocab_size
    logger.info(
        "attention_backend=%s cuda_graph_bs_decode=%s",
        model_runner.server_args.attention_backend,
        model_runner.server_args.cuda_graph_bs_decode,
    )

    if args.compare or args.compare_graph:
        _compare_engines(model_runner=model_runner, vocab_size=vocab_size, args=args)
        return

    engine = EnumDraftEngine(
        model_runner=model_runner,
        num_steps=args.num_steps,
        fanout=args.fanout,
    )
    if args.effective_fanout is not None:
        engine.effective_fanout = args.effective_fanout
    rng = random.Random(42)
    keys = []
    sims: dict[DraftReqKey, _VerifierSim] = {}
    last_units: dict[DraftReqKey, list] = {}
    for i in range(args.batch_size):
        key = DraftReqKey(src_verifier_rank=0, request_id=f"bench-{i}")
        engine.open(
            key,
            req_pool_idx=i,
            prompt_tokens=[rng.randrange(vocab_size) for _ in range(args.prompt_len)],
            committed_outputs=[],
        )
        keys.append(key)
        sims[key] = _VerifierSim(
            accept_len=args.accept_len,
            fanout=args.fanout,
            rng=rng,
            vocab_size=vocab_size,
        )

    _run_rounds(
        engine=engine,
        keys=keys,
        sims=sims,
        last_units=last_units,
        rounds=args.warmup_rounds,
        miss_every=args.miss_every,
    )
    engine.profiler.round_ct = 0
    engine.profiler.phase_ms = {}
    engine.hit_ct = 0
    engine.miss_ct = 0
    round_ms = _run_rounds(
        engine=engine,
        keys=keys,
        sims=sims,
        last_units=last_units,
        rounds=args.rounds,
        miss_every=args.miss_every,
        round_idx0=args.warmup_rounds,
    )

    if args.torch_profile_trace is not None:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            _run_rounds(
                engine=engine,
                keys=keys,
                sims=sims,
                last_units=last_units,
                rounds=3,
                miss_every=args.miss_every,
            )
        prof.export_chrome_trace(args.torch_profile_trace)
        logger.info(
            "torch profiler top ops:\n%s",
            prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=25, max_name_column_width=60
            ),
        )
        logger.info("chrome trace written to %s", args.torch_profile_trace)

    round_ms.sort()
    n = len(round_ms)
    logger.info(
        "drafter round (bs=%d K=%d F=%d prompt=%d delta=%d rounds=%d "
        "fast=%d slow=%d): p50=%.2fms mean=%.2fms p90=%.2fms max=%.2fms",
        args.batch_size,
        args.num_steps,
        args.fanout,
        args.prompt_len,
        args.accept_len + 1,
        n,
        engine.hit_ct,
        engine.miss_ct,
        round_ms[n // 2],
        sum(round_ms) / n,
        round_ms[(n * 9) // 10],
        round_ms[-1],
    )
    if args.profile:
        logger.info("phase breakdown: %s", engine.profiler.summary())


if __name__ == "__main__":
    main()
