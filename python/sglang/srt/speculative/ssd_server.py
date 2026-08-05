"""Draft-side control plane for SSD outcome-cache construction.

The generic ``/generate`` API returns every branch result to the target
process.  That duplicates all branch prefixes in the request and serializes
the complete outcome tree in the response, even though the target consumes
exactly one entry.  This module keeps the tree in the draft HTTP process and
exposes only a compact build/select protocol.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.speculative.ssd_draft_client import (
    DraftCandidate,
    OutcomeCache,
    SSDDraftClient,
    SSDDraftClientError,
)

logger = logging.getLogger(__name__)


@dataclass
class SSDOutcomeCacheBuildReq:
    cache_id: str
    canonical_prefix: List[int]
    draft_tokens: List[int]
    recovery_tokens: List[List[int]]
    draft_length: int
    fan_outs: List[int]


@dataclass
class SSDOutcomeCacheSelectReq:
    cache_id: str
    accepted_length: int
    recovery_token: int


@dataclass
class SSDOutcomeCacheDiscardReq:
    cache_id: str


class _OutcomeCacheStore:
    """Small bounded store for one-shot outcome trees.

    A tree is removed on the first select, whether that select hits or misses.
    The size and TTL bounds cover target cancellation/crash paths where a
    completed build is never selected.
    """

    def __init__(self, max_entries: int, ttl_seconds: float):
        if max_entries <= 0:
            raise ValueError("SSD outcome-cache max entries must be positive.")
        if ttl_seconds <= 0:
            raise ValueError("SSD outcome-cache TTL must be positive.")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._caches: OrderedDict[str, Tuple[float, OutcomeCache]] = OrderedDict()

    def _evict_expired(self, now: float) -> None:
        while self._caches:
            _, (created_at, _) = next(iter(self._caches.items()))
            if now - created_at <= self.ttl_seconds:
                break
            self._caches.popitem(last=False)

    def put(self, cache_id: str, cache: OutcomeCache) -> None:
        now = time.monotonic()
        self._evict_expired(now)
        self._caches.pop(cache_id, None)
        self._caches[cache_id] = (now, cache)
        while len(self._caches) > self.max_entries:
            self._caches.popitem(last=False)

    def pop_selected(
        self, cache_id: str, key: Tuple[int, int]
    ) -> Tuple[DraftCandidate | None, float | None]:
        now = time.monotonic()
        self._evict_expired(now)
        stored = self._caches.pop(cache_id, None)
        if stored is None:
            return None, None
        created_at, cache = stored
        return cache.get(key), (now - created_at) * 1e3

    def discard(self, cache_id: str) -> bool:
        return self._caches.pop(cache_id, None) is not None


def _candidate_payload(candidate: DraftCandidate) -> Dict[str, object]:
    return {
        "tokens": candidate.tokens,
        "recovery_tokens": [list(tokens) for tokens in candidate.recovery_tokens],
    }


def register_ssd_routes(
    app: FastAPI,
    get_tokenizer_manager: Callable[[], object],
) -> None:
    """Register the compact SSD endpoints on an SGLang HTTP server."""

    store = _OutcomeCacheStore(
        max_entries=int(os.environ.get("SGLANG_SSD_CACHE_MAX_ENTRIES", "32")),
        ttl_seconds=float(os.environ.get("SGLANG_SSD_CACHE_TTL_SECONDS", "120")),
    )

    @app.post("/ssd/build_outcome_cache")
    async def build_outcome_cache(
        obj: SSDOutcomeCacheBuildReq, request: Request
    ) -> ORJSONResponse:
        total_begin = time.perf_counter()
        try:
            fan_outs = SSDDraftClient.normalize_fan_outs(
                obj.draft_length, obj.fan_outs
            )
            candidate = SSDDraftClient.candidate_from_payload(
                {
                    "tokens": obj.draft_tokens,
                    "recovery_tokens": obj.recovery_tokens,
                },
                obj.draft_length,
                fan_outs,
            )
            branch_keys, branch_prefixes = SSDDraftClient.build_outcome_branches(
                obj.canonical_prefix,
                candidate,
                obj.draft_length,
                fan_outs,
            )
        except (SSDDraftClientError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        prepare_ms = (time.perf_counter() - total_begin) * 1e3
        generate_obj = GenerateReqInput(
            input_ids=branch_prefixes,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": obj.draft_length + 1,
                "ignore_eos": True,
                "skip_special_tokens": False,
            },
            return_logprob=True,
            top_logprobs_num=max(fan_outs) + 1,
            return_text_in_logprobs=False,
            log_metrics=False,
        )

        generate_begin = time.perf_counter()
        try:
            result = await get_tokenizer_manager().generate_request(
                generate_obj, request
            ).__anext__()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        generate_ms = (time.perf_counter() - generate_begin) * 1e3

        parse_begin = time.perf_counter()
        rows = result if isinstance(result, list) else [result]
        if len(rows) != len(branch_keys):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"SSD draft server returned {len(rows)} rows for "
                    f"{len(branch_keys)} outcome branches."
                ),
            )
        try:
            cache = {
                key: SSDDraftClient.candidate_from_row(
                    row, obj.draft_length, fan_outs
                )
                for key, row in zip(branch_keys, rows)
            }
        except (SSDDraftClientError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        parse_ms = (time.perf_counter() - parse_begin) * 1e3

        store.put(obj.cache_id, cache)
        total_ms = (time.perf_counter() - total_begin) * 1e3
        return ORJSONResponse(
            {
                "cache_id": obj.cache_id,
                "branches": len(cache),
                "timing_ms": {
                    "prepare": prepare_ms,
                    "generate": generate_ms,
                    "parse": parse_ms,
                    "total": total_ms,
                },
            }
        )

    @app.post("/ssd/select_outcome")
    async def select_outcome(
        obj: SSDOutcomeCacheSelectReq,
    ) -> ORJSONResponse:
        begin = time.perf_counter()
        candidate, cache_age_ms = store.pop_selected(
            obj.cache_id, (obj.accepted_length, obj.recovery_token)
        )
        lookup_ms = (time.perf_counter() - begin) * 1e3
        if candidate is None:
            return ORJSONResponse(
                {
                    "hit": False,
                    "cache_age_ms": cache_age_ms,
                    "lookup_ms": lookup_ms,
                }
            )
        return ORJSONResponse(
            {
                "hit": True,
                "candidate": _candidate_payload(candidate),
                "cache_age_ms": cache_age_ms,
                "lookup_ms": lookup_ms,
            }
        )

    @app.post("/ssd/discard_outcome_cache")
    async def discard_outcome_cache(
        obj: SSDOutcomeCacheDiscardReq,
    ) -> ORJSONResponse:
        return ORJSONResponse({"discarded": store.discard(obj.cache_id)})
