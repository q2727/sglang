"""HTTP client for the external draft-model process used by SSD.

The target server and the draft server are independent CUDA clients.  The
launcher can therefore give them fixed MPS active-thread percentages while
this client implements Saguaro/SSD's outcome cache on top of the draft
server's normal radix cache.
"""

from __future__ import annotations

import json
import http.client
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


OutcomeKey = Tuple[int, int]


@dataclass(frozen=True)
class DraftCandidate:
    """One K-token draft plus recovery choices at all K+1 endpoints."""

    tokens: List[int]
    recovery_tokens: Tuple[Tuple[int, ...], ...]
    # The official SSD backend selects directly from its GPU-resident outcome
    # tree.  It reports whether that selection hit without shipping the tree's
    # logits/recovery lists back to the target process.
    cache_hit: Optional[bool] = None


OutcomeCache = Dict[OutcomeKey, DraftCandidate]
FanOutSpec = Union[int, Sequence[int]]


@dataclass(frozen=True)
class OutcomeCacheHandle:
    """Opaque reference to an outcome tree retained by the draft server."""

    cache_id: str
    branches: int
    server_prepare_ms: float
    server_generate_ms: float
    server_parse_ms: float
    server_total_ms: float


class SSDDraftClientError(RuntimeError):
    """Raised when the external draft server returns an invalid response."""


@dataclass(frozen=True)
class SSDDraftClientStats:
    generate_calls: int
    generated_sequences: int


class SSDDraftClient:
    """Small synchronous client for one external SGLang draft server.

    Calls are intentionally synchronous here.  ``SSDWorker`` owns a background
    executor and overlaps ``build_outcome_cache`` with target verification.
    Keeping the HTTP client synchronous makes failures and cache construction
    deterministic and easy to test.
    """

    def __init__(self, server_url: str, timeout: float = 600.0):
        if not server_url:
            raise ValueError("The SSD draft server URL must not be empty.")
        if timeout <= 0:
            raise ValueError("The SSD draft request timeout must be positive.")
        parsed = urlsplit(server_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or parsed.hostname is None:
            raise ValueError(
                "The SSD draft server URL must be an absolute HTTP(S) URL."
            )
        self.server_url = server_url.rstrip("/")
        self.generate_url = self.server_url + "/generate"
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._path_prefix = parsed.path.rstrip("/")
        self._connection: http.client.HTTPConnection | None = None
        self._connection_lock = threading.Lock()
        self.timeout = timeout
        self._generate_calls = 0
        self._generated_sequences = 0

    @property
    def stats(self) -> SSDDraftClientStats:
        return SSDDraftClientStats(
            generate_calls=self._generate_calls,
            generated_sequences=self._generated_sequences,
        )

    def _new_connection(self) -> http.client.HTTPConnection:
        connection_cls = (
            http.client.HTTPSConnection
            if self._scheme == "https"
            else http.client.HTTPConnection
        )
        return connection_cls(self._host, self._port, timeout=self.timeout)

    def _post_json(self, path: str, payload: Mapping[str, object]) -> object:
        """POST JSON over one reusable connection.

        All SSD calls are synchronous and serialized at protocol boundaries,
        so a single locked keep-alive connection removes per-round TCP setup
        without introducing concurrent use of ``HTTPConnection``.
        """

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request_path = self._path_prefix + path
        with self._connection_lock:
            for attempt in range(2):
                if self._connection is None:
                    self._connection = self._new_connection()
                try:
                    self._connection.request(
                        "POST",
                        request_path,
                        body=body,
                        headers={
                            "Content-Type": "application/json",
                            "Content-Length": str(len(body)),
                            "Connection": "keep-alive",
                        },
                    )
                    response = self._connection.getresponse()
                    raw = response.read()
                    status = response.status
                    if response.will_close:
                        self._connection.close()
                        self._connection = None
                    break
                except (OSError, TimeoutError, http.client.HTTPException) as exc:
                    if self._connection is not None:
                        self._connection.close()
                    self._connection = None
                    if attempt == 1:
                        raise SSDDraftClientError(
                            f"SSD draft request to {self.server_url}{path} "
                            f"failed: {exc}"
                        ) from exc

        if status < 200 or status >= 300:
            detail = raw.decode("utf-8", errors="replace")
            raise SSDDraftClientError(
                f"SSD draft request to {self.server_url}{path} returned "
                f"HTTP {status}: {detail}"
            )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SSDDraftClientError(
                f"SSD draft request to {self.server_url}{path} returned invalid JSON."
            ) from exc

    def _post_generate(
        self,
        input_ids: Sequence[int] | Sequence[Sequence[int]],
        max_new_tokens: int,
        *,
        top_logprobs_num: int = 0,
    ) -> List[Mapping[str, object]]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        is_batch = bool(input_ids) and isinstance(input_ids[0], (list, tuple))
        expected = len(input_ids) if is_batch else 1
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "top_logprobs_num": top_logprobs_num,
        }
        result = self._post_json("/generate", payload)

        rows = result if isinstance(result, list) else [result]
        if len(rows) != expected:
            raise SSDDraftClientError(
                f"SSD draft server returned {len(rows)} rows for {expected} inputs."
            )
        if not all(isinstance(row, dict) for row in rows):
            raise SSDDraftClientError("SSD draft response rows must be JSON objects.")

        self._generate_calls += 1
        self._generated_sequences += expected
        return rows

    @staticmethod
    def _output_token_ids(row: Mapping[str, object], expected: int) -> List[int]:
        try:
            meta_info = row["meta_info"]
            entries = meta_info["output_token_logprobs"]
            token_ids = [int(entry[1]) for entry in entries]
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise SSDDraftClientError(
                "SSD draft response is missing output token IDs in meta_info."
            ) from exc
        if len(token_ids) != expected:
            raise SSDDraftClientError(
                f"SSD draft returned {len(token_ids)} tokens; expected {expected}."
            )
        return token_ids

    @staticmethod
    def _output_top_ids(row: Mapping[str, object], expected: int) -> List[List[int]]:
        try:
            meta_info = row["meta_info"]
            per_token = meta_info["output_top_logprobs"]
            result = [
                [int(entry[1]) for entry in token_entries]
                for token_entries in per_token
            ]
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise SSDDraftClientError(
                "SSD draft response is missing output top logprobs."
            ) from exc
        if len(result) != expected:
            raise SSDDraftClientError(
                f"SSD draft returned top logprobs for {len(result)} tokens; "
                f"expected {expected}."
            )
        return result

    @classmethod
    def candidate_from_row(
        cls,
        row: Mapping[str, object],
        draft_length: int,
        fan_outs: Sequence[int],
    ) -> DraftCandidate:
        generated = cls._output_token_ids(row, draft_length + 1)
        tokens = generated[:draft_length]
        top_ids = cls._output_top_ids(row, draft_length + 1)
        recoveries = []
        for accepted_length, endpoint_top_ids in enumerate(top_ids):
            exclude = (
                tokens[accepted_length]
                if accepted_length < draft_length
                else None
            )
            recoveries.append(
                tuple(
                    cls._select_recovery_tokens(
                        endpoint_top_ids,
                        exclude=exclude,
                        fan_out=fan_outs[accepted_length],
                    )
                )
            )
        return DraftCandidate(tokens=tokens, recovery_tokens=tuple(recoveries))

    @staticmethod
    def normalize_fan_outs(
        draft_length: int, fan_out: FanOutSpec
    ) -> Tuple[int, ...]:
        if isinstance(fan_out, int):
            fan_outs = (fan_out,) * (draft_length + 1)
        else:
            fan_outs = tuple(int(value) for value in fan_out)
        if len(fan_outs) != draft_length + 1:
            raise ValueError(
                f"Expected {draft_length + 1} fan-out values, got {len(fan_outs)}."
            )
        if any(value < 0 for value in fan_outs):
            raise ValueError("SSD fan-out values must be non-negative.")
        if sum(fan_outs) == 0:
            raise ValueError("SSD fan-out values must contain at least one branch.")
        return fan_outs

    @staticmethod
    def candidate_to_payload(candidate: DraftCandidate) -> Dict[str, object]:
        return {
            "tokens": [int(token) for token in candidate.tokens],
            "recovery_tokens": [
                [int(token) for token in endpoint]
                for endpoint in candidate.recovery_tokens
            ],
        }

    @classmethod
    def candidate_from_payload(
        cls,
        payload: Mapping[str, object],
        draft_length: int,
        fan_out: FanOutSpec,
    ) -> DraftCandidate:
        fan_outs = cls.normalize_fan_outs(draft_length, fan_out)
        try:
            tokens = [int(token) for token in payload["tokens"]]
            recovery_tokens = tuple(
                tuple(int(token) for token in endpoint)
                for endpoint in payload["recovery_tokens"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SSDDraftClientError("Invalid SSD draft-candidate payload.") from exc
        candidate = DraftCandidate(tokens=tokens, recovery_tokens=recovery_tokens)
        cls.validate_candidate(candidate, draft_length, fan_outs)
        return candidate

    @staticmethod
    def validate_candidate(
        candidate: DraftCandidate,
        draft_length: int,
        fan_outs: Sequence[int],
    ) -> None:
        if len(candidate.tokens) != draft_length:
            raise ValueError(
                f"Expected {draft_length} draft tokens, "
                f"got {len(candidate.tokens)}."
            )
        if len(candidate.recovery_tokens) != draft_length + 1:
            raise ValueError(
                f"Expected {draft_length + 1} recovery endpoints, "
                f"got {len(candidate.recovery_tokens)}."
            )
        for accepted_length, recoveries in enumerate(candidate.recovery_tokens):
            expected_fan_out = fan_outs[accepted_length]
            if len(recoveries) != expected_fan_out:
                raise ValueError(
                    f"Endpoint {accepted_length} has {len(recoveries)} recovery "
                    f"tokens; expected {expected_fan_out}."
                )

    def jit_draft(
        self, prefix: Sequence[int], draft_length: int, fan_out: FanOutSpec
    ) -> DraftCandidate:
        """Generate a fallback draft and retain all endpoint distributions.

        Generating K+1 tokens yields the next-token distribution at every
        endpoint from 0 through K.  The extra output token is discarded, but
        its distribution replaces a separate endpoint-logit request.
        """

        if not prefix:
            raise ValueError("SSD cannot draft from an empty prefix.")
        fan_outs = self.normalize_fan_outs(draft_length, fan_out)
        rows = self._post_generate(
            list(prefix),
            draft_length + 1,
            top_logprobs_num=max(fan_outs) + 1,
        )
        return self.candidate_from_row(rows[0], draft_length, fan_outs)

    @staticmethod
    def _select_recovery_tokens(
        top_ids: Iterable[int], *, exclude: int | None, fan_out: int
    ) -> List[int]:
        if fan_out == 0:
            return []
        selected: List[int] = []
        for token_id in top_ids:
            if token_id == exclude or token_id in selected:
                continue
            selected.append(token_id)
            if len(selected) == fan_out:
                break
        if len(selected) != fan_out:
            raise SSDDraftClientError(
                f"Only found {len(selected)} recovery tokens for fan_out={fan_out}."
            )
        return selected

    @classmethod
    def build_outcome_branches(
        cls,
        canonical_prefix: Sequence[int],
        candidate: DraftCandidate,
        draft_length: int,
        fan_out: FanOutSpec,
    ) -> Tuple[List[OutcomeKey], List[List[int]]]:
        if not canonical_prefix:
            raise ValueError("SSD cannot build an outcome tree from an empty prefix.")
        if draft_length <= 0:
            raise ValueError("draft_length must be positive.")
        fan_outs = cls.normalize_fan_outs(draft_length, fan_out)
        cls.validate_candidate(candidate, draft_length, fan_outs)

        base = list(canonical_prefix[:-1])
        path = [int(canonical_prefix[-1]), *map(int, candidate.tokens)]
        endpoint_prefixes = [
            base + path[: endpoint + 1] for endpoint in range(draft_length + 1)
        ]

        branch_keys: List[OutcomeKey] = []
        branch_prefixes: List[List[int]] = []
        for accepted_length, recoveries in enumerate(candidate.recovery_tokens):
            endpoint_prefix = endpoint_prefixes[accepted_length]
            for recovery_token in recoveries:
                branch_keys.append((accepted_length, recovery_token))
                branch_prefixes.append(endpoint_prefix + [recovery_token])
        return branch_keys, branch_prefixes

    def build_outcome_cache(
        self,
        canonical_prefix: Sequence[int],
        candidate: DraftCandidate,
        draft_length: int,
        fan_out: FanOutSpec,
    ) -> OutcomeCache:
        """Build next-round drafts for SSD's possible verification outcomes.

        ``canonical_prefix`` ends in the current target recovery token and
        ``candidate`` contains the K-token path being verified.  Endpoint ``j``
        represents accepting exactly ``j`` draft tokens.  For endpoints before
        K, the on-path next token is excluded because accepting it would move to
        the following endpoint instead of recovering at this one.
        """

        fan_outs = self.normalize_fan_outs(draft_length, fan_out)
        branch_keys, branch_prefixes = self.build_outcome_branches(
            canonical_prefix, candidate, draft_length, fan_outs
        )

        branch_rows = self._post_generate(
            branch_prefixes,
            draft_length + 1,
            top_logprobs_num=max(fan_outs) + 1,
        )
        return {
            key: self.candidate_from_row(row, draft_length, fan_outs)
            for key, row in zip(branch_keys, branch_rows)
        }

    def prepare_outcome_cache(
        self,
        canonical_prefix: Sequence[int],
        candidate: DraftCandidate,
        draft_length: int,
        fan_out: FanOutSpec,
    ) -> OutcomeCacheHandle:
        """Build an outcome tree and retain it in the draft server process."""

        if not canonical_prefix:
            raise ValueError("SSD cannot build an outcome tree from an empty prefix.")
        fan_outs = self.normalize_fan_outs(draft_length, fan_out)
        self.validate_candidate(candidate, draft_length, fan_outs)
        cache_id = uuid.uuid4().hex
        payload = {
            "cache_id": cache_id,
            "canonical_prefix": [int(token) for token in canonical_prefix],
            "draft_tokens": [int(token) for token in candidate.tokens],
            "recovery_tokens": [
                [int(token) for token in endpoint]
                for endpoint in candidate.recovery_tokens
            ],
            "draft_length": draft_length,
            "fan_outs": list(fan_outs),
        }
        result = self._post_json("/ssd/build_outcome_cache", payload)
        if not isinstance(result, dict) or result.get("cache_id") != cache_id:
            raise SSDDraftClientError(
                "SSD draft server returned an invalid outcome-cache handle."
            )
        try:
            branches = int(result["branches"])
            timing = result["timing_ms"]
            handle = OutcomeCacheHandle(
                cache_id=cache_id,
                branches=branches,
                server_prepare_ms=float(timing["prepare"]),
                server_generate_ms=float(timing["generate"]),
                server_parse_ms=float(timing["parse"]),
                server_total_ms=float(timing["total"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SSDDraftClientError(
                "SSD draft server returned malformed outcome-cache metadata."
            ) from exc
        expected_branches = sum(fan_outs)
        if branches != expected_branches:
            raise SSDDraftClientError(
                f"SSD draft server built {branches} branches; "
                f"expected {expected_branches}."
            )
        self._generate_calls += 1
        self._generated_sequences += branches
        return handle

    def select_outcome(
        self,
        handle: OutcomeCacheHandle,
        outcome_key: OutcomeKey,
        draft_length: int,
        fan_out: FanOutSpec,
    ) -> DraftCandidate | None:
        """Consume one selected entry from a draft-side outcome tree."""

        result = self._post_json(
            "/ssd/select_outcome",
            {
                "cache_id": handle.cache_id,
                "accepted_length": int(outcome_key[0]),
                "recovery_token": int(outcome_key[1]),
            },
        )
        if not isinstance(result, dict) or not isinstance(result.get("hit"), bool):
            raise SSDDraftClientError(
                "SSD draft server returned an invalid outcome selection."
            )
        if not result["hit"]:
            return None
        candidate_payload = result.get("candidate")
        if not isinstance(candidate_payload, dict):
            raise SSDDraftClientError(
                "SSD draft server omitted the selected draft candidate."
            )
        return self.candidate_from_payload(
            candidate_payload, draft_length, fan_out
        )
