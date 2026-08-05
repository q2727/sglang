"""HTTP client for the external draft-model process used by SSD.

The target server and the draft server are independent CUDA clients.  The
launcher can therefore give them fixed MPS active-thread percentages while
this client implements Saguaro/SSD's outcome cache on top of the draft
server's normal radix cache.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


OutcomeKey = Tuple[int, int]


@dataclass(frozen=True)
class DraftCandidate:
    """One K-token draft plus recovery choices at all K+1 endpoints."""

    tokens: List[int]
    recovery_tokens: Tuple[Tuple[int, ...], ...]


OutcomeCache = Dict[OutcomeKey, DraftCandidate]
FanOutSpec = Union[int, Sequence[int]]


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
        self.generate_url = server_url.rstrip("/") + "/generate"
        self.timeout = timeout
        self._generate_calls = 0
        self._generated_sequences = 0

    @property
    def stats(self) -> SSDDraftClientStats:
        return SSDDraftClientStats(
            generate_calls=self._generate_calls,
            generated_sequences=self._generated_sequences,
        )

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
        request = urllib.request.Request(
            self.generate_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SSDDraftClientError(
                f"SSD draft request to {self.generate_url} failed: {exc}"
            ) from exc

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

    def _candidate_from_row(
        self,
        row: Mapping[str, object],
        draft_length: int,
        fan_outs: Sequence[int],
    ) -> DraftCandidate:
        generated = self._output_token_ids(row, draft_length + 1)
        tokens = generated[:draft_length]
        top_ids = self._output_top_ids(row, draft_length + 1)
        recoveries = []
        for accepted_length, endpoint_top_ids in enumerate(top_ids):
            exclude = (
                tokens[accepted_length]
                if accepted_length < draft_length
                else None
            )
            recoveries.append(
                tuple(
                    self._select_recovery_tokens(
                        endpoint_top_ids,
                        exclude=exclude,
                        fan_out=fan_outs[accepted_length],
                    )
                )
            )
        return DraftCandidate(tokens=tokens, recovery_tokens=tuple(recoveries))

    @staticmethod
    def _normalize_fan_outs(
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
        fan_outs = self._normalize_fan_outs(draft_length, fan_out)
        rows = self._post_generate(
            list(prefix),
            draft_length + 1,
            top_logprobs_num=max(fan_outs) + 1,
        )
        return self._candidate_from_row(rows[0], draft_length, fan_outs)

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

        if not canonical_prefix:
            raise ValueError("SSD cannot build an outcome tree from an empty prefix.")
        if draft_length <= 0:
            raise ValueError("draft_length must be positive.")
        fan_outs = self._normalize_fan_outs(draft_length, fan_out)
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

        # Candidate generation already produced the distributions at all K+1
        # endpoints.  Build only the recovery branches here, in one batch.
        base = list(canonical_prefix[:-1])
        path = [int(canonical_prefix[-1]), *map(int, candidate.tokens)]
        endpoint_prefixes = [
            base + path[: endpoint + 1] for endpoint in range(draft_length + 1)
        ]

        branch_keys: List[OutcomeKey] = []
        branch_prefixes: List[List[int]] = []
        for accepted_length, recoveries in enumerate(candidate.recovery_tokens):
            expected_fan_out = fan_outs[accepted_length]
            if len(recoveries) != expected_fan_out:
                raise ValueError(
                    f"Endpoint {accepted_length} has {len(recoveries)} recovery "
                    f"tokens; expected {expected_fan_out}."
                )
            endpoint_prefix = endpoint_prefixes[accepted_length]
            for recovery_token in recoveries:
                branch_keys.append((accepted_length, recovery_token))
                branch_prefixes.append(endpoint_prefix + [recovery_token])

        branch_rows = self._post_generate(
            branch_prefixes,
            draft_length + 1,
            top_logprobs_num=max(fan_outs) + 1,
        )
        return {
            key: self._candidate_from_row(row, draft_length, fan_outs)
            for key, row in zip(branch_keys, branch_rows)
        }
