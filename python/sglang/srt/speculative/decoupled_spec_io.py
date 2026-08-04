from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class DraftMeshMessageType(str, Enum):
    CONTROL_BATCH = "control_batch"
    ENUMERATION_BUFFER_BATCH = "enumeration_buffer_batch"


@dataclass(frozen=True)
class DraftReqKey:
    """Request identity on the drafter side.

    The original request_id is only unique within the verifier that owns it.
    src_verifier_rank keeps the drafter-side request table unambiguous when
    multiple verifier ranks send work to the same drafter rank.
    """

    src_verifier_rank: int
    request_id: str


def build_draft_scheduler_rid(draft_key: DraftReqKey) -> str:
    return f"draft:{int(draft_key.src_verifier_rank)}:{draft_key.request_id}"


def parse_draft_scheduler_rid(rid: str) -> DraftReqKey:
    if rid.startswith("draft:"):
        encoded = rid[len("draft:") :]
        rank_text, sep, request_id = encoded.partition(":")
        if sep and request_id:
            return DraftReqKey(
                src_verifier_rank=int(rank_text),
                request_id=request_id,
            )

    raise ValueError(f"Invalid decoupled draft scheduler rid: {rid}")


@dataclass
class DraftSync:
    """Open or re-open a drafter request from a verifier-owned prefix.

    The verifier is the source of truth for committed tokens. DraftSync gives
    the drafter the prompt and already committed output prefix that it must
    align to before it can emit draft tail tokens.

    req_pool_idx is the seat (req_to_token row) the request holds on the owning
    verifier. The drafter echoes it in every enumeration block row, so the
    verifier lands rows with a pure GPU scatter and no host rid lookup. A seat
    only changes across a retraction re-admit, and that path re-opens the
    request with a fresh DraftSync carrying the new seat; a late block aimed at
    the old seat is defused by its base_committed_len stamp, not by routing.
    """

    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    req_pool_idx: int
    prompt_token_ids: list[int] = field(default_factory=list)
    committed_outputs: list[int] = field(default_factory=list)
    # Desync re-seed (same rid, same seat, chain intact): the verifier IPC
    # thread keeps its commit-ledger cursor and applies a send floor instead
    # of re-rooting it (see _drain_send_queue). False = initial seed or a
    # retraction re-root, where the snapshot IS the new absolute truth.
    desync_reseed: bool = False

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


@dataclass
class VerifyCommit:
    """
    Sent from verifier to drafter to commit a portion of the draft outputs.
    committed_tokens is the verifier-committed contiguous segment output_ids[
    pre_verify_committed_len : pre_verify_committed_len + len(committed_tokens)];
    the drafter aligns its reqs to it, sometimes truncating / reprefilling.

    Enumeration: committed_tokens is the accepted draft tokens followed by the
    bonus token, so accept_len = len(committed_tokens) - 1 and bonus =
    committed_tokens[-1]. A fallback round commits exactly [bonus] (accept_len == 0).
    """

    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    pre_verify_committed_len: int
    committed_tokens: list[int]
    # The seat (verifier req_to_token row) this commit belongs to: lets the
    # drafter's IPC thread land the commit into its GPU mirror without a
    # host rid lookup (the same seat-routing idea as the enumeration block's
    # pool_idx echo). -1 on messages from older verifiers.
    req_pool_idx: int = -1

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )

    def validate_committed_tokens(self) -> None:
        if not self.committed_tokens:
            raise ValueError(
                "VerifyCommit committed_tokens must be non-empty: "
                f"request_id={self.request_id} "
                f"pre_verify_committed_len={self.pre_verify_committed_len}"
            )
        if int(self.pre_verify_committed_len) < 0:
            raise ValueError(
                "VerifyCommit pre_verify_committed_len must be non-negative: "
                f"request_id={self.request_id} "
                f"pre_verify_committed_len={self.pre_verify_committed_len}"
            )


@dataclass
class DraftClose:
    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    reason: str

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


@dataclass
class DraftEnumerationBufferBatch:
    """One drafter -> one verifier, one scheduler step: a parallel-array batch
    (SGLang IPC convention, token ids only). The drafter pre-enumerates, from
    each request's committed base, every chain the verifier could select in the
    next round; the verifier GPU-selects the matching row, so a wrong guess is
    never committed.

    Dims (batch-uniform scalars): num_steps (K) = draft chain length per case,
    fanout (F) = bonus-token guesses per accept case; K + 1 = the previous
    round's possible accept lengths (0..K).

    tokens: a flat B * (K + 1) * F * (K + 1) tuple of vocab ids. Each
    (accept_case, guess) unit is K + 1 wide: [guess, chain_1 .. chain_K] --
    element 0 is the guessed bonus token itself (the GPU bonus-match key), the
    rest is the chain drafted after it. A hit unit IS the verify row
    (root = guess = the real bonus, then K draft tokens), so selection is one
    index_select with no re-assembly. Row i starts at i * row_stride,
    row_stride = (K + 1) * F * (K + 1), and within a row the unit for
    (accept_case, guess) starts at (accept_case * F + guess) * (K + 1).

    pool_indices[i] is the seat (verifier req_to_token row) row i lands in,
    echoed from the req_pool_idx the verifier announced in DraftSync; landing is
    a pure GPU scatter with no host rid lookup. A stale echo (the request left
    and the seat was reused) is defused by the stamp below, never by routing.

    base_committed_lens[i] is the staleness version (committed length row i was
    drafted from) the verifier judges usability on entirely on GPU.

    src_drafter_rank / dst_verifier_rank are scalars because a wire message has
    one sender and one destination; aggregation across drafters happens on the
    verifier, collecting the messages it receives keyed on src_drafter_rank.
    rids is optional debug labeling (parallel to pool_indices when present);
    nothing routes on it.
    """

    src_drafter_rank: int
    dst_verifier_rank: int
    num_steps: int
    fanout: int
    pool_indices: list[int] = field(default_factory=list)
    base_committed_lens: list[int] = field(default_factory=list)
    tokens: tuple[int, ...] = ()
    rids: list[str] = field(default_factory=list)
    # Drafter-side time.time() at submit; debug-profile latency only (both
    # processes share the host clock). None when profiling is off.
    sent_unix_ts: Optional[float] = None
    # True for a top-1 prerun block: enumerated from a PREDICTED commit (the
    # drafter bet on full-accept + its own top bonus guess) and shipped before
    # the real commit. Lands in the dedicated speculative generation; its
    # stamp only ever matches when the bet was right.
    speculative: bool = False

    @property
    def row_stride(self) -> int:
        # (K + 1) accept cases x F guesses x (K + 1)-wide [guess, chain] units.
        unit_width = int(self.num_steps) + 1
        return unit_width * int(self.fanout) * unit_width

    @property
    def batch_size(self) -> int:
        return len(self.pool_indices)

    @property
    def num_tokens(self) -> int:
        return self.batch_size * self.row_stride

    def row_tokens(self, i: int) -> tuple[int, ...]:
        return self.tokens[i * self.row_stride : (i + 1) * self.row_stride]

    def validate(self) -> None:
        if int(self.num_steps) < 1:
            raise ValueError(
                "DraftEnumerationBufferBatch num_steps must be >= 1: "
                f"batch_size={self.batch_size} num_steps={self.num_steps}"
            )
        if int(self.fanout) < 1:
            raise ValueError(
                "DraftEnumerationBufferBatch fanout must be >= 1: "
                f"batch_size={self.batch_size} fanout={self.fanout}"
            )
        if len(self.pool_indices) != len(self.base_committed_lens):
            raise ValueError(
                "DraftEnumerationBufferBatch pool_indices and "
                "base_committed_lens must have equal length: "
                f"len(pool_indices)={len(self.pool_indices)} "
                f"len(base_committed_lens)={len(self.base_committed_lens)}"
            )
        if self.rids and len(self.rids) != len(self.pool_indices):
            raise ValueError(
                "DraftEnumerationBufferBatch rids, when present, must parallel "
                "pool_indices: "
                f"len(rids)={len(self.rids)} "
                f"len(pool_indices)={len(self.pool_indices)}"
            )
        seen_pool_indices: set[int] = set()
        for pool_idx in self.pool_indices:
            pool_idx = int(pool_idx)
            if pool_idx < 1:
                # Seat 0 is the cuda-graph padding row on the verifier and is
                # never assigned to a live request.
                raise ValueError(
                    "DraftEnumerationBufferBatch pool_indices must be >= 1: "
                    f"batch_size={self.batch_size} pool_idx={pool_idx}"
                )
            if pool_idx in seen_pool_indices:
                raise ValueError(
                    "DraftEnumerationBufferBatch pool_indices must be unique "
                    "(one row per seat): duplicate rows make the verifier-side "
                    "scatter's winner nondeterministic: "
                    f"batch_size={self.batch_size} duplicate_pool_idx={pool_idx}"
                )
            seen_pool_indices.add(pool_idx)
        for i, base_committed_len in enumerate(self.base_committed_lens):
            if int(base_committed_len) < 0:
                raise ValueError(
                    "DraftEnumerationBufferBatch base_committed_lens must be "
                    "non-negative: "
                    f"pool_idx={self.pool_indices[i]} "
                    f"base_committed_len={base_committed_len}"
                )
        if len(self.tokens) != self.num_tokens:
            raise ValueError(
                "DraftEnumerationBufferBatch tokens length must equal "
                "batch_size * (num_steps + 1) * fanout * (num_steps + 1): "
                f"batch_size={self.batch_size} "
                f"num_steps={self.num_steps} fanout={self.fanout} "
                f"expected={self.num_tokens} actual={len(self.tokens)}"
            )


@dataclass
class DraftControlBatch:
    dst_drafter_rank: int
    sync_messages: list[DraftSync] = field(default_factory=list)
    verify_commit_messages: list[VerifyCommit] = field(default_factory=list)
    close_messages: list[DraftClose] = field(default_factory=list)


@dataclass
class VerifierCommitSegment:
    """Contiguous VerifyCommit messages coalesced for one drafter request.

    When receiving contiguous VerifyCommit messages for the same draft req,
    the transport thread(TokenSync thread at drafter side) coalesces them into a single VerifierCommitSegment.

    VerifierCommitSegment represents a contiguous verifier-committed token segment for drafter,
    and drafter scheduler should align with these segments before emitting tail tokens
    """

    draft_key: DraftReqKey
    dst_drafter_rank: int
    pre_verify_committed_len: int
    committed_tokens: list[int] = field(default_factory=list)
    # Per-round deltas within committed_tokens (one entry per coalesced
    # VerifyCommit): lets the consumer take exactly one verify round's worth
    # (generation lockstep) or the whole backlog (catch-up merge).
    round_lens: list[int] = field(default_factory=list)

    @property
    def end_committed_len(self) -> int:
        return int(self.pre_verify_committed_len) + len(self.committed_tokens)

    @property
    def pending_rounds(self) -> int:
        return len(self.round_lens)

    def append_message(self, message: VerifyCommit) -> None:
        """
        It runs on TokenSyncThread under _pending_lock. That loop only
        catches zmq.error.ContextTerminated, so a raise here escapes _run and
        silently kills the drafter control thread. It then stops applying
        ALL requests' controls while the verifier keeps pushing.

        TODO: 1. peer-data violations (non-contiguous / invalid len)
        should quarantine just that request (drop + add to close_keys), not
        crash the thread. 2. phase 5.c will handle the drafter failure by
        degrading the verifier into normal autoregressive decoding.
        """
        if message.draft_key != self.draft_key:
            raise RuntimeError(
                "Verifier commit segment received a commit for a different "
                f"request: segment_key={self.draft_key} message_key={message.draft_key}"
            )
        if int(message.dst_drafter_rank) != int(self.dst_drafter_rank):
            raise RuntimeError(
                "Verifier commit segment received a commit for a different "
                "drafter rank: "
                f"request_id={message.request_id} "
                f"segment_drafter_rank={self.dst_drafter_rank} "
                f"message_drafter_rank={message.dst_drafter_rank}"
            )
        message.validate_committed_tokens()
        pre_verify_committed_len = int(message.pre_verify_committed_len)
        if pre_verify_committed_len != self.end_committed_len:
            raise RuntimeError(
                "Verifier commit segment requires contiguous VerifyCommit "
                "messages: "
                f"request_id={message.request_id} "
                f"expected_pre_verify_committed_len={self.end_committed_len} "
                f"actual_pre_verify_committed_len={pre_verify_committed_len}"
            )

        token_ids = [int(token_id) for token_id in message.committed_tokens]
        self.committed_tokens.extend(token_ids)
        self.round_lens.append(len(token_ids))

    def extract_prefix(self, num_tokens: int) -> VerifierCommitSegment:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(
                "Verifier commit segment prefix length must be positive: "
                f"request_id={self.draft_key.request_id} num_tokens={num_tokens}"
            )
        if num_tokens > len(self.committed_tokens):
            raise ValueError(
                "Verifier commit segment prefix length exceeds segment length: "
                f"request_id={self.draft_key.request_id} "
                f"num_tokens={num_tokens} "
                f"segment_len={len(self.committed_tokens)}"
            )

        prefix_tokens = [
            int(token_id) for token_id in self.committed_tokens[:num_tokens]
        ]
        remaining_tokens = [
            int(token_id) for token_id in self.committed_tokens[num_tokens:]
        ]
        # Split the round boundaries alongside the tokens; a cut inside a
        # round leaves the remainder as a (shorter) leading round.
        prefix_rounds: list[int] = []
        remaining_rounds = list(self.round_lens)
        taken = 0
        while remaining_rounds and taken < num_tokens:
            round_len = remaining_rounds[0]
            if taken + round_len <= num_tokens:
                prefix_rounds.append(remaining_rounds.pop(0))
                taken += round_len
            else:
                head = num_tokens - taken
                prefix_rounds.append(head)
                remaining_rounds[0] = round_len - head
                taken = num_tokens
        prefix_segment = VerifierCommitSegment(
            draft_key=self.draft_key,
            dst_drafter_rank=int(self.dst_drafter_rank),
            pre_verify_committed_len=int(self.pre_verify_committed_len),
            committed_tokens=prefix_tokens,
            round_lens=prefix_rounds,
        )
        self.pre_verify_committed_len = int(self.pre_verify_committed_len) + num_tokens
        self.committed_tokens = remaining_tokens
        self.round_lens = remaining_rounds
        return prefix_segment


@dataclass
class DraftControlInbox:
    """Drafter-side inbox for verifier control messages.

    The TokenSync thread temporarily stores incoming control messages here.
    The drafter scheduler extracts and consumes them each time it finishes a decoding step.
    """

    sync_messages: list[DraftSync] = field(default_factory=list)
    verifier_commit_segments: dict[DraftReqKey, VerifierCommitSegment] = field(
        default_factory=dict
    )
    close_keys: set[DraftReqKey] = field(default_factory=set)

    def is_empty(self) -> bool:
        return (
            not self.sync_messages
            and not self.verifier_commit_segments
            and not self.close_keys
        )

    def pending_control_count(self) -> int:
        return (
            len(self.sync_messages)
            + len(self.verifier_commit_segments)
            + len(self.close_keys)
        )

    def add_control_batch_locked(self, batch: DraftControlBatch) -> None:
        for message in batch.close_messages:
            self.add_close_key_locked(message.draft_key)
        for message in batch.sync_messages:
            if message.draft_key not in self.close_keys:
                self.sync_messages.append(message)
        for message in batch.verify_commit_messages:
            self.add_verify_commit_locked(message)

    def add_close_key_locked(self, draft_key: DraftReqKey) -> None:
        self.close_keys.add(draft_key)
        self.verifier_commit_segments.pop(draft_key, None)
        self.sync_messages = [
            message for message in self.sync_messages if message.draft_key != draft_key
        ]

    def add_verify_commit_locked(self, message: VerifyCommit) -> None:
        if message.draft_key in self.close_keys:
            return
        segment = self.verifier_commit_segments.get(message.draft_key)
        if segment is None:
            segment = VerifierCommitSegment(
                draft_key=message.draft_key,
                dst_drafter_rank=int(message.dst_drafter_rank),
                pre_verify_committed_len=int(message.pre_verify_committed_len),
            )
            segment.append_message(message)
            self.verifier_commit_segments[message.draft_key] = segment
            return
        segment.append_message(message)

    def extract_ready_controls_locked(
        self,
        consumable_commit_len: Callable[[VerifierCommitSegment], int],
    ) -> ReadyDraftControls:
        ready_controls = ReadyDraftControls()

        if self.close_keys:
            ready_controls.close_keys = self.close_keys
            self.close_keys = set()

        if self.sync_messages:
            ready_controls.sync_messages = self.sync_messages
            self.sync_messages = []

        for draft_key, segment in list(self.verifier_commit_segments.items()):
            consumable_len = consumable_commit_len(segment)
            if consumable_len <= 0:
                continue

            ready_controls.ready_commit_segments.append(
                segment.extract_prefix(consumable_len)
            )
            if not segment.committed_tokens:
                self.verifier_commit_segments.pop(draft_key, None)

        return ready_controls


@dataclass
class ReadyDraftControls:
    sync_messages: list[DraftSync] = field(default_factory=list)
    close_keys: set[DraftReqKey] = field(default_factory=set)
    ready_commit_segments: list[VerifierCommitSegment] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.sync_messages
            and not self.close_keys
            and not self.ready_commit_segments
        )

    def extracted_control_count(self) -> int:
        return (
            len(self.sync_messages)
            + len(self.close_keys)
            + len(self.ready_commit_segments)
        )


@dataclass
class DraftMeshMessage:
    message_type: DraftMeshMessageType
    control_batch: Optional[DraftControlBatch] = None
    enumeration_buffer_batch: Optional[DraftEnumerationBufferBatch] = None

    @staticmethod
    def from_control_batch(message: DraftControlBatch) -> DraftMeshMessage:
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.CONTROL_BATCH,
            control_batch=message,
        )

    @staticmethod
    def from_enumeration_buffer_batch(
        message: DraftEnumerationBufferBatch,
    ) -> DraftMeshMessage:
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.ENUMERATION_BUFFER_BATCH,
            enumeration_buffer_batch=message,
        )


@dataclass(frozen=True)
class DecoupledSpecIpcConfig:
    bind_endpoint: str
    connect_endpoints: tuple[str, ...]
    rank: int
