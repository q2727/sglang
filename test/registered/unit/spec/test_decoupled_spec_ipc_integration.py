"""In-process integration test for the decoupled enumeration-spec IPC layer.

Wires the verifier-side VerifierIpcThread (recv daemon) + drafter-side
DrafterIpcThread over the fake transport, all in one process, and drives them
deterministically via ``_step()`` (no background threads, no GPU, no sockets).
The verifier lands received enumeration blocks into a CPU stand-in for the GPU
DecoupledEnumBuffer (real ingest prologue: verifier routing + validate, no torch
scatter); the drafter drains its control inbox. Exercises the whole open ->
enumerate -> land -> commit -> close loop end to end.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftEnumerationBufferBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    VerifyCommit,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    DecoupledSpecTransportKind,
    FakeTransportMesh,
    build_transport,
)
from sglang.srt.speculative.drafter_ipc_thread import DrafterIpcThread
from sglang.srt.speculative.decoupled_verify_manager import DecoupledVerifyManager
from sglang.srt.speculative.verifier_ipc_thread import VerifierIpcThread
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

V_EP = "ipc:///tmp/decoupled-spec-itest-v"
D_EP = "ipc:///tmp/decoupled-spec-itest-d"


def _block(pool_idx=1, dst=0, tok=100, epoch=1) -> DraftEnumerationBufferBatch:
    # Minimal K=1, F=1 enumeration block: row_stride = (K+1)*F*(K+1) = 4, one row.
    return DraftEnumerationBufferBatch(
        src_drafter_rank=0,
        dst_verifier_rank=dst,
        num_steps=1,
        fanout=1,
        pool_indices=[pool_idx],
        base_committed_lens=[0],
        epochs=[epoch],
        tokens=(tok, tok + 1, tok + 2, tok + 3),
    )


class _FakeEnumBuffer:
    """CPU stand-in for DecoupledEnumBuffer's ``land`` (no torch scatter here).

    Runs the real host-side ingest prologue -- the wrong-verifier reject, the
    K/F dims guard, and ``block.validate()`` -- then records the block instead
    of doing the GPU scatter. Faithful for the IPC layer: the wrong-verifier
    raise the daemon tests assert originates in this prologue, exactly as in
    the real ``DecoupledEnumBuffer.land``.
    """

    def __init__(self, *, verifier_rank=0, num_steps=1, fanout=1):
        self.verifier_rank = int(verifier_rank)
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        self.landed = []  # list[DraftEnumerationBufferBatch]

    def land(self, block):
        if int(block.dst_verifier_rank) != self.verifier_rank:
            raise RuntimeError("enumeration block routed to the wrong verifier")
        if int(block.num_steps) != self.num_steps or int(block.fanout) != self.fanout:
            raise RuntimeError("enumeration block dims differ from the buffer's config")
        block.validate()
        self.landed.append(block)


def _drain_sync_and_close(token):
    # Simulate the drafter scheduler draining its inbox; commit segments are not
    # consumed here (consumable length 0).
    return token.collect_ready_draft_controls(
        lambda inbox: inbox.extract_ready_controls_locked(lambda seg: 0)
    )


def _drain_all(token):
    # Drain everything including full commit segments.
    return token.collect_ready_draft_controls(
        lambda inbox: inbox.extract_ready_controls_locked(
            lambda seg: len(seg.committed_tokens)
        )
    )


class TestDecoupledSpecIpcIntegration(CustomTestCase):
    def test_desync_reseed_preserves_initial_prompt_snapshot(self):
        reset_slots = []
        manager = DecoupledVerifyManager.__new__(DecoupledVerifyManager)
        manager.ipc_config = SimpleNamespace(rank=0)
        manager.num_drafters = 1
        manager.verify_worker = SimpleNamespace(
            enum_buffer=SimpleNamespace(reset_slot=reset_slots.append)
        )
        manager.arrival_board = SimpleNamespace(reset_seat=lambda _seat: None)
        manager._rid_states = {}
        manager._seat_epochs = {}
        manager._gate_expected = {}
        manager._seat_timeout_streaks = {}
        manager._drafter_timeout_streaks = {}
        manager._drafter_cooldown_until = {}
        manager._force_resync_seats = set()

        req = SimpleNamespace(
            rid="r",
            req_pool_idx=3,
            origin_input_ids=[10, 11, 12, 13],
            output_ids=[],
            multimodal_inputs=None,
            finished=lambda: False,
        )
        controls = {}
        manager._collect_req_controls(req, controls, overlap=True)
        first = controls[0].sync_messages[0]
        self.assertEqual(first.prompt_token_ids, [10, 11, 12, 13])

        # The live Req is mutable under overlap scheduling. A recovery sync
        # must still use the original prompt rather than this restaged view.
        req.origin_input_ids[:] = [12, 13]
        req.output_ids[:] = [20, 21]
        manager._force_resync_seats.add(3)
        controls = {}
        manager._collect_req_controls(req, controls, overlap=True)
        reseed = controls[0].sync_messages[0]
        self.assertEqual(reseed.prompt_token_ids, [10, 11, 12, 13])
        self.assertEqual(manager._rid_states["r"].prompt_len, 4)

        # Recovery adopts the exact prefix selected by the IPC re-seed rather
        # than retaining a stale scheduler-side committed length.
        reseed.committed_outputs[:] = [20]
        manager._on_resync_sent(reseed)
        self.assertEqual(manager._rid_states["r"].committed_len, 1)
        self.assertEqual(manager._gate_expected[3], 5)

        manager._gate_expected[3] = 999  # deliberately stale inferred value
        expected = manager._collect_gate_expected(
            SimpleNamespace(reqs=[req], seq_lens_cpu=torch.tensor([17]))
        )
        self.assertEqual(expected, {3: 18})

        controls = {}
        manager._collect_req_controls(req, controls, overlap=True)
        commit = controls[0].verify_commit_messages[0]
        self.assertEqual(commit.pre_verify_committed_len, 1)
        self.assertEqual(commit.committed_tokens, [21])

    def test_epoch_filter_rejects_old_inflight_block(self):
        manager = DecoupledVerifyManager.__new__(DecoupledVerifyManager)
        manager._seat_epochs = {3: 2}
        manager._stale_epoch_drop_ct = 0

        self.assertIsNone(manager._filter_block_epoch(_block(pool_idx=3, epoch=1)))
        current = _block(pool_idx=3, epoch=2)
        self.assertIs(manager._filter_block_epoch(current), current)
        self.assertEqual(manager._stale_epoch_drop_ct, 1)

    def test_desync_reseed_uses_wire_ledger_high_water(self):
        _mesh, v_tp, d_tp, _enum_buffer, proxy, token = self._wire()
        adopted = []
        proxy._on_resync_sent = adopted.append
        proxy._sent_committed_lens["r"] = 3
        proxy._sent_committed_outputs["r"] = [10, 11, 12]
        try:
            proxy.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    sync_messages=[
                        DraftSync(
                            request_id="r",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            req_pool_idx=3,
                            epoch=2,
                            committed_outputs=[10],
                            desync_reseed=True,
                        )
                    ],
                )
            )
            proxy._step()
            token._step()
            ready = _drain_sync_and_close(token)
            self.assertEqual(ready.sync_messages[0].committed_outputs, [10, 11, 12])
            self.assertEqual(adopted[0].committed_outputs, [10, 11, 12])
        finally:
            v_tp.close()
            d_tp.close()

    def test_desync_reseed_uses_scheduler_high_water_when_wire_trails(self):
        _mesh, v_tp, d_tp, _enum_buffer, proxy, token = self._wire()
        adopted = []
        proxy._on_resync_sent = adopted.append
        proxy._sent_committed_lens["r"] = 2
        proxy._sent_committed_outputs["r"] = [10, 11]
        try:
            proxy.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    sync_messages=[
                        DraftSync(
                            request_id="r",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            req_pool_idx=3,
                            epoch=2,
                            committed_outputs=[10, 11, 12, 13],
                            desync_reseed=True,
                        )
                    ],
                )
            )
            proxy._step()
            token._step()
            ready = _drain_sync_and_close(token)
            self.assertEqual(
                ready.sync_messages[0].committed_outputs, [10, 11, 12, 13]
            )
            self.assertEqual(adopted[0].committed_outputs, [10, 11, 12, 13])
            self.assertEqual(proxy._resync_floors["r"], 4)
        finally:
            v_tp.close()
            d_tp.close()

    def _wire(self):
        mesh = FakeTransportMesh()
        v_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=V_EP,
            connect_endpoints=[D_EP],
            mesh=mesh,
        )
        d_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=D_EP,
            connect_endpoints=[V_EP],
            mesh=mesh,
        )
        v_tp.start()
        d_tp.start()
        enum_buffer = _FakeEnumBuffer(verifier_rank=0, num_steps=1, fanout=1)
        proxy = VerifierIpcThread(transport=v_tp, enum_buffer=enum_buffer)
        token = DrafterIpcThread(transport=d_tp, drafter_rank=0)
        return mesh, v_tp, d_tp, enum_buffer, proxy, token

    def test_full_loop_open_enumerate_land_commit_close(self):
        _mesh, v_tp, d_tp, enum_buffer, proxy, token = self._wire()
        try:
            # 1. verifier opens a draft request (DraftSync), announcing the seat
            #    (req_pool_idx) the drafter must echo in every enumeration row.
            proxy.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    sync_messages=[
                        DraftSync(
                            request_id="r",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            req_pool_idx=3,
                            epoch=1,
                            committed_outputs=[],
                        )
                    ],
                )
            )
            proxy._step()  # forward over the transport

            # 2. drafter receives the control into its inbox; the announced seat
            #    rides on the sync message.
            token._step()
            ready = _drain_sync_and_close(token)
            self.assertEqual([m.request_id for m in ready.sync_messages], ["r"])
            self.assertEqual([m.req_pool_idx for m in ready.sync_messages], [3])

            # 3. drafter pushes one enumeration block back, echoing the seat.
            token.submit_draft_results(_block(pool_idx=3, dst=0, tok=100))
            token._step()  # send to verifier

            # 4. verifier lands the block into its (fake) enum buffer at the
            #    echoed seat.
            proxy._step()
            self.assertEqual(len(enum_buffer.landed), 1)
            landed_block = enum_buffer.landed[0]
            self.assertEqual(landed_block.pool_indices, [3])

            # 5. verifier commits the token; drafter sees the committed segment.
            proxy.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    verify_commit_messages=[
                        VerifyCommit(
                            request_id="r",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            pre_verify_committed_len=0,
                            committed_tokens=[100],
                        )
                    ],
                )
            )
            proxy._step()
            token._step()
            ready2 = _drain_all(token)
            self.assertEqual(len(ready2.ready_commit_segments), 1)
            self.assertEqual(ready2.ready_commit_segments[0].committed_tokens, [100])
            self.assertEqual(proxy._sent_committed_outputs["r"], [100])

            # 6. verifier closes the request; a late block for the retired seat
            #    would land but stays on the fallback path via its stale
            #    base_committed_len stamp (plus reset_slot at seat reuse).
            proxy.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    close_messages=[
                        DraftClose(
                            request_id="r",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            reason="finished",
                        )
                    ],
                )
            )
            proxy._step()
            token._step()
            ready3 = _drain_sync_and_close(token)
            self.assertEqual(len(ready3.close_keys), 1)
        finally:
            v_tp.close()
            d_tp.close()

    def test_token_drops_control_for_wrong_drafter_rank(self):
        # token is drafter rank 0.
        _mesh, v_tp, d_tp, _enum_buffer, _proxy, token = self._wire()
        try:
            # Inject a control batch addressed to drafter rank 5 onto drafter 0's wire.
            v_tp.send(
                0,
                DraftMeshMessage.from_control_batch(
                    DraftControlBatch(
                        dst_drafter_rank=5,
                        sync_messages=[
                            DraftSync(
                                request_id="x",
                                src_verifier_rank=0,
                                dst_drafter_rank=5,
                                req_pool_idx=1,
                                epoch=1,
                                committed_outputs=[],
                            )
                        ],
                    )
                ),
            )
            token._step()
            ready = _drain_sync_and_close(token)
            self.assertEqual(ready.sync_messages, [])  # dropped: wrong drafter rank
        finally:
            v_tp.close()
            d_tp.close()

    def test_drafter_sends_each_block_to_its_verifier(self):
        # An enumeration block carries a single dst_verifier_rank, so a drafter
        # serving two verifiers submits one block per verifier and each routes to
        # its own peer (no per-row grouping).
        mesh = FakeTransportMesh()
        v0_ep = "ipc:///tmp/ds-fanout-v0"
        v1_ep = "ipc:///tmp/ds-fanout-v1"
        d_ep = "ipc:///tmp/ds-fanout-d"
        d_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=d_ep,
            connect_endpoints=[v0_ep, v1_ep],
            mesh=mesh,
        )
        v0_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=v0_ep,
            connect_endpoints=[d_ep],
            mesh=mesh,
        )
        v1_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=v1_ep,
            connect_endpoints=[d_ep],
            mesh=mesh,
        )
        d_tp.start()
        v0_tp.start()
        v1_tp.start()
        token = DrafterIpcThread(transport=d_tp, drafter_rank=0)
        try:
            token.submit_draft_results(_block(pool_idx=1, dst=0, tok=10))
            token.submit_draft_results(_block(pool_idx=2, dst=1, tok=20))
            token._step()
            m0 = v0_tp.try_recv()
            m1 = v1_tp.try_recv()
            self.assertEqual(m0.enumeration_buffer_batch.pool_indices, [1])
            self.assertEqual(m0.enumeration_buffer_batch.tokens[0], 10)
            self.assertEqual(m1.enumeration_buffer_batch.pool_indices, [2])
            self.assertEqual(m1.enumeration_buffer_batch.tokens[0], 20)
        finally:
            d_tp.close()
            v0_tp.close()
            v1_tp.close()

    def test_proxy_rejects_block_for_wrong_verifier_rank(self):
        # proxy is verifier rank 0.
        _mesh, v_tp, d_tp, _enum_buffer, proxy, _token = self._wire()
        try:
            # Inject a block addressed to verifier rank 9 onto verifier 0's wire.
            d_tp.send(
                0,
                DraftMeshMessage.from_enumeration_buffer_batch(
                    _block(pool_idx=1, dst=9, tok=1)
                ),
            )
            # Router-level invariant: land() (driven here via _step) rejects a
            # block for the wrong verifier. Production containment of that raise
            # by the daemon loop is covered by the _run test below.
            with self.assertRaises(RuntimeError):
                proxy._step()
        finally:
            v_tp.close()
            d_tp.close()

    def test_proxy_run_terminates_loudly_on_wrong_verifier_rank(self):
        # The daemon loop must NOT let a router RuntimeError escape and silently
        # kill the proxy for all requests: _run logs it and breaks cleanly.
        # (Phase 5c will quarantine the offending request instead.)
        _mesh, v_tp, d_tp, _enum_buffer, proxy, _token = self._wire()
        try:
            d_tp.send(
                0,
                DraftMeshMessage.from_enumeration_buffer_batch(
                    _block(pool_idx=1, dst=9, tok=1)
                ),
            )
            # _run returns (breaks) instead of propagating, and logs loudly.
            with self.assertLogs(
                "sglang.srt.speculative.verifier_ipc_thread", level="ERROR"
            ):
                proxy._run()
        finally:
            v_tp.close()
            d_tp.close()

    def test_evented_blocks_gate_on_head_event_and_send_in_fifo_order(self):
        # Evented push protocol semantics: per-seat generation rotation on the
        # verifier requires blocks to hit the wire in submit order, and a
        # block's payload may only be read once its staging copy's event
        # fired. A "look-ahead" rewrite that sends any ready block past a
        # not-yet-ready head would reorder generations; a rewrite that skips
        # the event gate would send a torn payload. Both must fail here.
        import torch

        from sglang.srt.speculative.drafter_ipc_thread import EventedDraftBlock

        class _FakeEvent:
            def __init__(self):
                self.ready = False

            def query(self):
                return self.ready

        _mesh, v_tp, d_tp, enum_buffer, proxy, token = self._wire()
        try:
            head_event = _FakeEvent()
            head = _block(pool_idx=1, dst=0, tok=0)
            head.tokens = ()
            released = []
            token.submit_evented_draft_results(
                EventedDraftBlock(
                    header=head,
                    event=head_event,
                    buffer=torch.tensor([100, 101, 102, 103, 999]),
                    num_tokens=4,
                    on_sent=lambda: released.append("head"),
                )
            )
            # Second block is ready immediately (event=None, inline payload).
            token.submit_evented_draft_results(
                EventedDraftBlock(
                    header=_block(pool_idx=2, dst=0, tok=200),
                    event=None,
                    buffer=None,
                    num_tokens=0,
                    on_sent=None,
                )
            )
            # Head not ready: NOTHING may ship (not even the ready second).
            token._step()
            proxy._step()
            self.assertEqual(len(enum_buffer.landed), 0)
            self.assertEqual(released, [])

            head_event.ready = True
            token._step()
            proxy._step()
            self.assertEqual([b.pool_indices for b in enum_buffer.landed], [[1], [2]])
            # Payload materialized from the first num_tokens of the buffer.
            self.assertEqual(enum_buffer.landed[0].tokens, (100, 101, 102, 103))
            self.assertEqual(released, ["head"])
        finally:
            v_tp.close()
            d_tp.close()

    def test_token_run_terminates_loudly_on_malformed_control(self):
        # Mirror of the proxy test on the drafter side: a malformed control
        # envelope makes _route_control_message raise; _run must contain it.
        _mesh, v_tp, d_tp, _enum_buffer, _proxy, token = self._wire()
        try:
            # CONTROL_BATCH message_type with a None payload is malformed.
            v_tp.send(
                0,
                DraftMeshMessage(
                    message_type=DraftMeshMessageType.CONTROL_BATCH,
                    control_batch=None,
                ),
            )
            with self.assertLogs(
                "sglang.srt.speculative.drafter_ipc_thread", level="ERROR"
            ):
                token._run()
        finally:
            v_tp.close()
            d_tp.close()


if __name__ == "__main__":
    unittest.main()
