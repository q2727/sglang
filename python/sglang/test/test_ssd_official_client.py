from __future__ import annotations

import socket
import tempfile
import threading
import unittest
from pathlib import Path

from sglang.srt.speculative.ssd_draft_client import DraftCandidate
from sglang.srt.speculative.ssd_official_client import OfficialSSDDraftClient
from sglang.srt.speculative.ssd_official_protocol import (
    FLAG_RESPONSE,
    BufferReader,
    BufferWriter,
    OfficialSSDOp,
    recv_frame,
    send_frame,
)


class FakeOfficialSSDServer:
    def __init__(self, path: Path):
        self.path = path
        self.calls = []
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(timeout=5):
            raise RuntimeError("fake server did not start")

    @staticmethod
    def _candidate(tokens, cache_hit):
        return BufferWriter().u8(cache_hit).f64(1.25).int_list(tokens).finish()

    def _run(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(self.path))
            listener.listen(1)
            self.ready.set()
            conn, _ = listener.accept()
            with conn:
                while True:
                    try:
                        op, flags, payload = recv_frame(conn)
                    except EOFError:
                        return
                    self.calls.append((op, payload))
                    self.assert_request_flags(flags)
                    if op == OfficialSSDOp.RESET:
                        response = b""
                    elif op == OfficialSSDOp.INIT:
                        response = self._candidate([7, 8], False)
                    elif op == OfficialSSDOp.BUILD:
                        response = (
                            BufferWriter()
                            .u32(3)
                            .f64(2.0)
                            .f64(3.0)
                            .f64(0.1)
                            .f64(5.1)
                            .finish()
                        )
                    elif op == OfficialSSDOp.SELECT:
                        response = self._candidate([9, 10], True)
                    else:
                        raise AssertionError(f"unexpected op {op}")
                    send_frame(conn, op, response, flags=FLAG_RESPONSE)

    @staticmethod
    def assert_request_flags(flags):
        if flags:
            raise AssertionError(f"request flags were {flags}")


class TestOfficialSSDDraftClient(unittest.TestCase):
    def test_compact_init_build_select_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = Path(tmpdir) / "draft.sock"
            server = FakeOfficialSSDServer(socket_path)
            server.start()
            client = OfficialSSDDraftClient(f"unix://{socket_path}", timeout=5)

            client.reset()
            initial = client.init_draft("req-1", [1, 2, 3], 2, (1, 1, 1))
            self.assertEqual(
                initial,
                DraftCandidate(
                    tokens=[7, 8],
                    recovery_tokens=((), (), ()),
                    cache_hit=False,
                ),
            )
            handle = client.build_outcome_cache(
                "req-1", [1, 2, 3], initial, 2, (1, 1, 1)
            )
            self.assertEqual(handle.branches, 3)
            self.assertAlmostEqual(handle.total_ms, 5.1)
            selected = client.select_outcome(
                handle,
                "req-1",
                [1, 2, 3, 7, 4],
                (1, 4),
                2,
                (1, 1, 1),
            )
            self.assertEqual(selected.tokens, [9, 10])
            self.assertTrue(selected.cache_hit)
            client.close()

            ops = [op for op, _ in server.calls]
            self.assertEqual(
                ops,
                [
                    OfficialSSDOp.RESET,
                    OfficialSSDOp.INIT,
                    OfficialSSDOp.BUILD,
                    OfficialSSDOp.SELECT,
                ],
            )

            build_reader = BufferReader(server.calls[2][1])
            self.assertEqual(build_reader.text(), "req-1")
            self.assertEqual(build_reader.i64(), 3)
            self.assertEqual(build_reader.i32(), 3)
            self.assertFalse(build_reader.u8())
            self.assertEqual(build_reader.int_list(), [7, 8])
            build_reader.finish()


if __name__ == "__main__":
    unittest.main()
