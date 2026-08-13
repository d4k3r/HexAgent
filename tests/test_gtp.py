from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from hex_reconstruction.gtp import (
    GTPClient,
    GTPProtocolError,
    GTPResponse,
    GTPTimeout,
    MoveKind,
    parse_completed_analysis,
)


class GTPTests(unittest.TestCase):
    def test_framing_ids_errors_and_raw_logs(self) -> None:
        engine = Path(__file__).with_name("fake_gtp_engine.py")
        with tempfile.TemporaryDirectory() as directory:
            log_directory = Path(directory)
            with GTPClient((sys.executable, str(engine)), log_directory=log_directory) as client:
                self.assertEqual(client.command("name").payload, "FakeHex")
                self.assertEqual(client.command("two").payload, "first")
                self.assertEqual(client.command("prebuffered").payload, "trailing")
                with self.assertRaises(GTPTimeout):
                    client.command("hang", timeout=0.01)
                with self.assertRaises(GTPProtocolError):
                    client.command("error")
            self.assertIn(b"startup banner", (log_directory / "stdout.raw").read_bytes())
            self.assertIn(b"1 name", (log_directory / "stdin.raw").read_bytes())

    def test_completed_kata_analysis_parser(self) -> None:
        raw = (
            "=7\n"
            "info move pass visits 5 utility 1 winrate 1 prior 0.8 lcb 0.9 utilityLcb 0.8 order 0 pv pass "
            "info move f6 visits 3 utility 0.7 winrate 0.85 prior 0.1 lcb 0.6 utilityLcb 0.5 order 1 pv f6 e6 pvVisits 3 2 "
            "info move e6 visits 2 utility 0.6 winrate 0.8 prior 0.1 lcb 0.5 order 2 pv e6\n"
            "play pass\n\n"
        )
        result = parse_completed_analysis(GTPResponse(7, True, raw[3:-2], raw))
        self.assertEqual(result.chosen_move_kind, MoveKind.PASS)
        self.assertEqual(len(result.candidates), 3)
        physical = result.candidates[1]
        self.assertEqual(physical.move, "f6")
        self.assertEqual(physical.visits, 3)
        self.assertEqual(physical.utility_lcb, 0.5)
        self.assertEqual(physical.order, 1)
        self.assertEqual(physical.pv, ("f6", "e6"))

    def test_dead_child_exposes_exit_status_and_stderr_not_timeout_only(self) -> None:
        engine = Path(__file__).with_name("fake_gtp_engine.py")
        with tempfile.TemporaryDirectory() as directory:
            with GTPClient((sys.executable, str(engine)), log_directory=Path(directory)) as client:
                with self.assertRaisesRegex(Exception, "deliberate fake startup/process crash") as raised:
                    client.command("crash", timeout=1.0)
                self.assertIn("returncode", str(raised.exception))
                self.assertEqual(client.process.poll(), 23)

    def test_timeout_keeps_normal_command_timeout_semantics(self) -> None:
        engine = Path(__file__).with_name("fake_gtp_engine.py")
        with tempfile.TemporaryDirectory() as directory:
            with GTPClient((sys.executable, str(engine)), log_directory=Path(directory)) as client:
                with self.assertRaisesRegex(GTPTimeout, "after 0.010s"):
                    client.command("hang", timeout=0.01)


if __name__ == "__main__":
    unittest.main()
