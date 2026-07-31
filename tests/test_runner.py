from __future__ import annotations

import sys
import time
import unittest

from videodub.runner import ProcessRunner


class ProcessRunnerTests(unittest.TestCase):
    def test_carriage_return_progress_is_logged_before_process_exits(self) -> None:
        logged: list[tuple[str, float]] = []
        started = time.monotonic()
        runner = ProcessRunner(lambda line: logged.append((line, time.monotonic())))

        lines = runner.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, time; "
                    "sys.stdout.write('25%\\r'); sys.stdout.flush(); "
                    "time.sleep(0.4); print('done')"
                ),
            ]
        )

        progress_time = next(timestamp for text, timestamp in logged if text == "25%")
        self.assertLess(progress_time - started, 0.3)
        self.assertIn("done", [text for text, _timestamp in logged])
        self.assertEqual(lines, ["done"])

    def test_carriage_return_progress_uses_progress_logger(self) -> None:
        logged: list[str] = []
        progress: list[str] = []
        runner = ProcessRunner(logged.append, progress_logger=progress.append)

        runner.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('10%\\r'); print('done')",
            ]
        )

        self.assertIn("10%", progress)
        self.assertNotIn("10%", logged)
        self.assertIn("done", logged)

    def test_ansi_terminal_codes_are_removed_from_logs_and_output(self) -> None:
        logged: list[str] = []
        runner = ProcessRunner(logged.append)

        lines = runner.run(
            [sys.executable, "-c", "print('\\x1b[31merror\\x1b[0m')"]
        )

        self.assertEqual(lines, ["error"])
        self.assertIn("error", logged)

    def test_cancel_callbacks_are_called_and_can_be_removed(self) -> None:
        called: list[str] = []
        runner = ProcessRunner()
        retained = lambda: called.append("retained")
        removed = lambda: called.append("removed")
        runner.add_cancel_callback(retained)
        runner.add_cancel_callback(removed)
        runner.remove_cancel_callback(removed)

        runner.cancel()

        self.assertEqual(called, ["retained"])
