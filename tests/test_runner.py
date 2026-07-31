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

        runner.run(
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

