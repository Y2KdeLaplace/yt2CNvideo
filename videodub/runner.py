from __future__ import annotations

import locale
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path


LogFn = Callable[[str], None]
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


class CancelledError(RuntimeError):
    pass


class CommandError(RuntimeError):
    def __init__(self, command: list[str], returncode: int, tail: str):
        super().__init__(f"命令执行失败（退出码 {returncode}）\n{tail}")
        self.command = command
        self.returncode = returncode
        self.tail = tail


class ProcessRunner:
    def __init__(self, logger: LogFn | None = None):
        self.logger = logger or (lambda _message: None)
        self.cancel_event = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        self.cancel_event.clear()

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._lock:
            process = self._process
        if process and process.poll() is None:
            process.terminate()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError("任务已由用户停止")

    def run(
        self,
        command: Iterable[str | Path],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        input_text: str | None = None,
        quiet: bool = False,
    ) -> list[str]:
        self.check_cancelled()
        args = [str(part) for part in command]
        if not quiet:
            self.logger("$ " + subprocess.list2cmdline(args))
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            creationflags=creationflags,
            env={**os.environ, **env} if env is not None else None,
        )
        with self._lock:
            self._process = process
        lines: list[str] = []
        last_progress_log = 0.0

        def emit(raw: bytes, is_progress: bool) -> None:
            nonlocal last_progress_log
            raw = raw.rstrip()
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                line = raw.decode(
                    locale.getpreferredencoding(False),
                    errors="replace",
                )
            line = ANSI_ESCAPE.sub("", line)
            if not is_progress:
                lines.append(line)
            if not line or quiet:
                return
            now = time.monotonic()
            if is_progress and now - last_progress_log < 0.15:
                return
            self.logger(line)
            if is_progress:
                last_progress_log = now

        try:
            if input_text is not None and process.stdin:
                process.stdin.write(input_text.encode("utf-8"))
                process.stdin.close()
            assert process.stdout is not None
            pending = b""
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    break
                pending += chunk
                while True:
                    carriage = pending.find(b"\r")
                    newline = pending.find(b"\n")
                    endings = [index for index in (carriage, newline) if index >= 0]
                    if not endings:
                        break
                    ending = min(endings)
                    is_carriage = pending[ending : ending + 1] == b"\r"
                    is_crlf = is_carriage and pending[ending + 1 : ending + 2] == b"\n"
                    is_progress = is_carriage and not is_crlf
                    emit(pending[:ending], is_progress)
                    pending = pending[ending + (2 if is_crlf else 1) :]
                    if self.cancel_event.is_set():
                        process.terminate()
                        raise CancelledError("Task cancelled")
            if pending:
                emit(pending, False)
            returncode = process.wait()
        finally:
            if process.stdout:
                process.stdout.close()
            with self._lock:
                self._process = None
        if returncode:
            raise CommandError(args, returncode, "\n".join(lines[-20:]))
        return lines
