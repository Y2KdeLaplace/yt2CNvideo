from __future__ import annotations

import locale
import os
import subprocess
import threading
from collections.abc import Callable, Iterable
from pathlib import Path


LogFn = Callable[[str], None]


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
        )
        with self._lock:
            self._process = process
        lines: list[str] = []
        try:
            if input_text is not None and process.stdin:
                process.stdin.write(input_text.encode("utf-8"))
                process.stdin.close()
            assert process.stdout is not None
            for raw_line in process.stdout:
                raw_line = raw_line.rstrip()
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw_line.decode(
                        locale.getpreferredencoding(False),
                        errors="replace",
                    )
                lines.append(line)
                if line and not quiet:
                    self.logger(line)
                if self.cancel_event.is_set():
                    process.terminate()
                    raise CancelledError("任务已由用户停止")
            returncode = process.wait()
        finally:
            with self._lock:
                self._process = None
        if returncode:
            raise CommandError(args, returncode, "\n".join(lines[-20:]))
        return lines
