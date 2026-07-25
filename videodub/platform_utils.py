from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _tool_candidates(command: str) -> list[Path]:
    executable = command + (".exe" if os.name == "nt" else "")
    candidates = [Path(sys.executable).resolve().parent / executable]
    if os.name == "nt":
        if command == "yt-dlp":
            candidates.insert(0, Path(r"D:\software\yt-dlp.exe"))
        else:
            candidates.insert(
                0, Path(rf"D:\software\ffmpeg\bin\{command}.exe")
            )
    else:
        candidates.extend(
            [
                Path("/opt/homebrew/bin") / executable,
                Path("/usr/local/bin") / executable,
                Path.home() / ".local" / "bin" / executable,
            ]
        )
        if sys.platform == "darwin":
            candidates.extend(
                sorted(
                    (Path.home() / "Library" / "Python").glob(
                        f"*/bin/{executable}"
                    ),
                    reverse=True,
                )
            )
    return candidates


def resolve_executable(configured: str, command: str) -> str:
    """Resolve a configured tool while keeping a useful value for UI errors."""
    configured = configured.strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return str(configured_path.resolve())
        located = shutil.which(configured)
        if located:
            return str(Path(located).resolve())
    located = shutil.which(command)
    if located:
        return str(Path(located).resolve())
    for candidate in _tool_candidates(command):
        if candidate.is_file():
            return str(candidate.resolve())
    return configured or command


def executable_exists(value: str) -> bool:
    return Path(value).expanduser().is_file() or shutil.which(value) is not None


def open_in_file_manager(path: str | Path) -> None:
    target = str(Path(path).resolve())
    if os.name == "nt":
        os.startfile(target)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        command = ["open", target]
    else:
        opener = shutil.which("xdg-open")
        if not opener:
            raise RuntimeError("没有找到 xdg-open，无法打开文件夹。")
        command = [opener, target]
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
