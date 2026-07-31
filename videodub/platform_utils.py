from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


# Keep the existing storage name so upgrading to scip does not lose settings.
APP_DIRECTORY_NAME = "YouTube Video Localizer"


def user_config_dir() -> Path:
    if os.name == "nt":
        root = Path(
            os.environ.get("APPDATA")
            or Path.home() / "AppData" / "Roaming"
        )
        return root / APP_DIRECTORY_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIRECTORY_NAME
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "youtube-video-localizer"


def user_cache_dir() -> Path:
    if os.name == "nt":
        root = Path(
            os.environ.get("LOCALAPPDATA")
            or Path.home() / "AppData" / "Local"
        )
        return root / APP_DIRECTORY_NAME / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / APP_DIRECTORY_NAME
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "youtube-video-localizer"


def application_cache_dir() -> Path:
    configured = os.environ.get("VIDEODUB_CACHE_DIR", "").strip()
    return Path(configured).expanduser() if configured else user_cache_dir()


def _tool_candidates(command: str) -> list[Path]:
    executable = command + (".exe" if os.name == "nt" else "")
    candidates = [Path(sys.executable).resolve().parent / executable]
    if os.name == "nt":
        for root in (
            os.environ.get("LOCALAPPDATA", ""),
            os.environ.get("USERPROFILE", ""),
            os.environ.get("ChocolateyInstall", ""),
        ):
            if not root:
                continue
            base = Path(root)
            candidates.extend(
                [
                    base / "Microsoft" / "WinGet" / "Links" / executable,
                    base / "scoop" / "shims" / executable,
                    base / "bin" / executable,
                ]
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
