from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from .config import AppConfig
from .platform_utils import executable_exists, resolve_executable
from .runner import CommandError, ProcessRunner


@dataclass(frozen=True)
class DependencyVersions:
    yt_dlp: str
    ffmpeg: str
    ffprobe: str

    def log_line(self) -> str:
        return (
            f"依赖检查通过：yt-dlp {self.yt_dlp}；"
            f"ffmpeg {self.ffmpeg}；ffprobe {self.ffprobe}。"
        )


def _version_line(
    runner: ProcessRunner,
    command: list[str],
    *,
    prefix: str = "",
) -> str:
    lines = runner.run(command, quiet=True)
    first = next((line.strip() for line in lines if line.strip()), "")
    if not first:
        raise RuntimeError(f"无法读取 {command[0]} 的版本")
    if prefix and first.lower().startswith(prefix.lower()):
        first = first[len(prefix) :].strip()
    return first


def inspect_dependency_versions(
    config: AppConfig,
    runner: ProcessRunner,
) -> DependencyVersions:
    problems = config.validate_core()
    if problems:
        raise RuntimeError("\n".join(problems))
    yt_dlp = _version_line(runner, [config.yt_dlp_path, "--version"])
    ffmpeg = _version_line(
        runner,
        [config.ffmpeg_path, "-version"],
        prefix="ffmpeg version",
    ).split()[0]
    ffprobe = _version_line(
        runner,
        [config.ffprobe_path, "-version"],
        prefix="ffprobe version",
    ).split()[0]
    return DependencyVersions(yt_dlp, ffmpeg, ffprobe)


def _check_homebrew_updates(runner: ProcessRunner) -> str:
    brew = resolve_executable("brew", "brew")
    if not executable_exists(brew):
        return "依赖更新：未找到 Homebrew，无法检查 ffmpeg 与 yt-dlp。"
    lines = runner.run(
        [brew, "outdated", "--formula", "--json=v2", "yt-dlp", "ffmpeg"],
        quiet=True,
    )
    try:
        payload = json.loads("\n".join(lines) or "{}")
        formulae = payload.get("formulae", [])
        outdated = {
            str(item.get("name")): str(item.get("current_version") or "有新版本")
            for item in formulae
            if isinstance(item, dict) and item.get("name")
        }
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"无法解析 Homebrew 更新结果：{exc}") from exc
    if not outdated:
        return "依赖更新：Homebrew 中的 ffmpeg、yt-dlp 已是最新。"
    details = "；".join(
        f"{name} → {outdated[name]}"
        for name in ("ffmpeg", "yt-dlp")
        if name in outdated
    )
    return f"依赖更新：Homebrew 可更新 {details}（运行 brew upgrade ffmpeg yt-dlp）。"


def _check_winget_updates(runner: ProcessRunner) -> str:
    winget = resolve_executable("winget", "winget")
    if not executable_exists(winget):
        return "依赖更新：未找到 winget，无法检查 ffmpeg 与 yt-dlp。"
    command = [
        winget,
        "list",
        "--upgrade-available",
        "--source",
        "winget",
        "--accept-source-agreements",
        "--disable-interactivity",
    ]
    try:
        lines = runner.run(command)
    except CommandError as exc:
        # Some winget versions use a non-zero code when the update list is empty.
        runner.logger(exc.tail)
        lines = exc.tail.splitlines()
    output = "\n".join(lines).lower()
    available = [
        name
        for package_id, name in (
            ("gyan.ffmpeg", "ffmpeg"),
            ("yt-dlp.yt-dlp", "yt-dlp"),
        )
        if package_id in output
    ]
    if available:
        return f"依赖更新：winget 可更新 {', '.join(available)}，详情见运行日志。"
    return "依赖更新：winget 未列出 ffmpeg 或 yt-dlp 的可用更新。"


def check_dependency_updates(
    runner: ProcessRunner,
    *,
    platform_name: str | None = None,
) -> str:
    current = platform_name or ("windows" if os.name == "nt" else sys.platform)
    if current == "darwin":
        return _check_homebrew_updates(runner)
    if current in {"nt", "win32", "windows"}:
        return _check_winget_updates(runner)
    return "依赖更新：当前平台未配置包管理器检查，请手动更新 ffmpeg 与 yt-dlp。"
