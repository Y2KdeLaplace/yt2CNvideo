from __future__ import annotations

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


def _update_homebrew_dependencies(runner: ProcessRunner) -> str:
    brew = resolve_executable("brew", "brew")
    if not executable_exists(brew):
        return "依赖自动更新：未找到 Homebrew，已跳过 ffmpeg 与 yt-dlp。"
    try:
        runner.run(
            [brew, "upgrade", "--formula", "--no-ask", "ffmpeg", "yt-dlp"]
        )
    except CommandError as exc:
        return f"依赖自动更新未完成：Homebrew 退出码 {exc.returncode}，详情见运行日志。"
    return "依赖自动更新完成：Homebrew 已处理 ffmpeg 与 yt-dlp。"


def _update_winget_dependencies(runner: ProcessRunner) -> str:
    winget = resolve_executable("winget", "winget")
    if not executable_exists(winget):
        return "依赖自动更新：未找到 winget，已跳过 ffmpeg 与 yt-dlp。"
    failed: list[str] = []
    for package_id, name in (
        ("Gyan.FFmpeg", "ffmpeg"),
        ("yt-dlp.yt-dlp", "yt-dlp"),
    ):
        try:
            runner.run(
                [
                    winget,
                    "upgrade",
                    "--id",
                    package_id,
                    "--exact",
                    "--source",
                    "winget",
                    "--silent",
                    "--accept-source-agreements",
                    "--accept-package-agreements",
                    "--disable-interactivity",
                ]
            )
        except CommandError:
            failed.append(name)
    if failed:
        return (
            "依赖自动更新未完成："
            f"{', '.join(failed)} 没有可用更新或 winget 执行失败，详情见运行日志。"
        )
    return "依赖自动更新完成：winget 已处理 ffmpeg 与 yt-dlp。"


def update_dependencies(
    runner: ProcessRunner,
    *,
    platform_name: str | None = None,
) -> str:
    current = platform_name or ("windows" if os.name == "nt" else sys.platform)
    if current == "darwin":
        return _update_homebrew_dependencies(runner)
    if current in {"nt", "win32", "windows"}:
        return _update_winget_dependencies(runner)
    return "依赖自动更新：当前平台未配置包管理器，请手动更新 ffmpeg 与 yt-dlp。"
