from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import AppConfig
from .media import VideoJob, discover_video_jobs
from .runner import CommandError, ProcessRunner
from .subtitles import find_source_subtitle


ID_MARKER = "VIDEODUB_ID:"


def snapshot_download_directories(config: AppConfig) -> set[Path]:
    root = Path(config.download_dir).resolve()
    return {
        item.resolve()
        for item in root.iterdir()
        if item.is_dir() and item.name != "output"
    }


def cleanup_new_download_directories(
    config: AppConfig,
    before: set[Path],
    runner: ProcessRunner,
) -> None:
    root = Path(config.download_dir).resolve()
    for item in root.iterdir():
        if not item.is_dir() or item.name == "output":
            continue
        resolved = item.resolve()
        if resolved in before:
            continue
        resolved.relative_to(root)
        shutil.rmtree(resolved)
        runner.logger(f"已删除未完成的下载目录：{resolved.name}")


def build_download_command(
    config: AppConfig,
    url: str,
    *,
    include_subtitles: bool | None = None,
    automatic_subtitles: bool = False,
    skip_download: bool = False,
) -> list[str]:
    if include_subtitles is None:
        include_subtitles = bool(config.subtitle_languages.strip())
    root = Path(config.download_dir)
    if config.link_type == "playlist":
        output = (
            root
            / "%(playlist_title).120B [%(playlist_id)s]"
            / "%(playlist_index)03d - %(title).150B [%(id)s]"
            / "%(playlist_index)03d - %(title).150B [%(id)s].%(ext)s"
        )
        list_flag = "--yes-playlist"
    else:
        output = root / "%(title).180B [%(id)s]" / "%(title).180B [%(id)s].%(ext)s"
        list_flag = "--no-playlist"

    command = [
        config.yt_dlp_path,
        "--newline",
        "--windows-filenames",
        list_flag,
        "--print",
        f"before_dl:{ID_MARKER}%(id)s",
        "--write-info-json",
        # The generic video template contains playlist_index/title fields.
        # Applying it to the playlist's own metadata creates an invalid nested
        # "...\\000 - playlist\\000 - playlist.info.json" path on Windows.
        "--no-write-playlist-metafiles",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output),
    ]
    if include_subtitles:
        command[7:7] = [
            "--write-auto-subs" if automatic_subtitles else "--write-subs",
            "--sub-langs",
            config.subtitle_languages,
            "--sub-format",
            "srt/vtt/best",
            "--convert-subs",
            "srt",
        ]
    elif config.link_type == "playlist":
        # A private/deleted item should not prevent the rest of a playlist downloading.
        command.append("--ignore-errors")
    command.extend(["--force-overwrites", "--no-continue"])
    if skip_download:
        command.append("--skip-download")
    command.append(url.strip())
    return command


def _matches_id(job: VideoJob, video_id: str) -> bool:
    if job.video_id == video_id:
        return True
    return bool(
        re.search(rf"\[{re.escape(video_id)}\](?:$|\.)", job.video_path.stem)
        or f"[{video_id}]" in job.video_path.name
    )


def _keep_one_source_subtitle(job: VideoJob, runner: ProcessRunner) -> None:
    selected = find_source_subtitle(job.video_path)
    if selected is None:
        return
    generated_suffixes = (
        ".asr.srt",
        ".corrected.srt",
        ".zh-CN.srt",
        ".speech.zh-CN.srt",
    )
    for candidate in job.video_path.parent.glob(job.video_path.stem + "*.srt"):
        if (
            candidate != selected
            and not candidate.name.endswith(generated_suffixes)
        ):
            candidate.unlink(missing_ok=True)
            runner.logger(f"已保留优先字幕并移除重复字幕：{candidate.name}")


def download(config: AppConfig, runner: ProcessRunner, url: str) -> list[VideoJob]:
    if not url.strip():
        raise ValueError("请先填写 YouTube 链接")
    config.ensure_directories()
    try:
        lines = runner.run(build_download_command(config, url))
    except CommandError as exc:
        detail = exc.tail.lower()
        subtitle_failure = (
            bool(config.subtitle_languages.strip())
            and "subtitle" in detail
            and any(
                marker in detail
                for marker in (
                    "unable",
                    "failed",
                    "error",
                    "too many requests",
                )
            )
        )
        if not subtitle_failure:
            raise
        runner.logger(
            "警告：YouTube 字幕请求失败或触发限流。"
            "程序将自动跳过字幕并继续下载视频。"
        )
        lines = runner.run(
            build_download_command(config, url, include_subtitles=False)
        )
    ids = list(
        dict.fromkeys(
            line.split(ID_MARKER, 1)[1].strip()
            for line in lines
            if ID_MARKER in line
        )
    )
    all_jobs = [
        job
        for job in discover_video_jobs(config.download_dir, config.output_dir)
        if job.source_subtitle_path is None
    ]
    if not ids:
        runner.logger("未从下载输出获得视频 ID，将处理下载目录中的视频。")
        jobs = all_jobs
    else:
        jobs = [job for job in all_jobs if any(_matches_id(job, item) for item in ids)]
        if not jobs:
            raise RuntimeError("下载命令已结束，但没有在下载目录中找到视频文件")
    if config.subtitle_languages.strip() and any(
        find_source_subtitle(job.video_path) is None for job in jobs
    ):
        runner.logger("未找到视频自带字幕，正在尝试 YouTube 自动字幕…")
        try:
            runner.run(
                build_download_command(
                    config,
                    url,
                    include_subtitles=True,
                    automatic_subtitles=True,
                    skip_download=True,
                )
            )
        except CommandError:
            runner.logger("警告：自动字幕下载失败或触发限流，已跳过。")
        refreshed = [
            job
            for job in discover_video_jobs(config.download_dir, config.output_dir)
            if job.source_subtitle_path is None
        ]
        by_path = {item.video_path.resolve(): item for item in refreshed}
        jobs = [by_path.get(item.video_path.resolve(), item) for item in jobs]
    for job in jobs:
        _keep_one_source_subtitle(job, runner)
    return jobs
