from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .runner import ProcessRunner


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
GENERATED_SUFFIXES = (".zh-CN.srt", ".corrected.srt")


@dataclass(frozen=True)
class VideoJob:
    video_path: Path
    info_path: Path | None = None
    video_id: str = ""
    title: str = ""

    @property
    def base_path(self) -> Path:
        return self.video_path.with_suffix("")

    @property
    def chinese_subtitle_path(self) -> Path:
        return self.base_path.with_name(self.base_path.name + ".zh-CN.srt")

    @property
    def corrected_subtitle_path(self) -> Path:
        return self.base_path.with_name(self.base_path.name + ".corrected.srt")


def _info_for_video(video: Path) -> Path | None:
    direct = video.with_suffix(".info.json")
    if direct.exists():
        return direct
    candidates = sorted(video.parent.glob("*.info.json"))
    id_match = re.search(r"\[([A-Za-z0-9_-]{6,})\]$", video.stem)
    if id_match:
        for candidate in candidates:
            if f"[{id_match.group(1)}]" in candidate.stem:
                return candidate
    return candidates[0] if len(candidates) == 1 else None


def discover_video_jobs(root: str | Path) -> list[VideoJob]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    jobs: list[VideoJob] = []
    for video in sorted(root_path.rglob("*")):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if video.name.endswith(".dubbed.mp4"):
            continue
        info_path = _info_for_video(video)
        video_id = ""
        title = video.stem
        if info_path:
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                video_id = str(info.get("id") or "")
                title = str(info.get("title") or title)
            except (OSError, ValueError):
                pass
        jobs.append(VideoJob(video, info_path, video_id, title))
    return jobs


def media_duration(config: AppConfig, runner: ProcessRunner, path: Path) -> float:
    lines = runner.run(
        [
            config.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        quiet=True,
    )
    try:
        return float(lines[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"无法读取媒体时长：{path}") from exc
