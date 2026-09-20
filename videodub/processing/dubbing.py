from __future__ import annotations

from pathlib import Path

from ..config import AppConfig
from ..media import VideoJob
from ..runner import ProcessRunner
from ..tts import dub_video


def run_dubbing_stage(
    config: AppConfig,
    runner: ProcessRunner,
    job: VideoJob,
    *,
    base_url: str,
) -> Path:
    """Run stage 4: synthesize speech and mux the dubbed output."""
    return dub_video(config, runner, job, qwen_base_url=base_url)
