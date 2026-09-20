from __future__ import annotations

from ..config import AppConfig
from ..media import VideoJob
from ..qwen_speech import extract_asr_subtitle
from ..runner import ProcessRunner


def run_extract_stage(
    config: AppConfig,
    runner: ProcessRunner,
    job: VideoJob,
    *,
    base_url: str,
) -> None:
    """Run stage 1: extract a timestamped ASR subtitle from the video."""
    if not job.has_video:
        runner.logger("未找到视频，已跳过语音提取。")
        return
    extract_asr_subtitle(
        config,
        runner,
        job,
        language=config.asr_language,
        base_url=base_url,
    )
