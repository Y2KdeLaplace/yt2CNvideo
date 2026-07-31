from __future__ import annotations

import tempfile
from pathlib import Path

from .config import AppConfig
from .media import VideoJob, media_duration
from .qwen_speech import check_qwen_service, synthesize_qwen
from .runner import ProcessRunner
from .subtitles import Cue, read_srt


def _atempo_chain(factor: float) -> str:
    factor = max(0.5, factor)
    filters: list[str] = []
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    filters.append(f"atempo={factor:.6f}")
    return ",".join(filters)


def _synthesize_qwen(
    config: AppConfig,
    runner: ProcessRunner,
    cues: list[Cue],
    raw_dir: Path,
    base_url: str,
) -> list[Path]:
    if config.tts_backend != "gguf":
        info = check_qwen_service(base_url, "tts")
        if not info.available:
            raise RuntimeError(f"Qwen3-TTS 模型服务未就绪：{info.error}")
        runner.logger(f"Qwen3-TTS 模型：{info.model or '未报告'}")
    outputs: list[Path] = []
    for i, cue in enumerate(cues):
        runner.check_cancelled()
        output = raw_dir / f"raw-{i:06d}.wav"
        synthesize_qwen(config, cue.text, output, runner, base_url=base_url)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Qwen3-TTS 未生成有效音频（字幕 {i + 1}）")
        outputs.append(output)
        if (i + 1) % 10 == 0 or i + 1 == len(cues):
            runner.logger(f"Qwen3-TTS 配音进度：{i + 1}/{len(cues)}")
    return outputs


def _prepare_timed_track(
    config: AppConfig,
    runner: ProcessRunner,
    cues: list[Cue],
    raw_files: list[Path],
    work_dir: Path,
    total_duration: float,
) -> Path:
    segment_files: list[Path] = []
    cursor_ms = 0
    for i, (cue, raw_file) in enumerate(zip(cues, raw_files, strict=True)):
        runner.check_cancelled()
        start_ms = max(cursor_ms, cue.start_ms)
        gap_ms = max(0, cue.start_ms - cursor_ms)
        target_ms = max(350, cue.end_ms - cue.start_ms)
        raw_duration_ms = max(1, round(media_duration(config, runner, raw_file) * 1000))
        filters: list[str] = []
        if raw_duration_ms > target_ms:
            filters.append(_atempo_chain(raw_duration_ms / target_ms))
        if gap_ms:
            filters.append(f"adelay={gap_ms}:all=1")
        segment_duration = (gap_ms + target_ms) / 1000
        filters.extend(
            [
                "apad",
                f"atrim=duration={segment_duration:.3f}",
                "aresample=24000",
                "aformat=sample_fmts=s16:channel_layouts=mono",
            ]
        )
        segment = work_dir / f"segment-{i:06d}.wav"
        runner.run(
            [
                config.ffmpeg_path,
                "-y",
                "-i",
                raw_file,
                "-af",
                ",".join(filters),
                "-c:a",
                "pcm_s16le",
                segment,
            ],
            quiet=True,
        )
        segment_files.append(segment)
        cursor_ms = start_ms + target_ms
        if (i + 1) % 25 == 0 or i + 1 == len(cues):
            runner.logger(f"配音对齐进度：{i + 1}/{len(cues)}")

    concat_file = work_dir / "segments.txt"
    concat_file.write_text(
        "\n".join(
            "file '" + str(path).replace("'", "'\\''") + "'" for path in segment_files
        ),
        encoding="utf-8",
    )
    output = work_dir / "chinese-voice.m4a"
    runner.run(
        [
            config.ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-af",
            "apad",
            "-t",
            f"{total_duration:.3f}",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            output,
        ]
    )
    return output


def _output_path(config: AppConfig, job: VideoJob) -> Path:
    target_dir = job.generated_dir or Path(config.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{job.video_path.stem}.中文配音.mp4"


def _mux_video(
    config: AppConfig,
    runner: ProcessRunner,
    job: VideoJob,
    voice_track: Path,
    subtitle_path: Path,
) -> Path:
    output = _output_path(config, job)
    command: list[str | Path] = [
        config.ffmpeg_path,
        "-y",
        "-i",
        job.video_path,
        "-i",
        voice_track,
    ]
    subtitle_input_index: int | None = None
    if config.embed_subtitles and subtitle_path.exists():
        command.extend(["-i", subtitle_path])
        subtitle_input_index = 2

    if config.audio_mode == "mix":
        command.extend(
            [
                "-filter_complex",
                (
                    f"[0:a:0]volume={config.original_volume:.3f}[bg];"
                    "[bg][1:a:0]amix=inputs=2:duration=first:normalize=0[aout]"
                ),
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
            ]
        )
    else:
        command.extend(["-map", "0:v:0", "-map", "1:a:0"])
    if subtitle_input_index is not None:
        command.extend(["-map", f"{subtitle_input_index}:0"])
    command.extend(["-c:v", "copy", "-c:a", "aac", "-map_metadata", "0"])
    if subtitle_input_index is not None:
        command.extend(
            [
                "-c:s",
                "mov_text",
                "-metadata:s:s:0",
                "language=chi",
                "-metadata:s:s:0",
                "title=简体中文",
            ]
        )
    command.extend(["-movflags", "+faststart", output])
    runner.run(command)
    return output


def dub_video(
    config: AppConfig,
    runner: ProcessRunner,
    job: VideoJob,
    *,
    speech_subtitle_path: Path | None = None,
    qwen_base_url: str = "http://127.0.0.1:9955",
) -> Path:
    subtitle_path = job.chinese_subtitle_path
    if not subtitle_path.exists():
        raise RuntimeError(f"缺少中文字幕：{subtitle_path.name}")
    narration_path = speech_subtitle_path or subtitle_path
    cues = read_srt(narration_path)
    if not cues:
        raise RuntimeError(f"中文字幕为空：{subtitle_path}")
    total_duration = media_duration(config, runner, job.video_path)
    runner.logger(f"正在生成 {len(cues)} 段中文语音…")
    with tempfile.TemporaryDirectory(prefix="videodub-tts-") as temp:
        work_dir = Path(temp)
        raw_dir = work_dir / "raw"
        raw_dir.mkdir()
        raw_files = _synthesize_qwen(
            config,
            runner,
            cues,
            raw_dir,
            qwen_base_url,
        )
        voice_track = _prepare_timed_track(
            config,
            runner,
            cues,
            raw_files,
            work_dir,
            total_duration,
        )
        return _mux_video(config, runner, job, voice_track, subtitle_path)
