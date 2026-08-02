from __future__ import annotations

import shutil
import tempfile
import wave
from pathlib import Path

from .config import AppConfig
from .media import VideoJob, media_duration
from .qwen_speech import check_qwen_service, synthesize_qwen
from .runner import ProcessRunner
from .subtitles import Cue, read_srt


LANGUAGE_METADATA_CODES = {
    "Chinese": "chi",
    "English": "eng",
    "Japanese": "jpn",
    "Korean": "kor",
    "German": "deu",
    "Spanish": "spa",
    "French": "fra",
    "Italian": "ita",
    "Portuguese": "por",
    "Russian": "rus",
}


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
) -> Path:
    if config.tts_backend != "gguf":
        info = check_qwen_service(base_url, "tts")
        if not info.available:
            raise RuntimeError(f"Qwen3-TTS 模型服务未就绪：{info.error}")
        runner.logger(f"Qwen3-TTS 模型：{info.model or '未报告'}")
    runner.check_cancelled()
    text = _joined_speech_text(cues)
    output = raw_dir / "raw-full.wav"
    synthesize_qwen(config, text, output, runner, base_url=base_url)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Qwen3-TTS 未生成有效的完整配音音频")
    runner.logger(f"Qwen3-TTS 已一次生成完整配音（{len(cues)} 条字幕）")
    return output


def _joined_speech_text(cues: list[Cue]) -> str:
    parts: list[str] = []
    for cue in cues:
        text = cue.text.strip()
        if not text:
            continue
        if (
            parts
            and parts[-1][-1].isascii()
            and parts[-1][-1].isalnum()
            and text[0].isascii()
            and text[0].isalnum()
        ):
            parts.append(" ")
        parts.append(text)
    return "".join(parts)


def _text_weight(text: str) -> float:
    weight = 0.0
    for character in text:
        if character.isspace():
            continue
        if character in "。！？.!?":
            weight += 2.0
        elif character in "，、；：,;:":
            weight += 0.75
        else:
            weight += 1.0
    return max(1.0, weight)


def _source_ranges(
    cues: list[Cue],
    total_ms: int,
    silence_midpoints: list[int],
) -> list[tuple[int, int]]:
    if not cues:
        return []
    weights = [_text_weight(cue.text) for cue in cues]
    total_weight = sum(weights)
    expected: list[int] = []
    cumulative = 0.0
    for weight in weights[:-1]:
        cumulative += weight
        expected.append(round(total_ms * cumulative / total_weight))

    cuts = [0]
    for index, target in enumerate(expected):
        previous_target = expected[index - 1] if index else 0
        next_target = expected[index + 1] if index + 1 < len(expected) else total_ms
        window_start = (previous_target + target) // 2
        window_end = (target + next_target) // 2
        lower = max(cuts[-1] + 1, window_start)
        upper = min(total_ms - (len(expected) - index), window_end)
        nearby = [
            point
            for point in silence_midpoints
            if lower <= point <= upper
        ]
        cut = min(nearby, key=lambda point: abs(point - target)) if nearby else target
        cut = max(cuts[-1] + 1, min(cut, total_ms - (len(expected) - index)))
        cuts.append(cut)
    cuts.append(total_ms)
    return list(zip(cuts, cuts[1:]))


def _silence_midpoints(
    config: AppConfig,
    runner: ProcessRunner,
    path: Path,
    total_ms: int,
) -> list[int]:
    lines = runner.run(
        [
            config.ffmpeg_path,
            "-hide_banner",
            "-i",
            path,
            "-af",
            "silencedetect=noise=-35dB:d=0.06",
            "-f",
            "null",
            "-",
        ],
        quiet=True,
    )
    starts: list[float] = []
    midpoints: list[int] = []
    for line in lines:
        if "silence_start:" in line:
            try:
                starts.append(float(line.split("silence_start:", 1)[1].split()[0]))
            except ValueError:
                continue
        elif "silence_end:" in line and starts:
            try:
                end = float(line.split("silence_end:", 1)[1].split()[0])
            except ValueError:
                continue
            start = starts.pop(0)
            midpoint = round((start + end) * 500)
            if 0 < midpoint < total_ms:
                midpoints.append(midpoint)
    return midpoints


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as source:
            return max(1, round(source.getnframes() / source.getframerate() * 1000))
    except (EOFError, wave.Error, ZeroDivisionError) as exc:
        raise ValueError(f"无法读取 WAV 时长：{path}") from exc


def _prepare_timed_track(
    config: AppConfig,
    runner: ProcessRunner,
    cues: list[Cue],
    raw_file: Path,
    work_dir: Path,
    total_duration: float,
) -> Path:
    try:
        raw_duration_ms = _wav_duration_ms(raw_file)
    except ValueError:
        raw_duration_ms = max(
            1,
            round(media_duration(config, runner, raw_file) * 1000),
        )
    source_ranges = _source_ranges(
        cues,
        raw_duration_ms,
        _silence_midpoints(config, runner, raw_file, raw_duration_ms),
    )
    segment_files: list[Path] = []
    cursor_ms = 0
    for i, (cue, source_range) in enumerate(
        zip(cues, source_ranges, strict=True)
    ):
        runner.check_cancelled()
        start_ms = max(cursor_ms, cue.start_ms)
        gap_ms = max(0, cue.start_ms - cursor_ms)
        target_ms = max(350, cue.end_ms - cue.start_ms)
        source_start_ms, source_end_ms = source_range
        source_duration_ms = max(1, source_end_ms - source_start_ms)
        filters: list[str] = []
        if source_duration_ms > target_ms:
            filters.append(_atempo_chain(source_duration_ms / target_ms))
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
                "-ss",
                f"{source_start_ms / 1000:.3f}",
                "-t",
                f"{source_duration_ms / 1000:.3f}",
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
    return target_dir / f"{job.video_path.stem}.{config.tts_language}配音.mp4"


def _audio_output_path(config: AppConfig, job: VideoJob) -> Path:
    target_dir = job.generated_dir or Path(config.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{job.video_path.stem}.{config.tts_language}配音.m4a"


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
                f"language={LANGUAGE_METADATA_CODES.get(config.tts_language, 'und')}",
                "-metadata:s:s:0",
                f"title={config.translation_language}",
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
    subtitle_path = job.translated_subtitle_path(config.translation_language)
    if not subtitle_path.exists():
        raise RuntimeError(f"缺少翻译字幕：{subtitle_path.name}")
    narration_path = speech_subtitle_path or subtitle_path
    cues = read_srt(narration_path)
    if not cues:
        raise RuntimeError(f"翻译字幕为空：{subtitle_path}")
    total_duration = (
        media_duration(config, runner, job.video_path)
        if job.has_video
        else max(cue.end_ms for cue in cues) / 1000
    )
    runner.logger(f"正在生成 {len(cues)} 段 {config.tts_language} 语音…")
    with tempfile.TemporaryDirectory(prefix="videodub-tts-") as temp:
        work_dir = Path(temp)
        raw_dir = work_dir / "raw"
        raw_dir.mkdir()
        raw_file = _synthesize_qwen(
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
            raw_file,
            work_dir,
            total_duration,
        )
        if job.has_video:
            return _mux_video(config, runner, job, voice_track, subtitle_path)
        output = _audio_output_path(config, job)
        shutil.copyfile(voice_track, output)
        return output
