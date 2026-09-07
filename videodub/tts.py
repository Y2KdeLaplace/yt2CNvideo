from __future__ import annotations

import hashlib
import math
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment

from .config import AppConfig
from .media import VideoJob, media_duration
from .qwen_speech import (
    check_qwen_service,
    synthesize_qwen,
    synthesize_qwen_batch,
)
from .runner import CancelledError, ProcessRunner
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
CROSSFADE_MS = 15
MAX_SENTENCE_DELAY_MS = 250
MAX_SENTENCE_LEAD_MS = 250
MAX_TOTAL_DELAY_MS = 800
GLOBAL_DURATION_RATIO_MIN = 0.80
GLOBAL_DURATION_RATIO_MAX = 1.20
LOCAL_DURATION_RATIO_MIN = 0.90
LOCAL_DURATION_RATIO_MAX = 1.00
TTS_BATCH_SIZE = 2
TTS_CHUNK_MAX_CHARS = 600
TTS_CHUNK_SILENCE_MS = 120
TTS_SYNTHESIS_ATTEMPTS = 2
SENTENCE_END_RE = re.compile(r"[.!?。！？][\"'”’》〉】』〕〗〙〛）)\]]*$")
TTS_SENTENCE_BREAK_RE = re.compile(
    r"[.!?。！？…]+[\"'”’》〉】』〕〗〙〛）)\]]*\s*"
)
TTS_SOFT_BREAK_RE = re.compile(r"[,;:，；：、]+\s*|\s+")


@dataclass(frozen=True)
class SentenceUnit:
    first_cue: int
    last_cue: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class SentenceAudio:
    unit: SentenceUnit
    path: Path
    duration_ms: int
    raw_duration_ms: int
    global_duration_ratio: float = 1.0
    local_duration_ratio: float = 1.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _atempo_chain(speed_factor: float) -> str:
    """Return an atempo chain for a speed factor (output duration is 1/factor)."""
    if not math.isfinite(speed_factor) or speed_factor <= 0:
        raise ValueError("atempo 速度系数必须是有限正数")
    filters: list[str] = []
    while speed_factor > 2.0:
        filters.append("atempo=2.0")
        speed_factor /= 2.0
    while speed_factor < 0.5:
        filters.append("atempo=0.5")
        speed_factor /= 0.5
    filters.append(f"atempo={speed_factor:.6f}")
    return ",".join(filters)


def _tts_text_chunks(text: str) -> list[str]:
    if len(text) <= TTS_CHUNK_MAX_CHARS:
        return [text]
    sentence_breaks = [match.end() for match in TTS_SENTENCE_BREAK_RE.finditer(text)]
    soft_breaks = [match.end() for match in TTS_SOFT_BREAK_RE.finditer(text)]
    chunks: list[str] = []
    start = 0
    while len(text) - start > TTS_CHUNK_MAX_CHARS:
        limit = start + TTS_CHUNK_MAX_CHARS
        boundary = max(
            (position for position in sentence_breaks if start < position <= limit),
            default=0,
        )
        if not boundary:
            boundary = max(
                (position for position in soft_breaks if start < position <= limit),
                default=limit,
            )
        chunks.append(text[start:boundary])
        start = boundary
    if start < len(text):
        chunks.append(text[start:])
    return chunks


def _join_tts_audio(parts: list[Path], output: Path) -> None:
    combined: AudioSegment | None = None
    silence = (
        AudioSegment.silent(duration=TTS_CHUNK_SILENCE_MS, frame_rate=24000)
        .set_channels(1)
        .set_sample_width(2)
    )
    for path in parts:
        with path.open("rb") as source:
            audio = (
                AudioSegment.from_wav(source)
                .set_frame_rate(24000)
                .set_channels(1)
                .set_sample_width(2)
            )
        if combined is None:
            combined = audio
            continue
        crossfade = min(CROSSFADE_MS, len(combined), len(silence))
        combined = combined.append(silence, crossfade=crossfade)
        crossfade = min(CROSSFADE_MS, len(combined), len(audio))
        combined = combined.append(audio, crossfade=crossfade)
    if combined is None:
        raise RuntimeError("Qwen3-TTS 没有生成可拼接的音频")
    with output.open("wb") as destination:
        combined.export(destination, format="wav")


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


def _sentence_units(cues: list[Cue]) -> list[SentenceUnit]:
    units: list[SentenceUnit] = []
    pending: list[Cue] = []
    for cue in cues:
        pending.append(cue)
        text = _joined_speech_text(pending)
        if SENTENCE_END_RE.search(text):
            units.append(
                SentenceUnit(
                    pending[0].index,
                    pending[-1].index,
                    pending[0].start_ms,
                    pending[-1].end_ms,
                    text,
                )
            )
            pending = []
    if pending:
        units.append(
            SentenceUnit(
                pending[0].index,
                pending[-1].index,
                pending[0].start_ms,
                pending[-1].end_ms,
                _joined_speech_text(pending),
            )
        )
    return units


def _audio_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as source:
            return max(1, round(source.getnframes() / source.getframerate() * 1000))
    except (EOFError, wave.Error, ZeroDivisionError) as exc:
        raise ValueError(f"无法读取 WAV 时长：{path}") from exc


def _run_tts_request(
    config: AppConfig,
    runner: ProcessRunner,
    texts: list[str],
    outputs: list[Path],
    base_url: str,
) -> None:
    for attempt in range(1, TTS_SYNTHESIS_ATTEMPTS + 1):
        runner.check_cancelled()
        for output in outputs:
            output.unlink(missing_ok=True)
        try:
            if len(texts) == 1:
                synthesize_qwen(
                    config,
                    texts[0],
                    outputs[0],
                    runner,
                    base_url=base_url,
                )
            else:
                synthesize_qwen_batch(
                    config,
                    texts,
                    outputs,
                    runner,
                    base_url=base_url,
                )
            if any(
                not path.is_file() or path.stat().st_size == 0
                for path in outputs
            ):
                raise RuntimeError("Qwen3-TTS 没有返回完整音频")
            return
        except CancelledError:
            raise
        except (OSError, RuntimeError) as exc:
            if attempt == TTS_SYNTHESIS_ATTEMPTS:
                raise RuntimeError(
                    f"Qwen3-TTS 连续 {TTS_SYNTHESIS_ATTEMPTS} 次生成失败：{exc}"
                ) from exc
            runner.logger(f"Qwen3-TTS 生成失败，正在自动重试：{exc}")


def _synthesize_sentence(
    config: AppConfig,
    runner: ProcessRunner,
    unit: SentenceUnit,
    index: int,
    work_dir: Path,
    base_url: str,
) -> SentenceAudio:
    output = work_dir / f"sentence-{index:05d}.raw.wav"
    chunks = _tts_text_chunks(unit.text)
    if len(chunks) == 1:
        _run_tts_request(config, runner, chunks, [output], base_url)
    else:
        part_paths = [
            work_dir / f"sentence-{index:05d}-part-{part:03d}.raw.wav"
            for part in range(len(chunks))
        ]
        for first in range(0, len(chunks), TTS_BATCH_SIZE):
            last = min(first + TTS_BATCH_SIZE, len(chunks))
            _run_tts_request(
                config,
                runner,
                chunks[first:last],
                part_paths[first:last],
                base_url,
            )
        _join_tts_audio(part_paths, output)
    duration_ms = _audio_duration_ms(output)
    return SentenceAudio(unit, output, duration_ms, duration_ms)


def _synthesize_sentence_units(
    config: AppConfig,
    runner: ProcessRunner,
    units: list[SentenceUnit],
    work_dir: Path,
    base_url: str,
) -> list[SentenceAudio]:
    sentences: list[SentenceAudio] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if config.tts_backend == "gguf" or len(_tts_text_chunks(unit.text)) > 1:
            sentences.append(
                _synthesize_sentence(config, runner, unit, index, work_dir, base_url)
            )
            index += 1
        else:
            last = index
            while (
                last < len(units)
                and last - index < TTS_BATCH_SIZE
                and len(_tts_text_chunks(units[last].text)) == 1
            ):
                last += 1
            batch_units = units[index:last]
            paths = [
                work_dir / f"sentence-{i:05d}.raw.wav"
                for i in range(index, last)
            ]
            _run_tts_request(
                config,
                runner,
                [item.text for item in batch_units],
                paths,
                base_url,
            )
            for item, path in zip(batch_units, paths, strict=True):
                duration_ms = _audio_duration_ms(path)
                sentences.append(SentenceAudio(item, path, duration_ms, duration_ms))
            index = last
        if index % 10 == 0 or index == len(units):
            runner.logger(f"自然句 TTS 进度：{index}/{len(units)}")
    return sentences


def _global_duration_factor(sentences: list[SentenceAudio]) -> float:
    """Return the clamped output/input duration ratio shared by all sentences."""
    valid = [item for item in sentences if item.raw_duration_ms > 0]
    if not valid:
        return 1.0
    raw_total = sum(item.raw_duration_ms for item in valid)
    target_total = sum(
        max(1, item.unit.end_ms - item.unit.start_ms) for item in valid
    )
    return _clamp(
        target_total / raw_total,
        GLOBAL_DURATION_RATIO_MIN,
        GLOBAL_DURATION_RATIO_MAX,
    )


def _local_duration_factor(
    raw_duration_ms: int,
    target_duration_ms: int,
    global_duration_ratio: float,
) -> float:
    """Return a per-sentence output/input ratio that only compresses further."""
    globally_adjusted_ms = max(1.0, raw_duration_ms * global_duration_ratio)
    needed_ratio = target_duration_ms / globally_adjusted_ms
    return _clamp(
        needed_ratio,
        LOCAL_DURATION_RATIO_MIN,
        LOCAL_DURATION_RATIO_MAX,
    )


def _adjust_sentence_audio(
    config: AppConfig,
    runner: ProcessRunner,
    sentence: SentenceAudio,
    index: int,
    global_duration_ratio: float,
    work_dir: Path,
) -> SentenceAudio:
    target_ms = max(1, sentence.unit.end_ms - sentence.unit.start_ms)
    local_duration_ratio = _local_duration_factor(
        sentence.raw_duration_ms,
        target_ms,
        global_duration_ratio,
    )
    duration_ratio = global_duration_ratio * local_duration_ratio
    speed_factor = 1.0 / duration_ratio
    output = work_dir / f"sentence-{index:05d}.fit.wav"
    runner.run(
        [
            config.ffmpeg_path,
            "-y",
            "-i",
            sentence.path,
            "-af",
            (
                f"{_atempo_chain(speed_factor)},aresample=24000,"
                "aformat=sample_fmts=s16:channel_layouts=mono"
            ),
            "-c:a",
            "pcm_s16le",
            output,
        ],
        quiet=True,
    )
    duration_ms = _audio_duration_ms(output)
    return SentenceAudio(
        sentence.unit,
        output,
        duration_ms,
        sentence.raw_duration_ms,
        global_duration_ratio,
        local_duration_ratio,
    )


def _fit_sentence_durations(
    config: AppConfig,
    runner: ProcessRunner,
    sentences: list[SentenceAudio],
    work_dir: Path,
) -> list[SentenceAudio]:
    global_duration_ratio = _global_duration_factor(sentences)
    raw_total = sum(item.raw_duration_ms for item in sentences)
    target_total = sum(
        max(1, item.unit.end_ms - item.unit.start_ms) for item in sentences
    )
    runner.logger(
        f"配音全局 duration ratio：{global_duration_ratio:.3f} "
        f"（原始 {raw_total}ms，目标 {target_total}ms）"
    )
    adjusted: list[SentenceAudio] = []
    for index, sentence in enumerate(sentences):
        fitted = _adjust_sentence_audio(
            config,
            runner,
            sentence,
            index,
            global_duration_ratio,
            work_dir,
        )
        adjusted.append(fitted)
        target_ms = max(1, fitted.unit.end_ms - fitted.unit.start_ms)
        unusual = (
            fitted.local_duration_ratio < LOCAL_DURATION_RATIO_MAX
            or fitted.duration_ms > target_ms
        )
        if unusual or (index + 1) % 10 == 0 or index + 1 == len(sentences):
            runner.logger(
                f"句子 {index + 1}/{len(sentences)}：目标时长 {target_ms}ms，"
                f"原始 TTS {fitted.raw_duration_ms}ms，"
                f"global factor {fitted.global_duration_ratio:.3f}，"
                f"local factor {fitted.local_duration_ratio:.3f}，"
                f"最终时长 {fitted.duration_ms}ms"
            )
    return adjusted


def _scheduled_sentence_start(
    unit: SentenceUnit,
    next_start_ms: int,
    audio_duration_ms: int,
    cursor_end_ms: int,
    total_duration_ms: int,
    *,
    is_last: bool,
) -> int:
    latest_safe_end = (
        total_duration_ms if is_last else next_start_ms + MAX_SENTENCE_DELAY_MS
    )
    desired_start = min(unit.start_ms, latest_safe_end - audio_duration_ms)
    desired_start = max(0, unit.start_ms - MAX_SENTENCE_LEAD_MS, desired_start)
    return max(desired_start, cursor_end_ms - CROSSFADE_MS)


def _render_sentence_track(
    runner: ProcessRunner,
    sentences: list[SentenceAudio],
    work_dir: Path,
    total_duration_ms: int,
) -> Path:
    rendered: list[tuple[AudioSegment, int]] = []
    cursor_end = 0
    total_delay = 0
    previous_delay = 0
    for index, sentence in enumerate(sentences):
        runner.check_cancelled()
        with sentence.path.open("rb") as source:
            audio = AudioSegment.from_wav(source)
        fade_ms = min(CROSSFADE_MS, len(audio) // 2)
        if fade_ms:
            audio = audio.fade_in(fade_ms).fade_out(fade_ms)
        next_start = (
            sentences[index + 1].unit.start_ms
            if index + 1 < len(sentences)
            else total_duration_ms
        )
        start_ms = _scheduled_sentence_start(
            sentence.unit,
            next_start,
            len(audio),
            cursor_end,
            total_duration_ms,
            is_last=index + 1 == len(sentences),
        )
        delay_ms = max(0, start_ms - sentence.unit.start_ms)
        total_delay += max(0, delay_ms - previous_delay)
        previous_delay = delay_ms
        if delay_ms > MAX_SENTENCE_DELAY_MS or total_delay > MAX_TOTAL_DELAY_MS:
            runner.logger(
                f"字幕 {sentence.unit.first_cue}–{sentence.unit.last_cue} 因串行调度产生 "
                f"{delay_ms}ms 延迟（累计 {total_delay}ms）。"
            )
        end_ms = start_ms + len(audio)
        if index + 1 == len(sentences) and end_ms > total_duration_ms:
            runner.logger(
                f"最后一句超出视频结尾 {end_ms - total_duration_ms}ms，"
                "已仅在最终视频边界截断。"
            )
            audio = audio[: max(0, total_duration_ms - start_ms)]
            end_ms = start_ms + len(audio)
        if audio and start_ms < total_duration_ms:
            rendered.append((audio, start_ms))
        cursor_end = max(cursor_end, end_ms)

    canvas = (
        AudioSegment.silent(duration=total_duration_ms, frame_rate=24000)
        .set_channels(1)
        .set_sample_width(2)
    )
    for audio, start_ms in rendered:
        canvas = canvas.overlay(audio, position=start_ms)
    output = work_dir / "aligned_audio.wav"
    with output.open("wb") as destination:
        canvas.export(destination, format="wav")
    return output


def _job_temp_dir(config: AppConfig, job: VideoJob) -> Path:
    identity = str(job.video_path.resolve()).encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:10]
    safe_stem = re.sub(r"[^\w.-]+", "_", job.video_path.stem)[:80] or "video"
    path = Path(config.cache_dir) / "tmp" / f"{safe_stem}-{suffix}"
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.mkdir(parents=True)
    return path


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
    qwen_base_url: str = "http://127.0.0.1:9955",
) -> Path:
    subtitle_path = job.translated_subtitle_path(config.translation_language)
    if not subtitle_path.exists():
        raise RuntimeError(f"缺少翻译字幕：{subtitle_path.name}")
    cues = read_srt(subtitle_path)
    if not cues:
        raise RuntimeError(f"翻译字幕为空：{subtitle_path}")
    for previous, current in zip(cues, cues[1:]):
        if current.start_ms < previous.end_ms:
            runner.logger(
                f"字幕 {previous.index} 与 {current.index} 时间轴重叠 "
                f"{previous.end_ms - current.start_ms}ms，"
                "配音阶段将按自然句顺序串行调度。"
            )
    units = _sentence_units(cues)
    if not units:
        raise RuntimeError(f"翻译字幕没有可配音的自然句：{subtitle_path}")
    total_duration = (
        media_duration(config, runner, job.video_path)
        if job.has_video
        else max(cue.end_ms for cue in cues) / 1000
    )
    work_dir = _job_temp_dir(config, job)
    if config.tts_backend != "gguf":
        info = check_qwen_service(qwen_base_url, "tts")
        if not info.available:
            raise RuntimeError(f"Qwen3-TTS 模型服务未就绪：{info.error}")
        runner.logger(f"Qwen3-TTS 模型：{info.model or '未报告'}")
    raw_sentences = _synthesize_sentence_units(
        config,
        runner,
        units,
        work_dir,
        qwen_base_url,
    )
    adjusted_sentences = _fit_sentence_durations(
        config,
        runner,
        raw_sentences,
        work_dir,
    )
    voice_track = _render_sentence_track(
        runner,
        adjusted_sentences,
        work_dir,
        round(total_duration * 1000),
    )
    if job.has_video:
        return _mux_video(config, runner, job, voice_track, subtitle_path)
    output = _audio_output_path(config, job)
    runner.run(
        [
            config.ffmpeg_path,
            "-y",
            "-i",
            voice_track,
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            output,
        ]
    )
    return output
