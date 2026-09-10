from __future__ import annotations

import hashlib
import math
import json
import tempfile
import re
import shutil
import wave
from dataclasses import dataclass, replace, asdict
from pathlib import Path

from pydub import AudioSegment

from .config import AppConfig
from .media import VideoJob, media_duration
from .qwen_speech import (
    check_qwen_service,
    resolve_tts_reference,
    synthesize_qwen,
    synthesize_qwen_batch,
)
from .runner import CancelledError, ProcessRunner
from .sentences import SentenceUnit, read_units, spoken_text, validate_units


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
DEFAULT_TTS_BATCH_SIZE = 2
MLX_TTS_BATCH_SIZE = 1
TTS_CHUNK_MAX_CHARS = 600
TTS_CHUNK_SILENCE_MS = 120
TTS_SYNTHESIS_ATTEMPTS = 2
TTS_SENTENCE_BREAK_RE = re.compile(
    r"[.!?。！？…]+[\"'”’》〉】』〕〗〙〛）)\]]*\s*"
)
TTS_SOFT_BREAK_RE = re.compile(r"[,;:，；：、]+\s*|\s+")


@dataclass(frozen=True)
class SentenceAudio:
    unit: SentenceUnit
    path: Path
    duration_ms: int
    raw_duration_ms: int
    global_duration_ratio: float = 1.0
    local_duration_ratio: float = 1.0


def _tts_batch_size(config: AppConfig) -> int:
    return (
        MLX_TTS_BATCH_SIZE
        if config.tts_backend == "mlx"
        else DEFAULT_TTS_BATCH_SIZE
    )


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


def _audio_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as source:
            if source.getnframes() <= 0 or source.getframerate() <= 0:
                raise ValueError(f"WAV 没有音频帧：{path}")
            frames = source.readframes(source.getnframes())
            if len(frames) != source.getnframes() * source.getnchannels() * source.getsampwidth():
                raise ValueError(f"WAV 数据不完整：{path}")
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
    pending = list(zip(texts, outputs, strict=True))
    for attempt in range(1, TTS_SYNTHESIS_ATTEMPTS + 1):
        runner.check_cancelled()
        error = ""
        try:
            requested_texts, requested_paths = map(list, zip(*pending))
            for path in requested_paths:
                path.unlink(missing_ok=True)
            if len(pending) == 1:
                synthesize_qwen(config, requested_texts[0], requested_paths[0], runner, base_url=base_url)
            else:
                synthesize_qwen_batch(config, requested_texts, requested_paths, runner, base_url=base_url)
        except CancelledError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            error = str(exc)
        failed = []
        for text, path in pending:
            try:
                _audio_duration_ms(path)
            except (OSError, ValueError):
                path.unlink(missing_ok=True)
                failed.append((text, path))
        if not failed:
            return
        pending = failed
        detail = "; ".join(f'{p.name} text={t!r}' for t, p in failed)
        if attempt == TTS_SYNTHESIS_ATTEMPTS:
            raise RuntimeError(f"Qwen3-TTS 连续 2 次生成失败：{detail}；{error or '没有有效音频'}")
        runner.logger(f"Qwen3-TTS 生成失败，正在自动重试：{detail}；{error}")


def _synthesize_sentence(
    config: AppConfig,
    runner: ProcessRunner,
    unit: SentenceUnit,
    index: int,
    work_dir: Path,
    base_url: str,
) -> SentenceAudio:
    output = work_dir / f"sentence-{index:05d}.raw.wav"
    text = spoken_text(unit.text)
    if not text:
        raise ValueError("TTS 输入为空、纯标点或纯声效")
    chunks = _tts_text_chunks(text)
    if len(chunks) == 1:
        _run_tts_request(config, runner, chunks, [output], base_url)
    else:
        part_paths = [
            work_dir / f"sentence-{index:05d}-part-{part:03d}.raw.wav"
            for part in range(len(chunks))
        ]
        batch_size = _tts_batch_size(config)
        for first in range(0, len(chunks), batch_size):
            last = min(first + batch_size, len(chunks))
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


def _voice_cache_identity(config: AppConfig) -> dict:
    settings = {k: v for k, v in asdict(config).items() if k.startswith("tts_")}
    # Resolve preset/custom reference exactly as model startup does.
    if config.tts_voice_preset or config.tts_use_custom_voice:
        audio, text = resolve_tts_reference(config)
        settings["resolved_reference_text"] = text
        settings["reference_sha256"] = hashlib.sha256(Path(audio).read_bytes()).hexdigest()
    elif config.tts_reference_audio:
        settings["reference_sha256"] = hashlib.sha256(Path(config.tts_reference_audio).read_bytes()).hexdigest()
    if config.tts_reference_text_file:
        settings["reference_text_sha256"] = hashlib.sha256(Path(config.tts_reference_text_file).read_bytes()).hexdigest()
    model_path = Path(config.tts_model_path) if config.tts_model_path else None
    if model_path is not None and model_path.exists():
        # Weight replacement at the same local model id invalidates old speech.
        files = [model_path] if model_path.is_file() else sorted(
            p for p in model_path.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".gguf", ".bin"})
        settings["model_files"] = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return {"generation_version": 2, "settings": settings,
            "mlx_sampling": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096},
            "chunk_chars": TTS_CHUNK_MAX_CHARS, "chunk_silence_ms": TTS_CHUNK_SILENCE_MS}


def _cache_key(identity: dict, text: str) -> str:
    return hashlib.sha256(json.dumps({**identity, "text": text}, ensure_ascii=False,
                                     sort_keys=True).encode("utf-8")).hexdigest()


def _save_cached_audio(source: Path, target: Path) -> None:
    _audio_duration_ms(source)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".wav", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _synthesize_sentence_units(
    config: AppConfig, runner: ProcessRunner, units: list[SentenceUnit],
    work_dir: Path, base_url: str,
) -> list[SentenceAudio]:
    normalized = [replace(u, text=spoken_text(u.text)) for u in units if u.kind == "spoken"]
    if any(not u.text for u in normalized):
        raise ValueError("TTS 输入为空、纯标点或纯声效")
    units = normalized
    cache = Path(config.cache_dir) / "tts-sentences-v2"
    cache.mkdir(parents=True, exist_ok=True)
    identity = _voice_cache_identity(config)
    paths = [work_dir / f"sentence-{i:05d}.raw.wav" for i in range(len(units))]
    cached = [cache / (_cache_key(identity, u.text) + ".wav") for u in units]
    results: dict[int, SentenceAudio] = {}
    for i, unit in enumerate(units):
        runner.check_cancelled()
        try:
            duration = _audio_duration_ms(cached[i])
        except (OSError, ValueError):
            continue
        shutil.copyfile(cached[i], paths[i])
        results[i] = SentenceAudio(unit, paths[i], duration, duration)
    runner.logger(f"自然句 TTS：{len(units)} 句，复用缓存 {len(results)} 句")
    missing = [i for i in range(len(units)) if i not in results]
    batch_size = _tts_batch_size(config)
    cursor = 0
    while cursor < len(missing):
        i = missing[cursor]
        batch = [i]
        if config.tts_backend != "gguf" and len(_tts_text_chunks(units[i].text)) == 1:
            for following in missing[cursor + 1:cursor + batch_size]:
                if len(_tts_text_chunks(units[following].text)) > 1:
                    break
                batch.append(following)
        for j in batch:
            paths[j].unlink(missing_ok=True)
        try:
            if len(batch) == 1:
                _synthesize_sentence(config, runner, units[i], i, work_dir, base_url)
            else:
                _run_tts_request(config, runner, [units[j].text for j in batch],
                                 [paths[j] for j in batch], base_url)
        except CancelledError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            failed = []
            for j in batch:
                try:
                    _audio_duration_ms(paths[j])
                except (OSError, ValueError):
                    u = units[j]
                    failed.append(f'自然句 TTS {j + 1}/{len(units)} cue: {u.first_cue}–{u.last_cue} text={u.text!r} target={u.end_ms-u.start_ms}ms')
            detail = "\n".join(failed)
            raise RuntimeError(f"{detail}\nmodel={config.tts_model_id or config.tts_model_path} backend={config.tts_backend}\n{exc}") from exc
        finally:
            # Commit each successful sentence even if its batch peer failed.
            for j in batch:
                try:
                    duration = _audio_duration_ms(paths[j])
                except (OSError, ValueError):
                    continue
                _save_cached_audio(paths[j], cached[j])
                results[j] = SentenceAudio(units[j], paths[j], duration, duration)
        cursor += len(batch)
        if len(results) % 10 == 0 or len(results) == len(units):
            runner.logger(f"自然句 TTS 进度：{len(results)}/{len(units)}")
    return [results[i] for i in range(len(units))]


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
    # Preserve natural speed for already-short sentences; leave room as silence.
    global_duration_ratio = min(1.0, global_duration_ratio)
    if sentence.raw_duration_ms <= target_ms:
        global_duration_ratio = 1.0
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
                f"自然句 TTS {index + 1}/{len(sentences)} cue: {fitted.unit.first_cue}–{fitted.unit.last_cue} "
                f"text={fitted.unit.text!r} target: {target_ms}ms，"
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
    units = read_units(subtitle_path)
    validate_units(units)
    subtitle_end_ms = max((u.end_ms for u in units), default=0)
    units = [u for u in units if u.kind == "spoken"]
    if not units:
        raise RuntimeError(f"翻译句级数据没有可配音的自然句：{subtitle_path}")
    total_duration = (
        media_duration(config, runner, job.video_path)
        if job.has_video
        else subtitle_end_ms / 1000
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
