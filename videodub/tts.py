from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import unicodedata
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from pydub import AudioSegment

from .config import AppConfig
from .media import VideoJob, media_duration
from .qwen_speech import (
    align_qwen,
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
ALIGNMENT_MIN_MS = 240_000
ALIGNMENT_MAX_MS = 294_000
ALIGNMENT_HARD_MAX_MS = 300_000
CROSSFADE_MS = 15
MAX_SENTENCE_DELAY_MS = 250
MAX_SENTENCE_LEAD_MS = 250
MAX_TOTAL_DELAY_MS = 800
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
class AlignedToken:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SentenceUnit:
    first_cue: int
    last_cue: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class AlignedSentence:
    unit: SentenceUnit
    source_start_ms: int
    source_end_ms: int


class _TTSAlignmentQualityError(RuntimeError):
    pass


def _atempo_chain(factor: float) -> str:
    factor = max(0.5, factor)
    filters: list[str] = []
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    filters.append(f"atempo={factor:.6f}")
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


def _join_tts_audio(parts: list[Path], output: Path) -> list[tuple[int, int]]:
    combined: AudioSegment | None = None
    ranges: list[tuple[int, int]] = []
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
            ranges.append((0, len(audio)))
            continue
        crossfade = min(CROSSFADE_MS, len(combined), len(silence))
        combined = combined.append(silence, crossfade=crossfade)
        crossfade = min(CROSSFADE_MS, len(combined), len(audio))
        start_ms = len(combined) - crossfade
        combined = combined.append(audio, crossfade=crossfade)
        ranges.append((start_ms, start_ms + len(audio)))
    if combined is None:
        raise RuntimeError("Qwen3-TTS 没有生成可拼接的音频")
    with output.open("wb") as destination:
        combined.export(destination, format="wav")
    return ranges


def _synthesize_qwen(
    config: AppConfig,
    runner: ProcessRunner,
    text: str,
    work_dir: Path,
    base_url: str,
) -> Path:
    if config.tts_backend != "gguf":
        info = check_qwen_service(base_url, "tts")
        if not info.available:
            raise RuntimeError(f"Qwen3-TTS 模型服务未就绪：{info.error}")
        runner.logger(f"Qwen3-TTS 模型：{info.model or '未报告'}")
    runner.check_cancelled()
    output = work_dir / "full_audio.wav"
    chunks = _tts_text_chunks(text)
    if len(chunks) == 1:
        for attempt in range(1, TTS_SYNTHESIS_ATTEMPTS + 1):
            runner.check_cancelled()
            output.unlink(missing_ok=True)
            try:
                synthesize_qwen(config, text, output, runner, base_url=base_url)
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError("Qwen3-TTS 没有返回有效音频")
                break
            except CancelledError:
                raise
            except (OSError, RuntimeError) as exc:
                if attempt == TTS_SYNTHESIS_ATTEMPTS:
                    raise RuntimeError(
                        f"Qwen3-TTS 连续 {TTS_SYNTHESIS_ATTEMPTS} 次生成失败：{exc}"
                    ) from exc
                runner.logger(f"Qwen3-TTS 生成失败，正在自动重试：{exc}")
    else:
        runner.logger(
            f"完整文稿超过单次 TTS 生成容量，按完整句子拆为 {len(chunks)} 段，"
            f"每批最多 {TTS_BATCH_SIZE} 段。"
        )
        chunk_paths = [work_dir / f"tts-{index:03d}.wav" for index in range(len(chunks))]
        for first in range(0, len(chunks), TTS_BATCH_SIZE):
            last = min(first + TTS_BATCH_SIZE, len(chunks))
            for attempt in range(1, TTS_SYNTHESIS_ATTEMPTS + 1):
                runner.check_cancelled()
                batch_paths = chunk_paths[first:last]
                for path in batch_paths:
                    path.unlink(missing_ok=True)
                try:
                    synthesize_qwen_batch(
                        config,
                        chunks[first:last],
                        batch_paths,
                        runner,
                        base_url=base_url,
                    )
                    if any(
                        not path.is_file() or path.stat().st_size == 0
                        for path in batch_paths
                    ):
                        raise RuntimeError("Qwen3-TTS 没有返回完整的批次音频")
                    break
                except CancelledError:
                    raise
                except (OSError, RuntimeError) as exc:
                    if attempt == TTS_SYNTHESIS_ATTEMPTS:
                        raise RuntimeError(
                            f"Qwen3-TTS 文稿分段 {first + 1}–{last} 连续 "
                            f"{TTS_SYNTHESIS_ATTEMPTS} 次生成失败：{exc}"
                        ) from exc
                    runner.logger(
                        f"Qwen3-TTS 文稿分段生成失败，正在自动重试：{exc}"
                    )
            runner.logger(f"完整文稿 TTS 进度：{last}/{len(chunks)}")
        if any(not path.is_file() or path.stat().st_size == 0 for path in chunk_paths):
            raise RuntimeError("Qwen3-TTS 没有生成全部文稿分段")
        ranges = _join_tts_audio(chunk_paths, output)
        (work_dir / "tts-chunks.json").write_text(
            json.dumps(
                [
                    {
                        "text": chunk,
                        "path": path.name,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                    }
                    for chunk, path, (start_ms, end_ms) in zip(
                        chunks, chunk_paths, ranges, strict=True
                    )
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Qwen3-TTS 未生成有效的完整配音音频")
    runner.logger("Qwen3-TTS 已生成完整配音长音频")
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


def _alignment_key(text: str) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", text)
        if unicodedata.category(character)[:1] in {"L", "N"}
    )


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别文稿文件编码：{path}")


def _prepare_transcript(job: VideoJob, language: str, cues: list[Cue]) -> str:
    path = job.translated_transcript_path(language)
    cue_text = _joined_speech_text(cues)
    if path.is_file():
        transcript = _read_text(path)
        if not transcript:
            raise RuntimeError(f"翻译文稿为空：{path}")
        if _alignment_key(transcript) != _alignment_key(cue_text):
            raise RuntimeError(f"翻译文稿与字幕内容不一致：{path.name}")
        return transcript
    path.write_text(cue_text, encoding="utf-8")
    return cue_text


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


def _alignment_cue_ranges(
    cues: list[Cue],
    raw_duration_ms: int,
) -> list[tuple[int, int]]:
    cue_weights = [max(1, len(_alignment_key(cue.text))) for cue in cues]
    cumulative_weights = [0]
    for weight in cue_weights:
        cumulative_weights.append(cumulative_weights[-1] + weight)
    total_weight = cumulative_weights[-1]
    ranges: list[tuple[int, int]] = []
    first = 0
    while first < len(cues):
        remaining_projected_ms = round(
            raw_duration_ms
            * (total_weight - cumulative_weights[first])
            / total_weight
        )
        if remaining_projected_ms <= ALIGNMENT_HARD_MAX_MS:
            ranges.append((first, len(cues)))
            break
        candidates: list[tuple[int, int, int]] = []
        for index in range(first, len(cues) - 1):
            projected_ms = round(
                raw_duration_ms
                * (cumulative_weights[index + 1] - cumulative_weights[first])
                / total_weight
            )
            if ALIGNMENT_MIN_MS <= projected_ms <= ALIGNMENT_MAX_MS:
                gap = max(0, cues[index + 1].start_ms - cues[index].end_ms)
                candidates.append(
                    (gap, -abs(projected_ms - 267_000), index + 1)
                )
        if not candidates:
            raise RuntimeError(
                f"从字幕 {cues[first].index} 开始，按实际 TTS 音频映射的 "
                "4.0–4.9 分钟文稿区间内没有字幕块边界，无法安全切分语音。"
            )
        boundary = max(candidates)[2]
        ranges.append((first, boundary))
        first = boundary
    return ranges


def _silence_intervals(
    config: AppConfig,
    runner: ProcessRunner,
    path: Path,
    *,
    noise_db: int,
    minimum_ms: int,
) -> list[tuple[int, int]]:
    lines = runner.run(
        [
            config.ffmpeg_path,
            "-hide_banner",
            "-i",
            path,
            "-af",
            f"silencedetect=noise={noise_db}dB:d={minimum_ms / 1000:.3f}",
            "-f",
            "null",
            "-",
        ],
        quiet=True,
    )
    starts: list[float] = []
    intervals: list[tuple[int, int]] = []
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
            intervals.append((round(start * 1000), round(end * 1000)))
    return intervals


def _audio_chunk_ranges(
    config: AppConfig,
    runner: ProcessRunner,
    path: Path,
    cues: list[Cue],
    cue_ranges: list[tuple[int, int]],
    total_ms: int,
) -> list[tuple[int, int]]:
    if len(cue_ranges) == 1:
        return [(0, total_ms)]
    cue_weights = [max(1, len(_alignment_key(cue.text))) for cue in cues]
    total_weight = sum(cue_weights)
    predicted = [
        round(total_ms * sum(cue_weights[:end]) / total_weight)
        for _, end in cue_ranges[:-1]
    ]
    detected: dict[tuple[int, int], list[tuple[int, int]]] = {}
    cuts = [0]
    for index, target in enumerate(predicted):
        next_target = predicted[index + 1] if index + 1 < len(predicted) else total_ms
        remaining_chunks = len(predicted) - index
        feasible_start = max(
            cuts[-1] + 1,
            total_ms - remaining_chunks * ALIGNMENT_HARD_MAX_MS,
        )
        feasible_end = min(
            cuts[-1] + ALIGNMENT_HARD_MAX_MS,
            next_target - 1,
        )
        full_window_ms = max(
            abs(target - feasible_start),
            abs(feasible_end - target),
        )
        strict_windows = [5_000, 15_000, 30_000, 60_000, full_window_ms]
        stages = [
            (window_ms, -42, 120)
            for position, window_ms in enumerate(strict_windows)
            if window_ms > 0 and window_ms not in strict_windows[:position]
        ]
        stages.extend(
            [
                (full_window_ms, -36, 80),
                (full_window_ms, -30, 40),
            ]
        )
        chosen: int | None = None
        for window_ms, noise_db, minimum_ms in stages:
            key = (noise_db, minimum_ms)
            if key not in detected:
                detected[key] = _silence_intervals(
                    config,
                    runner,
                    path,
                    noise_db=noise_db,
                    minimum_ms=minimum_ms,
                )
            intervals = detected[key]
            candidates = []
            for start_ms, end_ms in intervals:
                midpoint = (start_ms + end_ms) // 2
                if (
                    abs(midpoint - target) <= window_ms
                    and feasible_start <= midpoint <= feasible_end
                ):
                    candidates.append(
                        (abs(midpoint - target), -(end_ms - start_ms), midpoint)
                    )
            if candidates:
                chosen = min(candidates)[2]
                if (window_ms, noise_db, minimum_ms) != stages[0]:
                    runner.logger(
                        f"音频安全切点 {index + 1} 使用降级搜索："
                        f"±{window_ms / 1000:.0f}s，{noise_db}dB/{minimum_ms}ms"
                    )
                break
        if chosen is None:
            raise RuntimeError(
                f"无法在第 {index + 1} 个文稿分界附近找到安全静音，"
                "已扩大搜索窗并放宽静音标准，仍拒绝硬切语音。"
            )
        cuts.append(chosen)
    cuts.append(total_ms)
    ranges = list(zip(cuts, cuts[1:]))
    too_long = [
        end - start
        for start, end in ranges
        if end - start > ALIGNMENT_HARD_MAX_MS
    ]
    if too_long:
        raise RuntimeError(
            f"安全切分后仍有 {max(too_long) / 1000:.1f}s 的音频块超过 Forced Aligner 5 分钟限制。"
        )
    return ranges


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as source:
            return max(1, round(source.getnframes() / source.getframerate() * 1000))
    except (EOFError, wave.Error, ZeroDivisionError) as exc:
        raise ValueError(f"无法读取 WAV 时长：{path}") from exc


def _extract_alignment_chunk(
    config: AppConfig,
    runner: ProcessRunner,
    raw_file: Path,
    start_ms: int,
    end_ms: int,
    output: Path,
) -> None:
    runner.run(
        [
            config.ffmpeg_path,
            "-y",
            "-i",
            raw_file,
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-t",
            f"{(end_ms - start_ms) / 1000:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            output,
        ],
        quiet=True,
    )


def _aligned_tokens_from_items(
    items: list[dict[str, object]],
    audio_start_ms: int,
    chunk_number: int,
) -> list[AlignedToken]:
    tokens: list[AlignedToken] = []
    for item_number, item in enumerate(items, start=1):
        text = str(item.get("text") or "")
        try:
            raw_start = float(item["start"])
            raw_end = float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Forced Aligner 第 {chunk_number} 块第 {item_number} 个字符"
                "缺少有效时间戳。"
            ) from exc
        if not math.isfinite(raw_start) or not math.isfinite(raw_end):
            raise RuntimeError(
                f"Forced Aligner 第 {chunk_number} 块第 {item_number} 个字符 "
                f"{text!r} 返回了非有限时间戳：{raw_start}→{raw_end}。"
            )
        token = AlignedToken(
            text,
            audio_start_ms + round(raw_start * 1000),
            audio_start_ms + round(raw_end * 1000),
        )
        if token.start_ms < audio_start_ms or token.end_ms < token.start_ms:
            raise RuntimeError(
                f"Forced Aligner 第 {chunk_number} 块第 {item_number} 个字符 "
                f"{text!r} 返回了无效时间戳：{raw_start}→{raw_end} 秒。"
            )
        if tokens and (
            token.start_ms < tokens[-1].start_ms
            or token.end_ms < tokens[-1].end_ms
        ):
            raise RuntimeError(
                f"Forced Aligner 第 {chunk_number} 块第 {item_number} 个字符 "
                f"{text!r} 的时间戳发生倒退："
                f"{tokens[-1].start_ms}–{tokens[-1].end_ms}ms → "
                f"{token.start_ms}–{token.end_ms}ms。"
            )
        tokens.append(token)
    return tokens


def _align_full_audio(
    config: AppConfig,
    runner: ProcessRunner,
    cues: list[Cue],
    raw_file: Path,
    work_dir: Path,
    base_url: str,
) -> list[AlignedToken]:
    jobs: list[tuple[Path, str, int]] = []
    manifest_path = work_dir / "tts-chunks.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError("TTS 分段清单损坏，无法进行精确对齐") from exc
        if not isinstance(manifest, list) or not manifest:
            raise RuntimeError("TTS 分段清单为空，无法进行精确对齐")
        for index, item in enumerate(manifest, start=1):
            if not isinstance(item, dict):
                raise RuntimeError(f"TTS 分段清单第 {index} 项格式无效")
            text = str(item.get("text") or "")
            chunk = work_dir / Path(str(item.get("path") or "")).name
            try:
                audio_start = int(item["start_ms"])
                audio_end = int(item["end_ms"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"TTS 分段清单第 {index} 项缺少有效时间范围"
                ) from exc
            if (
                not text.strip()
                or not chunk.is_file()
                or audio_start < 0
                or audio_end <= audio_start
                or audio_end - audio_start > ALIGNMENT_HARD_MAX_MS
            ):
                raise RuntimeError(f"TTS 分段清单第 {index} 项无法安全对齐")
            jobs.append((chunk, text, audio_start))
        if _alignment_key("".join(item[1] for item in jobs)) != _alignment_key(
            _joined_speech_text(cues)
        ):
            raise RuntimeError("TTS 分段清单与字幕文稿不一致")
        runner.logger(f"使用 {len(jobs)} 个 TTS 原始文本分段进行精确对齐。")
    else:
        try:
            raw_duration_ms = _wav_duration_ms(raw_file)
        except ValueError:
            raw_duration_ms = max(
                1,
                round(media_duration(config, runner, raw_file) * 1000),
            )
        cue_ranges = _alignment_cue_ranges(cues, raw_duration_ms)
        audio_ranges = _audio_chunk_ranges(
            config,
            runner,
            raw_file,
            cues,
            cue_ranges,
            raw_duration_ms,
        )
        for index, ((cue_start, cue_end), (audio_start, audio_end)) in enumerate(
            zip(cue_ranges, audio_ranges, strict=True)
        ):
            chunk = work_dir / f"alignment-{index:03d}.wav"
            _extract_alignment_chunk(
                config, runner, raw_file, audio_start, audio_end, chunk
            )
            jobs.append(
                (chunk, _joined_speech_text(cues[cue_start:cue_end]), audio_start)
            )
    tokens: list[AlignedToken] = []
    for index, (chunk, text, audio_start) in enumerate(jobs):
        runner.check_cancelled()
        items = align_qwen(
            chunk,
            text,
            config.tts_language,
            base_url=base_url,
        )
        (work_dir / f"alignment-{index:03d}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        chunk_tokens = _aligned_tokens_from_items(items, audio_start, index + 1)
        point_tokens = sum(
            token.start_ms == token.end_ms for token in chunk_tokens
        )
        if point_tokens:
            runner.logger(
                f"强制对齐第 {index + 1} 块包含 {point_tokens} 个点时间戳，"
                "已保留用于整句边界匹配。"
            )
        if len(chunk_tokens) >= 20 and point_tokens * 2 > len(chunk_tokens):
            raise _TTSAlignmentQualityError(
                f"强制对齐第 {index + 1} 块有 {point_tokens}/{len(chunk_tokens)} "
                "个字符缺少声学时长，完整 TTS 音频已发生退化。"
            )
        if _alignment_key("".join(token.text for token in chunk_tokens)) != (
            _alignment_key(text)
        ):
            raise RuntimeError(
                f"Forced Aligner 第 {index + 1} 块结果与文稿不一致，拒绝继续映射。"
            )
        tokens.extend(chunk_tokens)
        runner.logger(f"强制对齐进度：{index + 1}/{len(jobs)}")
    (work_dir / "alignment.json").write_text(
        json.dumps([asdict(token) for token in tokens], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return tokens


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


def _map_sentences(
    cues: list[Cue],
    tokens: list[AlignedToken],
) -> list[AlignedSentence]:
    matched: list[tuple[SentenceUnit, list[AlignedToken]]] = []
    cursor = 0
    for unit in _sentence_units(cues):
        expected = _alignment_key(unit.text)
        first = cursor
        collected = ""
        while cursor < len(tokens) and len(collected) < len(expected):
            collected += _alignment_key(tokens[cursor].text)
            if not expected.startswith(collected):
                raise RuntimeError(
                    f"字幕 {unit.first_cue}–{unit.last_cue} 无法匹配强制对齐结果。"
                )
            cursor += 1
        if collected != expected or cursor == first:
            raise RuntimeError(
                f"字幕 {unit.first_cue}–{unit.last_cue} 的强制对齐结果不完整。"
            )
        matched.append((unit, tokens[first:cursor]))
    if cursor != len(tokens):
        raise RuntimeError("Forced Aligner 返回了无法映射到字幕的额外文本")

    aligned: list[AlignedSentence] = []
    used_audio_end = 0
    for index, (unit, unit_tokens) in enumerate(matched):
        audible = [token for token in unit_tokens if token.end_ms > token.start_ms]
        if audible:
            source_start_ms = audible[0].start_ms
            source_end_ms = audible[-1].end_ms
        else:
            point_ms = unit_tokens[0].start_ms
            next_audible_start: int | None = None
            for _, future_tokens in matched[index + 1 :]:
                future_audible = next(
                    (
                        token
                        for token in future_tokens
                        if token.end_ms > token.start_ms
                    ),
                    None,
                )
                if future_audible is not None:
                    next_audible_start = future_audible.start_ms
                    break
            source_start_ms = max(point_ms, used_audio_end)
            source_end_ms = max(
                source_start_ms,
                next_audible_start
                if next_audible_start is not None
                else source_start_ms,
            )
        aligned.append(AlignedSentence(unit, source_start_ms, source_end_ms))
        used_audio_end = max(used_audio_end, source_end_ms)
    return aligned


def _scheduled_sentence_start(
    target_start_ms: int,
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
    desired_start = min(target_start_ms, latest_safe_end - audio_duration_ms)
    desired_start = max(0, target_start_ms - MAX_SENTENCE_LEAD_MS, desired_start)
    return max(desired_start, cursor_end_ms - CROSSFADE_MS)


def _render_aligned_audio(
    config: AppConfig,
    runner: ProcessRunner,
    raw_file: Path,
    sentences: list[AlignedSentence],
    work_dir: Path,
    total_duration_ms: int,
) -> Path:
    rendered: list[tuple[AudioSegment, int]] = []
    cursor_end = 0
    total_delay = 0
    previous_delay = 0
    for index, sentence in enumerate(sentences):
        unit = sentence.unit
        source_ms = sentence.source_end_ms - sentence.source_start_ms
        if source_ms <= 0:
            runner.logger(
                f"字幕 {unit.first_cue}–{unit.last_cue} 没有可分离的 TTS 语音，"
                "已保留目标时间窗静音并继续处理。"
            )
            previous_delay = 0
            if (index + 1) % 25 == 0 or index + 1 == len(sentences):
                runner.logger(f"逐句配音处理进度：{index + 1}/{len(sentences)}")
            continue
        target_ms = unit.end_ms - unit.start_ms
        next_start = (
            sentences[index + 1].unit.start_ms
            if index + 1 < len(sentences)
            else total_duration_ms
        )
        speed = max(1.0, source_ms / max(1, target_ms))
        ratio = target_ms / max(1, source_ms)
        if ratio < 0.75 or ratio > 1.25:
            runner.logger(
                f"字幕 {unit.first_cue}–{unit.last_cue} 的时长比例为 {ratio:.3f}，"
                "已按目标时间窗强制变速或补静音，不再停止任务。"
            )
        filters = [] if speed == 1.0 else [_atempo_chain(speed)]
        filters.extend(
            [
                "aresample=24000",
                "aformat=sample_fmts=s16:channel_layouts=mono",
            ]
        )
        segment_path = work_dir / f"sentence-{index:05d}.wav"
        runner.run(
            [
                config.ffmpeg_path,
                "-y",
                "-ss",
                f"{sentence.source_start_ms / 1000:.3f}",
                "-t",
                f"{source_ms / 1000:.3f}",
                "-i",
                raw_file,
                "-af",
                ",".join(filters),
                "-c:a",
                "pcm_s16le",
                segment_path,
            ],
            quiet=True,
        )
        with segment_path.open("rb") as source:
            audio = AudioSegment.from_wav(source)
        fade_ms = min(CROSSFADE_MS, len(audio) // 2)
        if fade_ms:
            audio = audio.fade_in(fade_ms).fade_out(fade_ms)
        start_ms = _scheduled_sentence_start(
            unit.start_ms,
            next_start,
            len(audio),
            cursor_end,
            total_duration_ms,
            is_last=index + 1 == len(sentences),
        )
        lead_ms = max(0, unit.start_ms - start_ms)
        delay_ms = max(0, start_ms - unit.start_ms)
        if lead_ms:
            runner.logger(
                f"字幕 {unit.first_cue}–{unit.last_cue} 使用前置静音，"
                f"提前 {lead_ms}ms 开始以避免过度变速。"
            )
        total_delay += max(0, delay_ms - previous_delay)
        previous_delay = delay_ms
        if delay_ms > MAX_SENTENCE_DELAY_MS or total_delay > MAX_TOTAL_DELAY_MS:
            runner.logger(
                f"字幕 {unit.first_cue}–{unit.last_cue} 因单声道顺序产生 "
                f"{delay_ms}ms 延迟（累计 {total_delay}ms），继续输出。"
            )
        end_ms = start_ms + len(audio)
        if end_ms > total_duration_ms:
            keep_ms = max(0, total_duration_ms - start_ms)
            runner.logger(
                f"字幕 {unit.first_cue}–{unit.last_cue} 超出视频结尾 "
                f"{end_ms - total_duration_ms}ms，已截去越界尾音并继续输出。"
            )
            audio = audio[:keep_ms]
            end_ms = start_ms + len(audio)
        if not audio:
            continue
        rendered.append((audio, start_ms))
        cursor_end = end_ms
        if (index + 1) % 25 == 0 or index + 1 == len(sentences):
            runner.logger(f"逐句配音处理进度：{index + 1}/{len(sentences)}")

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
    aligner_base_url: str = "http://127.0.0.1:9957",
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
                "配音阶段将按原句序串行处理。"
            )
    transcript = _prepare_transcript(job, config.translation_language, cues)
    total_duration = (
        media_duration(config, runner, job.video_path)
        if job.has_video
        else max(cue.end_ms for cue in cues) / 1000
    )
    work_dir = _job_temp_dir(config, job)
    info = check_qwen_service(aligner_base_url, "aligner")
    if not info.available:
        raise RuntimeError(f"Qwen3 Forced Aligner 服务未就绪：{info.error}")
    runner.logger(f"Qwen3 Forced Aligner 模型：{info.model or '未报告'}")
    raw_file: Path | None = None
    tokens: list[AlignedToken] | None = None
    for attempt in range(1, 3):
        runner.logger(f"正在一次生成完整的 {config.tts_language} 配音…")
        raw_file = _synthesize_qwen(
            config,
            runner,
            transcript,
            work_dir,
            qwen_base_url,
        )
        try:
            tokens = _align_full_audio(
                config,
                runner,
                cues,
                raw_file,
                work_dir,
                aligner_base_url,
            )
            break
        except _TTSAlignmentQualityError as exc:
            if attempt == 2:
                raise RuntimeError(
                    f"{exc}\n完整 TTS 音频连续两次无法可靠对齐，已停止。"
                ) from exc
            runner.logger(f"{exc} 正在重新生成一次完整配音。")
    if raw_file is None or tokens is None:
        raise RuntimeError("完整 TTS 音频未能通过强制对齐质量检查")
    sentences = _map_sentences(cues, tokens)
    voice_track = _render_aligned_audio(
        config,
        runner,
        raw_file,
        sentences,
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
