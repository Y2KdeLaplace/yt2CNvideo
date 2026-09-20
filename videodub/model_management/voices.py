from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import PROJECT_ROOT
from ..runner import ProcessRunner
from ..subtitles import parse_srt_text, subtitle_transcript


VOICE_SAMPLE_DIR = PROJECT_ROOT / "sample_voice"
AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
TEXT_EXTENSIONS = {".txt", ".text", ".md", ".srt", ".vtt"}


@dataclass(frozen=True)
class VoiceSample:
    name: str
    audio_path: Path
    text_path: Path


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            content = raw.decode(encoding).strip()
            if path.suffix.casefold() in {".srt", ".vtt"}:
                cues = parse_srt_text(content)
                if not cues:
                    raise ValueError(f"字幕文件中没有可用文本：{path}")
                return subtitle_transcript(cues)
            return content
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别声音样本文本编码：{path}")


def list_voice_samples(root: Path = VOICE_SAMPLE_DIR) -> list[VoiceSample]:
    if not root.is_dir():
        return []
    samples: list[VoiceSample] = []
    for folder in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not folder.is_dir():
            continue
        preferred_audio = folder / f"{folder.name}.wav"
        audio = preferred_audio if preferred_audio.is_file() else next(
            (
                item
                for item in sorted(folder.iterdir())
                if item.is_file() and item.suffix.casefold() in AUDIO_EXTENSIONS
            ),
            None,
        )
        preferred_text = folder / f"{folder.name}.txt"
        text = preferred_text if preferred_text.is_file() else next(
            (
                item
                for item in sorted(folder.iterdir())
                if item.is_file() and item.suffix.casefold() in TEXT_EXTENSIONS
            ),
            None,
        )
        if audio is not None and text is not None:
            samples.append(VoiceSample(folder.name, audio.resolve(), text.resolve()))
    return samples


def resolve_voice_sample(name: str) -> VoiceSample | None:
    return next((sample for sample in list_voice_samples() if sample.name == name), None)


def import_voice_sample(
    media_path: str | Path,
    text_path: str | Path,
    ffmpeg_path: str,
    runner: ProcessRunner,
    *,
    name: str | None = None,
    root: Path = VOICE_SAMPLE_DIR,
) -> VoiceSample:
    media = Path(media_path).expanduser().resolve()
    text = Path(text_path).expanduser().resolve()
    if not media.is_file():
        raise ValueError(f"声音样本文件不存在：{media}")
    if not text.is_file():
        raise ValueError(f"对应文本文件不存在：{text}")
    transcript = _read_text(text)
    if not transcript:
        raise ValueError(f"对应文本为空：{text}")

    base_name = re.sub(
        r"[\\/:*?\"<>|]+",
        "-",
        name.strip() if name is not None else media.stem,
    ).strip(" .-")
    base_name = base_name or "voice"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / base_name
    suffix = 2
    while destination.exists():
        destination = root / f"{base_name}-{suffix}"
        suffix += 1
    destination.mkdir()
    sample_name = destination.name
    audio_target = destination / f"{sample_name}.wav"
    text_target = destination / f"{sample_name}.txt"
    try:
        action = (
            "转换音频"
            if media.suffix.casefold() in AUDIO_EXTENSIONS
            else "从媒体文件提取音频"
        )
        runner.logger(f"正在{action}：{media.name}")
        runner.run(
            [
                ffmpeg_path,
                "-nostdin",
                "-y",
                "-i",
                media,
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                audio_target,
            ]
        )
        if not audio_target.is_file() or audio_target.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg 没有生成有效的 WAV 文件：{audio_target}")
        text_target.write_text(transcript + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    runner.logger(f"声音样本已导入：{sample_name}")
    return VoiceSample(sample_name, audio_target.resolve(), text_target.resolve())
