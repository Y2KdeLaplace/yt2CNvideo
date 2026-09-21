from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .model_management.voices import resolve_voice_sample


def _read_reference_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别参考文本编码：{path}")


def resolve_tts_reference(config: AppConfig) -> tuple[str, str]:
    if config.tts_use_custom_voice:
        audio = Path(config.tts_reference_audio).expanduser()
        text_file = Path(config.tts_reference_text_file).expanduser()
        label = "自定义声音"
    else:
        sample = resolve_voice_sample(config.tts_voice_preset)
        if sample is None:
            raise ValueError("请为 Base TTS 模型选择一个声音样本。")
        audio, text_file = sample.audio_path, sample.text_path
        label = f"预设声音 {config.tts_voice_preset}"
    if not audio.is_file():
        raise ValueError(f"{label}的参考音频不存在：{audio}")
    if not text_file.is_file():
        raise ValueError(f"{label}的对应文本文件不存在：{text_file}")
    text = _read_reference_text(text_file)
    if not text:
        raise ValueError(f"{label}的对应文本为空：{text_file}")
    return str(audio.resolve()), text
