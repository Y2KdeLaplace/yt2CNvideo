from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from dataclasses import asdict, dataclass, fields

from .platform_utils import executable_exists, resolve_executable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
LEGACY_SUBTITLE_LANGUAGES = "en-orig,en.*,zh-Hans,zh-CN,zh,-live_chat"
DEFAULT_SUBTITLE_LANGUAGES = "en-orig,en"


def _portable_directory(value: str, default_name: str) -> str:
    raw = str(value).strip()
    foreign_windows_path = os.name != "nt" and PureWindowsPath(raw).is_absolute()
    foreign_posix_path = (
        os.name == "nt"
        and PurePosixPath(raw).is_absolute()
        and not Path(raw).exists()
    )
    if not raw or foreign_windows_path or foreign_posix_path:
        return str((PROJECT_ROOT / default_name).resolve())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


@dataclass
class AppConfig:
    download_dir: str = str(PROJECT_ROOT / "yt-video")
    output_dir: str = str(PROJECT_ROOT / "bl-video")
    yt_dlp_path: str = "yt-dlp"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    link_type: str = "single"
    download_subtitles: bool = True
    subtitle_languages: str = DEFAULT_SUBTITLE_LANGUAGES
    overwrite: bool = False

    subtitle_api_base_url: str = "http://127.0.0.1:8000/v1"
    subtitle_model: str = ""
    subtitle_use_vision: bool = False
    subtitle_detection_batch_size: int = 40
    subtitle_translation_batch_size: int = 30
    subtitle_context_radius: int = 3
    subtitle_suspect_threshold: float = 0.55
    subtitle_screenshot_count: int = 3

    tts_provider: str = "edge"
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    tts_rate: int = 0
    piper_model_path: str = ""

    audio_mode: str = "replace"
    original_volume: float = 0.12
    embed_subtitles: bool = True

    def normalize(self) -> "AppConfig":
        self.download_dir = _portable_directory(self.download_dir, "yt-video")
        self.output_dir = _portable_directory(self.output_dir, "bl-video")
        self.yt_dlp_path = resolve_executable(self.yt_dlp_path, "yt-dlp")
        self.ffmpeg_path = resolve_executable(self.ffmpeg_path, "ffmpeg")
        self.ffprobe_path = resolve_executable(self.ffprobe_path, "ffprobe")
        self.subtitle_detection_batch_size = max(
            10, min(int(self.subtitle_detection_batch_size), 100)
        )
        self.subtitle_translation_batch_size = max(
            10, min(int(self.subtitle_translation_batch_size), 80)
        )
        self.subtitle_context_radius = max(
            1, min(int(self.subtitle_context_radius), 10)
        )
        self.subtitle_suspect_threshold = max(
            0.0, min(float(self.subtitle_suspect_threshold), 1.0)
        )
        self.subtitle_screenshot_count = max(
            1, min(int(self.subtitle_screenshot_count), 5)
        )
        self.tts_rate = max(-50, min(int(self.tts_rate), 50))
        self.original_volume = max(0.0, min(float(self.original_volume), 1.0))
        self.subtitle_api_base_url = self.subtitle_api_base_url.rstrip("/")
        return self

    def validate_core(self) -> list[str]:
        problems: list[str] = []
        for label, value in (
            ("yt-dlp", self.yt_dlp_path),
            ("ffmpeg", self.ffmpeg_path),
            ("ffprobe", self.ffprobe_path),
        ):
            if not executable_exists(value):
                problems.append(f"{label} 不存在：{value}")
        return problems

    def ensure_directories(self) -> None:
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def load_config(path: Path = SETTINGS_FILE) -> AppConfig:
    config = AppConfig()
    if not path.exists():
        return config.normalize()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not raw.get("subtitle_model"):
            raw["subtitle_model"] = raw.get("subtitle_text_model", "")
        if raw.get("tts_provider") not in {"edge", "piper"}:
            raw["tts_provider"] = "edge"
            raw["tts_voice"] = "zh-CN-XiaoxiaoNeural"
        allowed = {item.name for item in fields(AppConfig)}
        values = {key: value for key, value in raw.items() if key in allowed}
        config = AppConfig(**values)
        if config.subtitle_languages == LEGACY_SUBTITLE_LANGUAGES:
            config.subtitle_languages = DEFAULT_SUBTITLE_LANGUAGES
        return config.normalize()
    except (OSError, ValueError, TypeError):
        return config.normalize()


def save_config(config: AppConfig, path: Path = SETTINGS_FILE) -> None:
    config.normalize()
    path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def api_key_from_runtime(entered_key: str = "") -> str:
    return entered_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
