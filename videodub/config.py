from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import platform
import shutil
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath

from .platform_utils import (
    application_cache_dir,
    executable_exists,
    resolve_executable,
    user_config_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = PROJECT_ROOT / "work"
DEFAULT_CACHE_DIR = application_cache_dir()
SETTINGS_FILE = user_config_dir() / "settings.json"
LANGUAGE_MODEL_SETTINGS_NAME = "settings.json"
DEFAULT_SUBTITLE_LANGUAGES = "en-orig,en"
SUPPORTED_LANGUAGES = {
    "中文": "Chinese",
    "英语": "English",
    "日语": "Japanese",
    "韩语": "Korean",
    "德语": "German",
    "西班牙语": "Spanish",
    "法语": "French",
    "意大利语": "Italian",
    "葡萄牙语": "Portuguese",
    "俄语": "Russian",
}


def _portable_directory(value: str, default: Path) -> str:
    raw = str(value).strip()
    foreign_windows_path = os.name != "nt" and PureWindowsPath(raw).is_absolute()
    foreign_posix_path = (
        os.name == "nt"
        and PurePosixPath(raw).is_absolute()
        and not Path(raw).exists()
    )
    if not raw or foreign_windows_path or foreign_posix_path:
        return str(default.resolve())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _merge_move(source: Path, target: Path) -> None:
    """Move one tree into another, replacing files while merging directories."""
    if source.is_dir():
        if target.exists() and not target.is_dir():
            raise RuntimeError(f"迁移目录冲突：{target} 已经是文件")
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _merge_move(child, target / child.name)
        source.rmdir()
        return
    if target.exists() and target.is_dir():
        raise RuntimeError(f"迁移文件冲突：{target} 已经是目录")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))


def migrate_work_directory(source: str | Path, target: str | Path) -> None:
    """Move work contents while permanently retaining the default work folder."""
    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    if source_path == target_path:
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "output").mkdir(exist_ok=True)
        DEFAULT_WORK_DIR.mkdir(parents=True, exist_ok=True)
        return
    target_path.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        for child in list(source_path.iterdir()):
            _merge_move(child, target_path / child.name)
    (target_path / "output").mkdir(parents=True, exist_ok=True)
    # The project-local work directory is a permanent fallback and must remain.
    DEFAULT_WORK_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_WORK_DIR / "output").mkdir(exist_ok=True)


def configure_cache_directory(cache_dir: str | Path) -> Path:
    root = Path(cache_dir).expanduser().resolve()
    os.environ["VIDEODUB_CACHE_DIR"] = str(root)
    return root


def migrate_cache_directory(source: str | Path, target: str | Path) -> None:
    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    if source_path == target_path:
        target_path.mkdir(parents=True, exist_ok=True)
        return
    target_path.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        for child in list(source_path.iterdir()):
            _merge_move(child, target_path / child.name)
        source_path.rmdir()


def _encryption_key() -> bytes:
    material = "|".join(
        (
            getpass.getuser(),
            platform.node(),
            str(uuid.getnode()),
            "youtube-video-localizer-2.1",
        )
    ).encode("utf-8")
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        material,
        b"videodub-api-key-v1",
        250_000,
        dklen=32,
    )
    return base64.urlsafe_b64encode(derived)


def encrypt_api_key(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    from cryptography.fernet import Fernet

    return Fernet(_encryption_key()).encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_api_key(value: str) -> str:
    if not value:
        return ""
    try:
        from cryptography.fernet import Fernet, InvalidToken

        return Fernet(_encryption_key()).decrypt(value.encode("ascii")).decode("utf-8")
    except (ImportError, InvalidToken, ValueError, UnicodeDecodeError):
        return ""


@dataclass
class AppConfig:
    work_dir: str = str(DEFAULT_WORK_DIR)
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    yt_dlp_path: str = "yt-dlp"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    link_type: str = "single"
    subtitle_languages: str = DEFAULT_SUBTITLE_LANGUAGES
    overwrite: bool = True

    subtitle_api_base_url: str = "http://127.0.0.1:8000/v1"
    subtitle_model: str = ""
    subtitle_api_key_encrypted: str = ""
    save_model_info: bool = True
    subtitle_detection_batch_size: int = 40
    subtitle_translation_batch_size: int = 30
    asr_language: str = "English"
    translation_language: str = "Chinese"
    tts_language: str = "Chinese"

    asr_backend: str = ""
    asr_model_id: str = ""
    asr_model_path: str = ""
    tts_backend: str = ""
    tts_model_id: str = ""
    tts_model_path: str = ""
    tts_codec_path: str = ""
    tts_speaker: str = "Vivian"
    tts_voice_preset: str = ""
    tts_use_custom_voice: bool = False
    tts_reference_audio: str = ""
    tts_reference_text: str = ""
    tts_reference_text_file: str = ""

    audio_mode: str = "replace"
    original_volume: float = 0.12
    embed_subtitles: bool = True

    @property
    def download_dir(self) -> str:
        return self.work_dir

    @download_dir.setter
    def download_dir(self, value: str) -> None:
        self.work_dir = value

    @property
    def output_dir(self) -> str:
        return str(Path(self.work_dir) / "output")

    @output_dir.setter
    def output_dir(self, _value: str) -> None:
        return

    def normalize(self) -> "AppConfig":
        self.work_dir = _portable_directory(self.work_dir, DEFAULT_WORK_DIR)
        self.cache_dir = _portable_directory(self.cache_dir, DEFAULT_CACHE_DIR)
        if not Path(self.work_dir).is_dir():
            self.work_dir = str(DEFAULT_WORK_DIR.resolve())
        self.yt_dlp_path = resolve_executable(self.yt_dlp_path, "yt-dlp")
        self.ffmpeg_path = resolve_executable(self.ffmpeg_path, "ffmpeg")
        self.ffprobe_path = resolve_executable(self.ffprobe_path, "ffprobe")
        self.overwrite = True
        self.subtitle_detection_batch_size = max(
            10, min(int(self.subtitle_detection_batch_size), 100)
        )
        self.subtitle_translation_batch_size = max(
            10, min(int(self.subtitle_translation_batch_size), 80)
        )
        self.original_volume = max(0.0, min(float(self.original_volume), 1.0))
        self.subtitle_api_base_url = self.subtitle_api_base_url.rstrip("/")
        supported = set(SUPPORTED_LANGUAGES.values())
        if self.asr_language not in supported:
            self.asr_language = "English"
        if self.translation_language not in supported:
            self.translation_language = "Chinese"
        if self.tts_language not in supported:
            self.tts_language = "Chinese"
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
        DEFAULT_WORK_DIR.mkdir(parents=True, exist_ok=True)
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None) -> AppConfig:
    source = path or SETTINGS_FILE
    config = AppConfig()
    if not source.exists():
        config.normalize()
        try:
            config.ensure_directories()
        except OSError:
            # A remembered removable/network folder may still appear to exist
            # while no longer being writable or traversable.
            config.work_dir = str(DEFAULT_WORK_DIR.resolve())
            config.ensure_directories()
        return config
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(AppConfig)}
        values = {key: value for key, value in raw.items() if key in allowed}
        config = AppConfig(**values)
        config.normalize()
        try:
            config.ensure_directories()
        except OSError:
            config.work_dir = str(DEFAULT_WORK_DIR.resolve())
            config.ensure_directories()
        return config
    except (OSError, ValueError, TypeError):
        config.work_dir = str(DEFAULT_WORK_DIR.resolve())
        config.normalize()
        config.ensure_directories()
        return config


def save_config(config: AppConfig, path: Path = SETTINGS_FILE) -> None:
    config.normalize()
    config.ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    values = asdict(config)
    values["subtitle_api_key_encrypted"] = ""
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def language_model_settings_path(config: AppConfig) -> Path:
    return Path(config.cache_dir) / LANGUAGE_MODEL_SETTINGS_NAME


def load_language_model_info(config: AppConfig) -> AppConfig:
    path = language_model_settings_path(config)
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return config
    if not isinstance(values, dict) or not values.get("save_model_info", True):
        return config
    config.subtitle_api_base_url = str(
        values.get("subtitle_api_base_url") or config.subtitle_api_base_url
    ).strip()
    config.subtitle_model = str(
        values.get("subtitle_model") or config.subtitle_model
    ).strip()
    config.subtitle_api_key_encrypted = str(
        values.get("subtitle_api_key_encrypted") or ""
    ).strip()
    return config


def save_language_model_info(config: AppConfig) -> Path:
    path = language_model_settings_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            values = {}
    except (OSError, ValueError, TypeError):
        values = {}
    enabled = bool(config.save_model_info)
    values.update(
        {
            "save_model_info": enabled,
            "subtitle_api_base_url": (
                config.subtitle_api_base_url if enabled else ""
            ),
            "subtitle_model": config.subtitle_model if enabled else "",
            "subtitle_api_key_encrypted": (
                config.subtitle_api_key_encrypted if enabled else ""
            ),
        }
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def api_key_from_runtime(
    entered_key: str = "",
    config: AppConfig | None = None,
) -> str:
    return (
        entered_key.strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or decrypt_api_key(config.subtitle_api_key_encrypted if config else "")
    )
