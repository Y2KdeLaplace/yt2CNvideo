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

from .platform_utils import executable_exists, resolve_executable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = PROJECT_ROOT / "work"
EXTERNAL_DIR = PROJECT_ROOT / "external"
SETTINGS_FILE = EXTERNAL_DIR / "settings.json"
LEGACY_SETTINGS_FILE = PROJECT_ROOT / "settings.json"
LEGACY_SUBTITLE_LANGUAGES = "en-orig,en.*,zh-Hans,zh-CN,zh,-live_chat"
DEFAULT_SUBTITLE_LANGUAGES = "en-orig,en"


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


def migrate_legacy_layout(work_dir: str | Path) -> None:
    """Import the retired yt-video/bl-video layout without scanning elsewhere."""
    work_path = Path(work_dir).expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    output_path = work_path / "output"
    output_path.mkdir(exist_ok=True)
    for source, target in (
        (PROJECT_ROOT / "yt-video", work_path),
        (PROJECT_ROOT / "bl-video", output_path),
    ):
        if not source.exists():
            continue
        for child in source.iterdir():
            if child.name == ".gitkeep":
                continue
            _merge_move(child, target / child.name)
        (source / ".gitkeep").unlink(missing_ok=True)
        try:
            source.rmdir()
        except OSError:
            pass


def _encryption_key() -> bytes:
    material = "|".join(
        (
            getpass.getuser(),
            platform.node(),
            str(uuid.getnode()),
            str(PROJECT_ROOT.resolve()),
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
    subtitle_use_vision: bool = False
    subtitle_detection_batch_size: int = 40
    subtitle_translation_batch_size: int = 30
    subtitle_context_radius: int = 3
    subtitle_suspect_threshold: float = 0.55
    subtitle_screenshot_count: int = 3

    asr_backend: str = ""
    asr_model_id: str = ""
    asr_model_path: str = ""
    tts_backend: str = ""
    tts_model_id: str = ""
    tts_model_path: str = ""
    tts_codec_path: str = ""
    tts_reference_audio: str = ""
    tts_reference_text: str = "请告诉我 prompt。"

    # Retained lightweight fallback settings.
    tts_provider: str = "edge"
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    tts_rate: int = 0
    piper_model_path: str = ""
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
        if not Path(self.work_dir).is_dir():
            self.work_dir = str(DEFAULT_WORK_DIR.resolve())
        self.yt_dlp_path = resolve_executable(self.yt_dlp_path, "yt-dlp")
        self.ffmpeg_path = resolve_executable(self.ffmpeg_path, "ffmpeg")
        self.ffprobe_path = resolve_executable(self.ffprobe_path, "ffprobe")
        self.overwrite = True
        self.subtitle_use_vision = False
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
        if self.tts_provider not in {"edge", "piper", "qwen"}:
            self.tts_provider = "edge"
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
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None) -> AppConfig:
    source = path
    if source is None:
        source = SETTINGS_FILE if SETTINGS_FILE.exists() else LEGACY_SETTINGS_FILE
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
        if not raw.get("subtitle_model"):
            raw["subtitle_model"] = raw.get("subtitle_text_model", "")
        if not raw.get("work_dir"):
            raw["work_dir"] = raw.get("download_dir", "")
        # Migrate version 2.0 service fields to model-manager fields.
        if not raw.get("asr_model_id") and raw.get("qwen_asr_enabled"):
            raw["asr_model_id"] = (
                "mlx-community/Qwen3-ASR-0.6B-8bit"
                if platform.system() == "Darwin"
                else "Qwen/Qwen3-ASR-0.6B"
            )
        if not raw.get("tts_model_id") and raw.get("qwen_tts_enabled"):
            raw["tts_model_id"] = (
                "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
                if platform.system() == "Darwin"
                else "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            )
        allowed = {item.name for item in fields(AppConfig)}
        values = {key: value for key, value in raw.items() if key in allowed}
        config = AppConfig(**values)
        if config.subtitle_languages == LEGACY_SUBTITLE_LANGUAGES:
            config.subtitle_languages = DEFAULT_SUBTITLE_LANGUAGES
        config.normalize()
        try:
            config.ensure_directories()
        except OSError:
            config.work_dir = str(DEFAULT_WORK_DIR.resolve())
            config.ensure_directories()
        if source == LEGACY_SETTINGS_FILE and path is None:
            save_config(config)
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
    temporary.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def api_key_from_runtime(
    entered_key: str = "",
    config: AppConfig | None = None,
) -> str:
    return (
        entered_key.strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or decrypt_api_key(config.subtitle_api_key_encrypted if config else "")
    )
