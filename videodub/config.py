from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath

from .platform_utils import executable_exists, resolve_executable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = PROJECT_ROOT / "work"
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
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
    """Move the current work tree after the user chooses another directory."""
    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    if source_path == target_path:
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "output").mkdir(exist_ok=True)
        return
    target_path.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        for child in source_path.iterdir():
            _merge_move(child, target_path / child.name)
        try:
            source_path.rmdir()
        except OSError:
            pass
    (target_path / "output").mkdir(parents=True, exist_ok=True)


def migrate_legacy_layout(work_dir: str | Path) -> None:
    """Import the retired yt-video/bl-video layout without scanning its contents."""
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
        try:
            (source / ".gitkeep").unlink()
        except FileNotFoundError:
            pass
        try:
            source.rmdir()
        except OSError:
            pass


@dataclass
class AppConfig:
    work_dir: str = str(DEFAULT_WORK_DIR)
    yt_dlp_path: str = "yt-dlp"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    link_type: str = "single"
    subtitle_languages: str = DEFAULT_SUBTITLE_LANGUAGES
    overwrite: bool = True

    qwen_asr_base_url: str = "http://127.0.0.1:9956"
    qwen_tts_base_url: str = "http://127.0.0.1:9955"
    qwen_service_root: str = ""
    qwen_asr_enabled: bool = True
    qwen_tts_enabled: bool = False

    subtitle_api_base_url: str = "http://127.0.0.1:8000/v1"
    subtitle_model: str = ""
    subtitle_use_vision: bool = False
    save_model_info: bool = True
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
        # Kept for callers loading settings written by older versions. Output is
        # now always the work directory's fixed "output" child.
        return

    def normalize(self) -> "AppConfig":
        self.work_dir = _portable_directory(self.work_dir, DEFAULT_WORK_DIR)
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
        self.qwen_asr_base_url = self.qwen_asr_base_url.rstrip("/")
        self.qwen_tts_base_url = self.qwen_tts_base_url.rstrip("/")
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
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def load_config(path: Path = SETTINGS_FILE) -> AppConfig:
    config = AppConfig()
    if not path.exists():
        return config.normalize()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not raw.get("subtitle_model"):
            raw["subtitle_model"] = raw.get("subtitle_text_model", "")
        if not raw.get("work_dir"):
            legacy_download = str(raw.get("download_dir") or "").strip()
            if legacy_download and Path(legacy_download).name.lower() not in {
                "yt-video",
                "bl-video",
            }:
                raw["work_dir"] = legacy_download
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
