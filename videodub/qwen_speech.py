from __future__ import annotations

import base64
import json
import mimetypes
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig, PROJECT_ROOT
from .media import VideoJob, media_duration
from .model_manager import (
    InstalledModel,
    crispasr_executable,
    first_model_file,
    read_installed_model,
)
from .runner import ProcessRunner
from .subtitles import Cue, read_srt, write_srt


ASR_SERVICE_URL = "http://127.0.0.1:9956"
TTS_SERVICE_URL = "http://127.0.0.1:9955"
TTS_VOICE_PRESETS = {
    name: (
        PROJECT_ROOT / "sample_voice" / name / f"{name}.wav",
        PROJECT_ROOT / "sample_voice" / name / f"{name}.txt",
    )
    for name in ("Diana", "Eileen")
}

_INCOMPATIBLE_CRISPASR_ASR_REPOSITORIES = {
    "handy-computer/qwen3-asr-0.6b-gguf",
}


@dataclass(frozen=True)
class QwenServiceInfo:
    available: bool
    service_type: str
    model: str = ""
    backend: str = ""
    error: str = ""

    @property
    def display(self) -> str:
        if self.available:
            details = " · ".join(item for item in (self.model, self.backend) if item)
            return details or "服务可用"
        return "未检测到服务"


def _json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen 服务返回 HTTP {exc.code}：{detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 Qwen 服务：{exc.reason}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Qwen 服务返回格式异常")
    return value


def check_qwen_service(
    base_url: str,
    service_type: str,
    *,
    timeout: int = 2,
) -> QwenServiceInfo:
    try:
        data = _json_request(base_url.rstrip("/") + "/health", timeout=timeout)
        reported_type = str(data.get("type") or service_type).lower()
        if str(data.get("status") or "").lower() != "ok":
            raise RuntimeError("服务尚未就绪")
        if reported_type != service_type:
            raise RuntimeError(f"端口上运行的是 {reported_type} 服务")
        return QwenServiceInfo(
            True,
            service_type,
            str(data.get("model") or ""),
            str(data.get("backend") or ""),
        )
    except Exception as exc:
        return QwenServiceInfo(False, service_type, error=str(exc))


def _multipart_audio(path: Path, language: str) -> tuple[bytes, str]:
    boundary = "----VideoDub" + uuid.uuid4().hex
    newline = b"\r\n"
    content_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    parts = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="language"',
        b"",
        language.encode("utf-8"),
        f"--{boundary}".encode(),
        (
            f'Content-Disposition: form-data; name="audio"; '
            f'filename="{path.name}"'
        ).encode("utf-8"),
        f"Content-Type: {content_type}".encode(),
        b"",
        path.read_bytes(),
        f"--{boundary}--".encode(),
        b"",
    ]
    return newline.join(parts), boundary


def _post_audio(
    base_url: str,
    audio_path: Path,
    *,
    language: str,
    timeout: int = 3600,
) -> dict[str, Any]:
    body, boundary = _multipart_audio(audio_path, language)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/asr",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen3-ASR 返回 HTTP {exc.code}：{detail[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 Qwen3-ASR：{exc.reason}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Qwen3-ASR 返回格式异常")
    return value


def _segments_to_cues(
    segments: Any,
    fallback_text: str,
    duration_seconds: float,
) -> list[Cue]:
    cues: list[Cue] = []
    if isinstance(segments, list):
        for item in segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                start_ms = max(0, round(float(item.get("start") or 0) * 1000))
                end_ms = max(start_ms + 1, round(float(item.get("end")) * 1000))
            except (TypeError, ValueError):
                continue
            cues.append(Cue(len(cues) + 1, start_ms, end_ms, text))
    if not cues and fallback_text.strip():
        cues.append(
            Cue(
                1,
                0,
                max(1, round(duration_seconds * 1000)),
                fallback_text.strip(),
            )
        )
    return cues


def _validate_crispasr_asr_model(model: Path) -> InstalledModel | None:
    installed = read_installed_model(model)
    if (
        installed is not None
        and installed.repo_id.casefold()
        in _INCOMPATIBLE_CRISPASR_ASR_REPOSITORIES
    ):
        raise RuntimeError(
            "当前 GGUF 文件不是 CrispASR 所需的转换格式。请在模型菜单中卸载旧版 "
            "handy-computer 模型，并重新下载 cstr/qwen3-asr-0.6b-GGUF。"
        )
    return installed


def _crispasr_language_code(language: str) -> str:
    normalized = language.strip().casefold()
    return {
        "english": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "cantonese": "yue",
        "japanese": "ja",
        "korean": "ko",
        "german": "de",
        "spanish": "es",
        "french": "fr",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
    }.get(normalized, normalized or "en")


def _crispasr_asr_runtime_options(language: str, vad_model: Path) -> list[str]:
    return [
        "-l",
        _crispasr_language_code(language),
        "--vad",
        "-vm",
        str(vad_model),
    ]


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
        try:
            audio, text_file = TTS_VOICE_PRESETS[config.tts_voice_preset]
        except KeyError as exc:
            raise ValueError("请为 Base TTS 模型选择 Diana、Eileen 或自定义声音。") from exc
        label = f"预设声音 {config.tts_voice_preset}"
    if not audio.is_file():
        raise ValueError(f"{label}的参考音频不存在：{audio}")
    if not text_file.is_file():
        raise ValueError(f"{label}的对应文本文件不存在：{text_file}")
    text = _read_reference_text(text_file)
    if not text:
        raise ValueError(f"{label}的对应文本为空：{text_file}")
    return str(audio.resolve()), text


def extract_asr_subtitle(
    config: AppConfig,
    runner: ProcessRunner,
    job: VideoJob,
    *,
    language: str = "English",
    base_url: str = ASR_SERVICE_URL,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="videodub-asr-") as temp:
        wav_path = Path(temp) / "audio.wav"
        runner.logger("正在从视频音轨直接解码为 16 kHz 无损音频…")
        runner.run(
            [
                config.ffmpeg_path,
                "-y",
                "-i",
                job.video_path,
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                wav_path,
            ]
        )
        runner.check_cancelled()
        if config.asr_backend == "gguf":
            executable = crispasr_executable()
            if executable is None:
                raise RuntimeError("CrispASR 运行环境未安装")
            model = first_model_file(config.asr_model_path, "*.gguf")
            installed = _validate_crispasr_asr_model(model)
            vad_model = Path(installed.vad_path) if installed else Path()
            if not installed or not vad_model.is_file():
                raise RuntimeError(
                    "ASR GGUF 缺少 Silero VAD 依赖，请在模型菜单中重新下载该模型。"
                )
            prefix = Path(temp) / "recognized"
            runner.run(
                [
                    executable,
                    "--backend",
                    "qwen3",
                    "-m",
                    model,
                    "-f",
                    wav_path,
                    *_crispasr_asr_runtime_options(language, vad_model),
                    "-osrt",
                    "-of",
                    prefix,
                ]
            )
            srt = prefix.with_suffix(".srt")
            if not srt.is_file():
                candidates = list(Path(temp).glob("*.srt"))
                if not candidates:
                    raise RuntimeError("CrispASR 未生成 SRT 字幕")
                srt = candidates[0]
            cues = read_srt(srt)
            if not cues:
                raise RuntimeError("CrispASR 生成的字幕为空")
            write_srt(job.asr_subtitle_path, cues)
            runner.logger(f"ASR 字幕：{job.asr_subtitle_path}")
            return job.asr_subtitle_path
        info = check_qwen_service(base_url, "asr")
        if not info.available:
            raise RuntimeError(f"Qwen3-ASR 模型服务未就绪：{info.error}")
        runner.logger(f"Qwen3-ASR 模型：{info.model or '未报告'}")
        result = _post_audio(base_url, wav_path, language=language)
    cues = _segments_to_cues(
        result.get("segments"),
        str(result.get("text") or ""),
        media_duration(config, runner, job.video_path),
    )
    if not cues:
        raise RuntimeError("Qwen3-ASR 没有返回可写入的字幕内容")
    write_srt(job.asr_subtitle_path, cues)
    runner.logger(f"ASR 字幕：{job.asr_subtitle_path}")
    return job.asr_subtitle_path


def _copy_qwen_audio(response: dict[str, Any], target: Path) -> None:
    encoded = response.get("audio_base64")
    if isinstance(encoded, str) and encoded:
        target.write_bytes(base64.b64decode(encoded))
        return
    raise RuntimeError("Qwen3-TTS 没有返回音频")


def synthesize_qwen(
    config: AppConfig,
    text: str,
    output: Path,
    runner: ProcessRunner,
    *,
    base_url: str = TTS_SERVICE_URL,
) -> None:
    if config.tts_backend == "gguf":
        executable = crispasr_executable()
        installed = read_installed_model(config.tts_model_path)
        if executable is None or installed is None:
            raise RuntimeError("Qwen3-TTS GGUF 运行环境或模型不存在")
        reference_audio = config.tts_reference_audio
        reference_text = config.tts_reference_text
        if installed.variant == "base":
            reference_audio, reference_text = resolve_tts_reference(config)
        runner.run(
            (
                [
                    executable,
                    "--backend",
                    "qwen3-tts",
                    "-m",
                    first_model_file(config.tts_model_path, "*.gguf"),
                    "--codec-model",
                    installed.codec_path,
                    "--voice",
                    reference_audio,
                    "--ref-text",
                    reference_text,
                    "-l",
                    _crispasr_language_code(config.tts_language),
                    "--tts",
                    text,
                    "--tts-output",
                    output,
                ]
                if installed.variant == "base"
                else [
                    executable,
                    "--backend",
                    "qwen3-tts-customvoice",
                    "-m",
                    first_model_file(config.tts_model_path, "*.gguf"),
                    "--codec-model",
                    installed.codec_path,
                    "--voice",
                    config.tts_speaker,
                    "-l",
                    _crispasr_language_code(config.tts_language),
                    "--tts",
                    text,
                    "--tts-output",
                    output,
                ]
            ),
            quiet=True,
        )
        return
    response = _json_request(
        base_url.rstrip("/") + "/v1/tts",
        payload={"text": text, "language": config.tts_language},
        timeout=1800,
    )
    _copy_qwen_audio(response, output)
