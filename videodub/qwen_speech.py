from __future__ import annotations

import base64
import json
import mimetypes
import shutil
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
from .runner import ProcessRunner
from .subtitles import Cue, write_srt


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


def extract_asr_subtitle(
    config: AppConfig,
    runner: ProcessRunner,
    job: VideoJob,
    *,
    language: str = "English",
) -> Path:
    info = check_qwen_service(config.qwen_asr_base_url, "asr")
    if not info.available:
        raise RuntimeError(
            "未检测到可用的 Qwen3-ASR 服务。请按 README 启动服务后重试。"
            f"\n{info.error}"
        )
    runner.logger(f"Qwen3-ASR 模型：{info.model or '未报告'}")
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
        result = _post_audio(
            config.qwen_asr_base_url,
            wav_path,
            language=language,
        )
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


def _copy_qwen_audio(
    config: AppConfig,
    response: dict[str, Any],
    target: Path,
) -> None:
    encoded = response.get("audio_base64")
    if isinstance(encoded, str) and encoded:
        target.write_bytes(base64.b64decode(encoded))
        return
    audio_url = response.get("audio_url")
    if isinstance(audio_url, str) and audio_url:
        with urllib.request.urlopen(
            urllib.parse.urljoin(config.qwen_tts_base_url + "/", audio_url),
            timeout=300,
        ) as source:
            target.write_bytes(source.read())
        return
    raw_path = str(response.get("audio_path") or "").strip()
    if raw_path:
        candidate = Path(raw_path).expanduser()
        candidates = [candidate] if candidate.is_absolute() else []
        if config.qwen_service_root.strip():
            candidates.append(Path(config.qwen_service_root).expanduser() / candidate)
        candidates.extend([PROJECT_ROOT / candidate, Path.cwd() / candidate])
        for source in candidates:
            if source.is_file():
                shutil.copy2(source, target)
                return
    raise RuntimeError(
        "Qwen3-TTS 已完成生成，但客户端无法读取音频。"
        "请在“模型路径”中填写 qwen-speech-mlx 项目目录，"
        "或让兼容服务返回 audio_base64/audio_url。"
    )


def synthesize_qwen(
    config: AppConfig,
    text: str,
    output: Path,
) -> None:
    response = _json_request(
        config.qwen_tts_base_url.rstrip("/") + "/v1/tts",
        payload={"text": text},
        timeout=1800,
    )
    _copy_qwen_audio(config, response, output)
