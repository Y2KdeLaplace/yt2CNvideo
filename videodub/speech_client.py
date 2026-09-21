from __future__ import annotations

import base64
import json
import mimetypes
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .media import VideoJob
from .runner import ProcessRunner
from .speech.constants import DEFAULT_SPEECH_PORT, speech_base_url
from .subtitles import Cue, write_srt


SPEECH_SERVICE_URL = speech_base_url(DEFAULT_SPEECH_PORT)


@dataclass(frozen=True)
class SpeechServiceInfo:
    available: bool
    loaded: bool = False
    model: str = ""
    kind: str = ""
    runtime: str = ""
    error: str = ""
    pid: int | None = None

    @property
    def display(self) -> str:
        if self.available:
            details = " · ".join(item for item in (self.model, self.runtime) if item)
            return details or "服务可用"
        return "未检测到服务"


def _json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 10,
    method: str | None = None,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        value = json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict):
                detail = str(parsed.get("detail") or detail)
        except ValueError:
            pass
        raise RuntimeError(f"Speech service HTTP {exc.code}: {detail[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 speech service：{exc.reason}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Speech service 返回格式异常")
    return value


def check_speech_service(base_url: str, *, timeout: int = 2) -> SpeechServiceInfo:
    try:
        data = _json_request(base_url.rstrip("/") + "/health", timeout=timeout)
        if str(data.get("status") or "").casefold() != "ok":
            raise RuntimeError("服务尚未就绪")
        raw_pid = data.get("pid")
        pid = raw_pid if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) and raw_pid > 0 else None
        return SpeechServiceInfo(
            True,
            bool(data.get("loaded")),
            str(data.get("model") or ""),
            str(data.get("type") or ""),
            str(data.get("runtime") or ""),
            pid=pid,
        )
    except Exception as exc:
        return SpeechServiceInfo(False, error=str(exc))


class SpeechClient:
    def __init__(self, base_url: str = SPEECH_SERVICE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.last_speech_errors: list[str] = []

    def health(self, timeout: int = 2) -> SpeechServiceInfo:
        return check_speech_service(self.base_url, timeout=timeout)

    def models(self) -> list[dict[str, Any]]:
        value = _json_request(self.base_url + "/models")
        data = value.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Speech service 模型列表格式异常")
        return [item for item in data if isinstance(item, dict)]

    def load_model(
        self,
        model: str,
        model_path: str,
        dependencies: dict[str, str],
        options: dict[str, Any] | None = None,
    ) -> None:
        _json_request(
            self.base_url + "/models/load",
            payload={
                "model": model,
                "model_path": model_path,
                "dependencies": dependencies,
                "options": options or {},
            },
            timeout=1800,
        )

    def unload_model(self) -> None:
        _json_request(self.base_url + "/models/unload", payload={}, timeout=30)

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str,
        timeout: int = 3600,
    ) -> dict[str, Any]:
        boundary = "----SCIPSpeech" + uuid.uuid4().hex
        newline = b"\r\n"
        content_type = mimetypes.guess_type(audio_path.name)[0] or "audio/wav"
        parts = [
            f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="model"', b"", model.encode(),
            f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="language"', b"", language.encode(),
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"'.encode(),
            f"Content-Type: {content_type}".encode(), b"", audio_path.read_bytes(),
            f"--{boundary}--".encode(), b"",
        ]
        request = urllib.request.Request(
            self.base_url + "/audio/transcriptions",
            data=newline.join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(detail)
                detail = str(parsed.get("detail") or detail) if isinstance(parsed, dict) else detail
            except ValueError:
                pass
            raise RuntimeError(f"ASR 返回 HTTP {exc.code}：{detail[:1200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接 speech service：{exc.reason}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("ASR 返回格式异常")
        return value

    def synthesize(
        self,
        texts: list[str],
        *,
        model: str,
        language: str,
        timeout: int = 1800,
    ) -> list[bytes | None]:
        payload: dict[str, Any] = {"model": model, "language": language}
        if len(texts) == 1:
            payload["input"] = texts[0]
        else:
            payload["texts"] = texts
        response = _json_request(self.base_url + "/audio/speech", payload=payload, timeout=timeout)
        values = response.get("audio_base64_list")
        if not isinstance(values, list) or len(values) != len(texts):
            raise RuntimeError("TTS 批量返回的音频数量不一致")
        errors = response.get("errors") if isinstance(response.get("errors"), list) else []
        outputs: list[bytes | None] = []
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value:
                outputs.append(None)
            else:
                outputs.append(base64.b64decode(value))
        self.last_speech_errors = [
            str(errors[index]) if index < len(errors) and errors[index] else ""
            for index in range(len(texts))
        ]
        return outputs


def _segments_to_cues(segments: Any, fallback_text: str) -> list[Cue]:
    cues: list[Cue] = []
    if isinstance(segments, list):
        for item in segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                start_ms = round(float(item["start"]) * 1000)
                end_ms = round(float(item["end"]) * 1000)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"ASR 时间戳无效：cue {len(cues)+1} text={text!r} "
                    f"start={item.get('start')} end={item.get('end')}"
                ) from exc
            cues.append(Cue(len(cues) + 1, start_ms, end_ms, text))
    if not cues and fallback_text.strip():
        raise RuntimeError("ASR 只有文本，没有声学时间戳；不能用整段视频时长代替句子时间窗")
    from .sentences import validate_timeline

    validate_timeline(cues, allow_punctuation=True)
    return cues


def extract_asr_subtitle(
    config: AppConfig,
    runner: ProcessRunner,
    job: VideoJob,
    *,
    language: str = "English",
    base_url: str = SPEECH_SERVICE_URL,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="videodub-asr-") as temp:
        wav_path = Path(temp) / "audio.wav"
        runner.logger("正在从视频音轨直接解码为 16 kHz 无损音频…")
        runner.run([
            config.ffmpeg_path, "-nostdin", "-y", "-i", job.video_path,
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", wav_path,
        ])
        runner.logger("16 kHz 音频解码完成，准备提交 ASR 识别…")
        runner.check_cancelled()
        client = SpeechClient(base_url)
        info = client.health()
        if not info.available or not info.loaded:
            raise RuntimeError(f"Speech ASR 服务未就绪：{info.error or '模型未加载'}")
        runner.logger(f"ASR 模型：{info.model or '未报告'}")
        result = client.transcribe(
            wav_path,
            model=info.model,
            language=language,
        )
    cues = _segments_to_cues(result.get("segments"), str(result.get("text") or ""))
    if not cues:
        raise RuntimeError("ASR 没有返回可写入的字幕内容")
    write_srt(job.asr_subtitle_path, cues)
    runner.logger(f"ASR 字幕：{job.asr_subtitle_path}")
    return job.asr_subtitle_path


def synthesize_speech_batch(
    config: AppConfig,
    texts: list[str],
    outputs: list[Path],
    runner: ProcessRunner,
    *,
    base_url: str = SPEECH_SERVICE_URL,
) -> None:
    if len(texts) != len(outputs) or not texts:
        raise ValueError("TTS 批量输入与输出数量不一致")
    client = SpeechClient(base_url)
    info = client.health()
    if not info.available or not info.loaded:
        raise RuntimeError(f"Speech TTS 服务未就绪：{info.error or '模型未加载'}")
    values = client.synthesize(
        texts,
        model=info.model,
        language=config.tts_language,
    )
    failures: list[str] = []
    for index, (value, output) in enumerate(zip(values, outputs, strict=True)):
        if value is None:
            detail = client.last_speech_errors[index] if index < len(client.last_speech_errors) else ""
            failures.append(detail or f"{output.name} 没有音频")
        else:
            output.write_bytes(value)
    if failures:
        raise RuntimeError("TTS 批量生成部分失败：" + "; ".join(failures))


def synthesize_speech(
    config: AppConfig,
    text: str,
    output: Path,
    runner: ProcessRunner,
    *,
    base_url: str = SPEECH_SERVICE_URL,
) -> None:
    synthesize_speech_batch(config, [text], [output], runner, base_url=base_url)
