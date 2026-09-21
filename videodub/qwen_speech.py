"""Backward-compatible names for the generic speech HTTP client.

New application code must import :mod:`videodub.speech_client`. This module
contains no model runtime or direct inference path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .speech_client import (
    SpeechServiceInfo,
    _json_request,
    _segments_to_cues,
    extract_asr_subtitle,
    synthesize_speech,
    synthesize_speech_batch,
)
from .speech_settings import resolve_tts_reference


@dataclass(frozen=True)
class QwenServiceInfo:
    available: bool
    service_type: str
    model: str = ""
    backend: str = ""
    error: str = ""
    pid: int | None = None

    @property
    def display(self) -> str:
        if self.available:
            details = " · ".join(item for item in (self.model, self.backend) if item)
            return details or "服务可用"
        return "未检测到服务"


def check_qwen_service(
    base_url: str,
    service_type: str,
    *,
    timeout: int = 2,
) -> QwenServiceInfo:
    """Compatibility health result backed by the generic speech endpoint."""
    try:
        data = _json_request(base_url.rstrip("/") + "/health", timeout=timeout)
        if str(data.get("status") or "").casefold() != "ok":
            raise RuntimeError("服务尚未就绪")
        reported = str(data.get("type") or service_type).casefold()
        if reported not in {service_type.casefold(), "speech"}:
            raise RuntimeError(f"端口上运行的是 {reported} 服务")
        raw_pid = data.get("pid")
        pid = raw_pid if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) and raw_pid > 0 else None
        return QwenServiceInfo(
            True,
            service_type,
            str(data.get("model") or ""),
            str(data.get("runtime") or data.get("backend") or ""),
            pid=pid,
        )
    except Exception as exc:
        return QwenServiceInfo(False, service_type, error=str(exc))


def synthesize_qwen(*args, **kwargs) -> None:
    synthesize_speech(*args, **kwargs)


def synthesize_qwen_batch(*args, **kwargs) -> None:
    synthesize_speech_batch(*args, **kwargs)


__all__ = [
    "QwenServiceInfo",
    "SpeechServiceInfo",
    "check_qwen_service",
    "extract_asr_subtitle",
    "resolve_tts_reference",
    "synthesize_qwen",
    "synthesize_qwen_batch",
]
