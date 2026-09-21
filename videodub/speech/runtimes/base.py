from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import ModelSpec


class RuntimeAdapter(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.loaded = False

    @abstractmethod
    def load(self, model_path: Path, dependencies: dict[str, Path], options: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    def transcribe(self, audio_path: Path, language: str) -> dict[str, Any]:
        raise RuntimeError(f"Runtime {self.spec.runtime} does not support ASR")

    def synthesize(self, texts: list[str], language: str) -> list[bytes | RuntimeError]:
        raise RuntimeError(f"Runtime {self.spec.runtime} does not support TTS")


def create_runtime_adapter(spec: ModelSpec) -> RuntimeAdapter:
    if spec.runtime in {"qwen3-asr-mlx", "qwen3-asr-transformers", "qwen3-tts-mlx", "qwen3-tts-transformers"}:
        from .qwen import QwenASRAdapter, QwenTTSAdapter

        return QwenASRAdapter(spec) if spec.kind == "asr" else QwenTTSAdapter(spec)
    if spec.runtime in {"qwen3-asr-gguf", "qwen3-tts-gguf"}:
        from .qwen_gguf import QwenGGUFASRAdapter, QwenGGUFTTSAdapter

        return QwenGGUFASRAdapter(spec) if spec.kind == "asr" else QwenGGUFTTSAdapter(spec)
    raise RuntimeError(f"No runtime adapter registered for {spec.runtime}")
