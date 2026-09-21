from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import ModelSpec, ModelStatus
from .registry import MODEL_REGISTRY, get_model_spec
from .runtimes import RuntimeAdapter, create_runtime_adapter


class SpeechModelManager:
    def __init__(self) -> None:
        self.adapter: RuntimeAdapter | None = None
        self.spec: ModelSpec | None = None
        self.known_status: dict[str, ModelStatus] = {}

    def list_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for spec in MODEL_REGISTRY:
            status = self.known_status.get(spec.id)
            item = {
                "id": spec.id,
                "display_name": spec.display_name,
                "kind": spec.kind,
                "family": spec.family,
                "source": spec.source,
                "repo_id": spec.repo_id,
                "runtime": spec.runtime,
                "engine": spec.engine,
                "capabilities": list(spec.capabilities),
                "dependencies": [
                    {
                        "id": dependency.id,
                        "display_name": dependency.display_name,
                        "source": dependency.source,
                        "repo_id": dependency.repo_id,
                        "capabilities": list(dependency.capabilities),
                    }
                    for dependency in spec.dependencies
                ],
                "downloaded": status.downloaded if status else False,
                "supported": True,
                "runtime_available": status.runtime_available if status else False,
                "dependency_complete": status.dependency_complete if status else False,
                "available_capabilities": list(status.available_capabilities) if status else [],
                "missing_dependencies": list(status.missing_dependencies) if status else [],
                "loaded": status.loaded if status else False,
            }
            models.append(item)
        return models

    def status(
        self,
        spec: ModelSpec,
        model_path: str,
        dependency_paths: dict[str, str],
        *,
        runtime_available: bool = False,
    ) -> ModelStatus:
        downloaded = bool(model_path and Path(model_path).exists())
        missing = tuple(
            dependency.id
            for dependency in spec.dependencies
            if not dependency_paths.get(dependency.id)
            or not Path(dependency_paths[dependency.id]).exists()
        )
        unavailable_capabilities = {
            capability
            for dependency in spec.dependencies
            if dependency.id in missing
            for capability in dependency.capabilities
        }
        return ModelStatus(
            spec.id,
            downloaded,
            True,
            runtime_available,
            not missing,
            self.spec == spec and bool(self.adapter and self.adapter.loaded),
            tuple(capability for capability in spec.capabilities if capability not in unavailable_capabilities),
            missing,
        )

    def load(
        self,
        model_id: str,
        model_path: str,
        dependency_paths: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ModelStatus:
        spec = get_model_spec(model_id)
        paths = dependency_paths or {}
        status = self.status(spec, model_path, paths)
        self.known_status[spec.id] = status
        if not status.downloaded:
            raise RuntimeError(f"Model path does not exist: {model_path}")
        self.unload()
        adapter = create_runtime_adapter(spec)
        adapter.load(
            Path(model_path),
            {key: Path(value) for key, value in paths.items() if value},
            options or {},
        )
        self.adapter = adapter
        self.spec = spec
        loaded_status = self.status(
            spec, model_path, paths, runtime_available=True
        )
        self.known_status[spec.id] = loaded_status
        return loaded_status

    def unload(self) -> None:
        previous = self.spec
        if self.adapter is not None:
            self.adapter.unload()
        self.adapter = None
        self.spec = None
        if previous is not None and previous.id in self.known_status:
            self.known_status[previous.id] = replace(
                self.known_status[previous.id], loaded=False
            )

    def transcribe(self, audio_path: Path, language: str) -> dict[str, Any]:
        if self.adapter is None or self.spec is None or self.spec.kind != "asr":
            raise RuntimeError("No ASR model is loaded")
        result = self.adapter.transcribe(audio_path, language)
        result["model"] = self.spec.id
        return result

    def synthesize(self, texts: list[str], language: str) -> list[bytes | RuntimeError]:
        if self.adapter is None or self.spec is None or self.spec.kind != "tts":
            raise RuntimeError("No TTS model is loaded")
        return self.adapter.synthesize(texts, language)
