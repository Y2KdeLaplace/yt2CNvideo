from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

from .manager import SpeechModelManager
from .providers import delete_repository, download_repository
from .registry import get_model_spec


def create_speech_app(manager: SpeechModelManager | None = None) -> Any:
    from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile

    active = manager or SpeechModelManager()
    app = FastAPI(title="SCIP Speech Service")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "scip-speech",
            "pid": os.getpid(),
            "loaded": bool(active.adapter and active.adapter.loaded),
            "model": active.spec.id if active.spec else "",
            "type": active.spec.kind if active.spec else "speech",
            "runtime": active.spec.runtime if active.spec else "",
        }

    @app.get("/models")
    def models() -> dict[str, Any]:
        return {"data": active.list_models()}

    @app.get("/models/{model_id}")
    def model(model_id: str) -> dict[str, Any]:
        for item in active.list_models():
            if item["id"] == model_id:
                return item
        raise HTTPException(status_code=404, detail="Unknown model")

    @app.post("/models/load")
    def load(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            status = active.load(
                str(payload.get("model") or ""),
                str(payload.get("model_path") or ""),
                dict(payload.get("dependencies") or {}),
                dict(payload.get("options") or {}),
            )
            return {"status": "loaded", "model": status.id}
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/models/unload")
    def unload() -> dict[str, str]:
        active.unload()
        return {"status": "unloaded"}

    @app.post("/models/download")
    def download(payload: dict[str, Any] = Body(...)) -> dict[str, str]:
        from ..runner import ProcessRunner

        try:
            model_id = str(payload.get("model") or "")
            spec = get_model_spec(model_id) if model_id else None
            repo_id = spec.repo_id if spec else str(payload.get("repo_id") or "")
            source = spec.source if spec else str(payload.get("source") or "")
            files = tuple(str(item) for item in payload.get("files") or ())
            download_repository(repo_id, source, ProcessRunner(), files=files)
            for dependency in spec.dependencies if spec else ():
                download_repository(
                    dependency.repo_id,
                    dependency.source,
                    ProcessRunner(),
                    files=dependency.files,
                )
            if spec is not None:
                from ..model_management.backend import (
                    resolve_huggingface_model,
                    resolve_modelscope_model,
                )

                def resolve(item_source: str, item_repo: str) -> Path | None:
                    return (
                        resolve_modelscope_model(item_repo)
                        if item_source == "modelscope"
                        else resolve_huggingface_model(item_repo)
                    )

                model_path = resolve(spec.source, spec.repo_id)
                dependency_paths: dict[str, str] = {}
                for dependency in spec.dependencies:
                    dependency_root = resolve(dependency.source, dependency.repo_id)
                    dependency_path = dependency_root
                    if dependency_root is not None and dependency.files:
                        dependency_path = next(
                            (
                                dependency_root / name
                                for name in dependency.files
                                if (dependency_root / name).is_file()
                            ),
                            None,
                        )
                    dependency_paths[dependency.id] = str(dependency_path or "")
                active.known_status[spec.id] = active.status(
                    spec,
                    str(model_path or ""),
                    dependency_paths,
                )
            return {"status": "downloaded", "repo_id": repo_id, "source": source}
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/models/{model_id}")
    def delete(model_id: str) -> dict[str, str]:
        if active.spec and active.spec.id == model_id:
            active.unload()
        try:
            spec = get_model_spec(model_id)
            path = delete_repository(spec.repo_id, spec.source)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        active.known_status.pop(model_id, None)
        return {"status": "deleted", "model": model_id, "path": str(path)}

    @app.post("/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(""),
        language: str = Form("English"),
    ) -> dict[str, Any]:
        if model and active.spec and model != active.spec.id:
            raise HTTPException(status_code=409, detail="Requested model is not loaded")
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as output:
                output.write(await file.read())
                temporary = Path(output.name)
            return active.transcribe(temporary, language)
        except (OSError, RuntimeError, ValueError) as exc:
            status = 422 if exc.__class__.__name__ == "TimelineError" else 500
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @app.post("/audio/speech")
    def speech(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        requested_model = str(payload.get("model") or "")
        if requested_model and active.spec and requested_model != active.spec.id:
            raise HTTPException(status_code=409, detail="Requested model is not loaded")
        texts = payload.get("texts")
        if texts is None:
            texts = [payload.get("input", "")]
        if not isinstance(texts, list) or not texts or any(not str(text).strip() for text in texts):
            raise HTTPException(status_code=400, detail="Speech input is empty")
        try:
            outputs = active.synthesize(
                [str(text) for text in texts],
                str(payload.get("language") or "Chinese"),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        encoded: list[str | None] = []
        errors: list[str | None] = []
        for output in outputs:
            if isinstance(output, RuntimeError):
                encoded.append(None)
                errors.append(str(output))
            else:
                encoded.append(base64.b64encode(output).decode("ascii"))
                errors.append(None)
        if payload.get("texts") is None and errors[0]:
            raise HTTPException(status_code=500, detail=errors[0])
        response: dict[str, Any] = {"audio_base64_list": encoded, "errors": errors}
        if payload.get("texts") is None:
            response["audio_base64"] = encoded[0]
        return response

    return app
