"""Optional Qwen3-ASR/TTS HTTP services for Windows and Linux.

This module is intentionally not imported by the desktop application unless one
of its console entry points is launched. The large model runtime therefore stays
out of the default installation.
"""

import argparse
import base64
import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


ASR_MODEL = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
ALIGNER_MODEL = os.environ.get(
    "QWEN_ALIGNER_MODEL",
    "Qwen/Qwen3-ForcedAligner-0.6B",
)
TTS_MODEL = os.environ.get(
    "QWEN_TTS_MODEL",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
)


def _torch_options() -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        return {"device_map": "cuda:0", "dtype": torch.bfloat16}
    return {"device_map": "cpu", "dtype": torch.float32}


def _timestamp_segments(result: Any) -> list[dict[str, Any]]:
    stamps = getattr(result, "time_stamps", None)
    if isinstance(stamps, list) and stamps:
        stamps = stamps[0]
    items = list(getattr(stamps, "items", []) or [])
    segments: list[dict[str, Any]] = []
    current: list[Any] = []
    for item in items:
        current.append(item)
        text = "".join(str(getattr(part, "text", "")) for part in current)
        start = float(getattr(current[0], "start_time", 0.0))
        end = float(getattr(current[-1], "end_time", start))
        if end - start >= 7.0 or text.rstrip().endswith((".", "!", "?", "。", "！", "？")):
            segments.append({"text": text.strip(), "start": start, "end": end})
            current = []
    if current:
        start = float(getattr(current[0], "start_time", 0.0))
        end = float(getattr(current[-1], "end_time", start))
        text = "".join(str(getattr(part, "text", "")) for part in current)
        segments.append({"text": text.strip(), "start": start, "end": end})
    return [item for item in segments if item["text"]]


def create_asr_app() -> Any:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile

    state: dict[str, Any] = {"model": None}

    @asynccontextmanager
    async def lifespan(_app: Any):
        from qwen_asr import Qwen3ASRModel

        options = _torch_options()
        state["model"] = Qwen3ASRModel.from_pretrained(
            ASR_MODEL,
            forced_aligner=ALIGNER_MODEL,
            forced_aligner_kwargs=options,
            max_inference_batch_size=4,
            max_new_tokens=1024,
            **options,
        )
        yield
        state["model"] = None

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {
            "status": "ok",
            "model": ASR_MODEL,
            "backend": "transformers",
            "type": "asr",
        }

    @app.post("/v1/asr")
    async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form("English"),
    ) -> dict[str, Any]:
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(await audio.read())
                temp_path = Path(handle.name)
            result = model.transcribe(
                audio=str(temp_path),
                language=language or None,
                return_time_stamps=True,
            )[0]
            return {
                "text": str(result.text),
                "language": str(result.language),
                "model": ASR_MODEL,
                "segments": _timestamp_segments(result),
            }
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    return app


def create_tts_app() -> Any:
    import numpy as np
    import soundfile as sf
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    state: dict[str, Any] = {"model": None}

    @asynccontextmanager
    async def lifespan(_app: Any):
        from qwen_tts import Qwen3TTSModel

        state["model"] = Qwen3TTSModel.from_pretrained(
            TTS_MODEL,
            **_torch_options(),
        )
        yield
        state["model"] = None

    app = FastAPI(lifespan=lifespan)

    class TTSRequest(BaseModel):
        text: str

    @app.get("/health")
    def health() -> dict[str, str]:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {
            "status": "ok",
            "model": TTS_MODEL,
            "backend": "transformers",
            "type": "tts",
        }

    @app.post("/v1/tts")
    def synthesize(request: TTSRequest) -> dict[str, Any]:
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        wavs, sample_rate = model.generate_custom_voice(
            text=request.text,
            language="Chinese",
            speaker="Vivian",
        )
        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(wavs[0]), sample_rate, format="WAV")
        return {
            "audio_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "model": TTS_MODEL,
        }

    return app


def _serve(service_type: str, default_port: int) -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()
    app = create_asr_app() if service_type == "asr" else create_tts_app()
    uvicorn.run(app, host=args.host, port=args.port)


def main_asr() -> None:
    _serve("asr", 9956)


def main_tts() -> None:
    _serve("tts", 9955)
