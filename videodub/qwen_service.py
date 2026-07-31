"""Private, task-scoped Qwen ASR/TTS services.

The desktop app starts this module with the Python interpreter belonging to an
installed model, waits for /health, runs one task, and always terminates it.
"""

import argparse
import base64
import io
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


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
        segments.append(
            {
                "text": "".join(str(getattr(part, "text", "")) for part in current).strip(),
                "start": float(getattr(current[0], "start_time", 0.0)),
                "end": float(getattr(current[-1], "end_time", 0.0)),
            }
        )
    return [item for item in segments if item["text"]]


def _mlx_segments(result: Any) -> list[dict[str, Any]]:
    raw = getattr(result, "segments", None)
    if raw is None and isinstance(result, dict):
        raw = result.get("segments")
    segments: list[dict[str, Any]] = []
    for item in raw or []:
        value = item if isinstance(item, dict) else vars(item)
        text = str(value.get("text") or "").strip()
        if text:
            segments.append(
                {
                    "text": text,
                    "start": float(value.get("start") or 0),
                    "end": float(value.get("end") or 0),
                }
            )
    return segments


def create_asr_app(args: argparse.Namespace) -> Any:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile

    state: dict[str, Any] = {"model": None}

    @asynccontextmanager
    async def lifespan(_app: Any):
        if args.backend == "mlx":
            from mlx_audio.stt.utils import load_model

            state["model"] = load_model(args.model)
        else:
            from qwen_asr import Qwen3ASRModel

            options = _torch_options()
            state["model"] = Qwen3ASRModel.from_pretrained(
                args.model,
                forced_aligner=args.aligner,
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
        return {"status": "ok", "model": args.model, "backend": args.backend, "type": "asr"}

    @app.post("/v1/asr")
    async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form("English"),
    ) -> dict[str, Any]:
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(await audio.read())
                temp_path = Path(handle.name)
            if args.backend == "mlx":
                model = state["model"]
                try:
                    from mlx_audio.stt.generate import generate_transcription

                    result = generate_transcription(
                        model=model,
                        audio_path=str(temp_path),
                        output_path=str(temp_path.with_suffix(".txt")),
                        format="txt",
                        verbose=False,
                    )
                except (ImportError, TypeError):
                    result = model.generate(
                        str(temp_path),
                        language=language or None,
                    )
                text = str(
                    (result.get("text") if isinstance(result, dict) else getattr(result, "text", ""))
                    or ""
                )
                return {"text": text, "language": language, "model": args.model, "segments": _mlx_segments(result)}
            result = state["model"].transcribe(
                audio=str(temp_path),
                language=language or None,
                return_time_stamps=True,
            )[0]
            return {
                "text": str(result.text),
                "language": str(result.language),
                "model": args.model,
                "segments": _timestamp_segments(result),
            }
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    return app


def _audio_bytes(audio: Any, sample_rate: int) -> str:
    import numpy as np
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio).squeeze(), sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_tts_app(args: argparse.Namespace) -> Any:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    state: dict[str, Any] = {"model": None}

    @asynccontextmanager
    async def lifespan(_app: Any):
        if args.backend == "mlx":
            from mlx_audio.tts.utils import load_model

            state["model"] = load_model(args.model)
        else:
            from qwen_tts import Qwen3TTSModel

            state["model"] = Qwen3TTSModel.from_pretrained(
                args.model,
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
        return {"status": "ok", "model": args.model, "backend": args.backend, "type": "tts"}

    @app.post("/v1/tts")
    def synthesize(request: TTSRequest) -> dict[str, Any]:
        model = state["model"]
        if args.backend == "mlx":
            if args.variant == "base":
                results = model.generate(
                    text=request.text,
                    ref_audio=args.reference_audio,
                    ref_text=args.reference_text,
                )
            else:
                results = model.generate_custom_voice(
                    text=request.text,
                    speaker=args.speaker,
                    language="Chinese",
                )
            result = next(iter(results)) if hasattr(results, "__iter__") else results
            audio = getattr(result, "audio", result)
            sample_rate = int(getattr(result, "sample_rate", 24000))
        else:
            if args.variant == "base":
                wavs, sample_rate = model.generate_voice_clone(
                    text=request.text,
                    language="Chinese",
                    ref_audio=args.reference_audio,
                    ref_text=args.reference_text,
                )
            else:
                wavs, sample_rate = model.generate_custom_voice(
                    text=request.text,
                    language="Chinese",
                    speaker=args.speaker,
                )
            audio = wavs[0]
        return {"audio_base64": _audio_bytes(audio, sample_rate), "model": args.model}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("asr", "tts"))
    parser.add_argument("--backend", choices=("hf", "mlx"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--aligner", default="")
    parser.add_argument("--variant", choices=("base", "custom_voice"), default="custom_voice")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--reference-audio", default="")
    parser.add_argument("--reference-text", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    import uvicorn

    app = create_asr_app(args) if args.service == "asr" else create_tts_app(args)
    uvicorn.run(app, host=args.host, port=args.port or (9956 if args.service == "asr" else 9955))


if __name__ == "__main__":
    main()
