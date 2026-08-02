"""Private, task-scoped Qwen ASR/TTS services.

The desktop app starts this module with the Python interpreter belonging to an
installed model, waits for /health, runs one task, and always terminates it.
"""

import argparse
import base64
import io
import tempfile
import unicodedata
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
    return _alignment_to_segments(
        stamps,
        str(getattr(result, "text", "") or ""),
        0.0,
    )


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


def _mlx_audio_chunks(audio_path: Path) -> list[tuple[Any, float]]:
    import numpy as np
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
        split_audio_into_chunks,
    )
    from mlx_audio.stt.utils import load_audio

    return split_audio_into_chunks(
        np.array(load_audio(str(audio_path))),
        sr=16000,
        chunk_duration=240.0,
        min_chunk_duration=1.0,
    )


def _alignment_character(character: str) -> bool:
    return character == "'" or unicodedata.category(character)[:1] in {"L", "N"}


def _aligned_display_tokens(transcript: str, items: list[Any]) -> list[str]:
    normalized: list[str] = []
    source_positions: list[int] = []
    for position, character in enumerate(transcript):
        if not _alignment_character(character):
            continue
        for folded in character.casefold():
            normalized.append(folded)
            source_positions.append(position)
    searchable = "".join(normalized)
    starts: list[int] = []
    cursor = 0
    for item in items:
        needle = "".join(
            character.casefold()
            for character in str(getattr(item, "text", ""))
            if _alignment_character(character)
        )
        found = searchable.find(needle, cursor) if needle else -1
        if found < 0:
            return [
                str(getattr(value, "text", "")).strip() + " "
                for value in items
            ]
        starts.append(source_positions[found])
        cursor = found + len(needle)
    return [
        transcript[
            0 if index == 0 else start : starts[index + 1]
            if index + 1 < len(starts)
            else len(transcript)
        ]
        for index, start in enumerate(starts)
    ]


def _ends_with(text: str, punctuation: str) -> bool:
    stripped = text.rstrip(" \t\r\n\"'”’)]}）】")
    return stripped.endswith(tuple(punctuation))


def _alignment_to_segments(
    alignment: Any,
    transcript: str,
    offset_seconds: float,
) -> list[dict[str, Any]]:
    items = list(getattr(alignment, "items", []) or [])
    if not items:
        return []
    display_tokens = _aligned_display_tokens(transcript, items)
    segments: list[dict[str, Any]] = []
    first = 0
    for index, item in enumerate(items):
        text = "".join(display_tokens[first : index + 1]).strip()
        next_item = items[index + 1] if index + 1 < len(items) else None
        pause = (
            float(getattr(next_item, "start_time", 0.0))
            - float(getattr(item, "end_time", 0.0))
            if next_item is not None
            else 0.0
        )
        text_limit = (
            28 if any("\u3400" <= character <= "\u9fff" for character in text) else 72
        )
        start_time = float(getattr(items[first], "start_time", 0.0))
        current_end = float(getattr(item, "end_time", 0.0))
        duration = current_end - start_time
        natural_long_sentence_break = (
            duration >= 1.75
            and (
                _ends_with(text, ",:;，：；、")
                or pause >= 0.25
                or (len(text) >= text_limit and pause >= 0.12)
                or (duration >= 4.8 and pause >= 0.12)
            )
        )
        boundary = (
            _ends_with(text, ".!?。！？…")
            or pause >= 0.65
            or natural_long_sentence_break
            or next_item is None
        )
        if not boundary:
            continue
        start_item = items[first]
        segments.append(
            {
                "text": text,
                "start": round(
                    offset_seconds
                    + float(getattr(start_item, "start_time", 0.0)),
                    3,
                ),
                "end": round(
                    offset_seconds + float(getattr(item, "end_time", 0.0)),
                    3,
                ),
            }
        )
        first = index + 1
    return segments


def _transcribe_mlx(
    model: Any,
    aligner: Any,
    audio_path: Path,
    language: str,
) -> dict[str, Any]:
    texts: list[str] = []
    segments: list[dict[str, Any]] = []
    for audio_chunk, offset_seconds in _mlx_audio_chunks(audio_path):
        result = model.generate(audio_chunk, language=language or None)
        text = str(
            (
                result.get("text")
                if isinstance(result, dict)
                else getattr(result, "text", "")
            )
            or ""
        ).strip()
        if not text:
            continue
        alignment = aligner.generate(audio_chunk, text, language)
        aligned = _alignment_to_segments(alignment, text, offset_seconds)
        if not aligned:
            raise RuntimeError("MLX Forced Aligner 没有返回逐词时间戳")
        texts.append(text)
        segments.extend(aligned)
    return {"text": " ".join(texts), "segments": segments}


def create_asr_app(args: argparse.Namespace) -> Any:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile

    state: dict[str, Any] = {"model": None, "aligner": None}

    @asynccontextmanager
    async def lifespan(_app: Any):
        if args.backend == "mlx":
            from mlx_audio.stt.utils import load_model

            state["model"] = load_model(args.model)
            if not args.aligner:
                raise RuntimeError("Mac ASR 缺少 MLX Forced Aligner 模型")
            state["aligner"] = load_model(args.aligner)
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
        state["aligner"] = None

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
                result = _transcribe_mlx(
                    model,
                    state["aligner"],
                    temp_path,
                    language,
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
        language: str = "Chinese"

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
                    language=request.language,
                )
            else:
                results = model.generate_custom_voice(
                    text=request.text,
                    speaker=args.speaker,
                    language=request.language,
                )
            result = next(iter(results)) if hasattr(results, "__iter__") else results
            audio = getattr(result, "audio", result)
            sample_rate = int(getattr(result, "sample_rate", 24000))
        else:
            if args.variant == "base":
                wavs, sample_rate = model.generate_voice_clone(
                    text=request.text,
                    language=request.language,
                    ref_audio=args.reference_audio,
                    ref_text=args.reference_text,
                )
            else:
                wavs, sample_rate = model.generate_custom_voice(
                    text=request.text,
                    language=request.language,
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
