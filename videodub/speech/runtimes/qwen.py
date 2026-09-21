from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from .base import RuntimeAdapter


class QwenASRAdapter(RuntimeAdapter):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        self.model: Any = None
        self.aligner: Any = None

    def load(self, model_path: Path, dependencies: dict[str, Path], options: dict[str, Any]) -> None:
        from ... import qwen_service

        aligner_path = next(
            (dependencies[item.id] for item in self.spec.dependencies if "timestamps" in item.capabilities and item.id in dependencies),
            None,
        )
        if self.spec.engine == "mlx":
            from mlx_audio.stt.utils import load_model

            self.model = load_model(str(model_path))
            self.aligner = load_model(str(aligner_path)) if aligner_path else None
        else:
            from qwen_asr import Qwen3ASRModel

            torch_options = qwen_service._torch_options()
            aligner_options = (
                {
                    "forced_aligner": str(aligner_path),
                    "forced_aligner_kwargs": torch_options,
                }
                if aligner_path
                else {}
            )
            self.model = Qwen3ASRModel.from_pretrained(
                str(model_path),
                max_inference_batch_size=4,
                max_new_tokens=1024,
                **aligner_options,
                **torch_options,
            )
            self.aligner = str(aligner_path) if aligner_path else None
        self.loaded = True

    def unload(self) -> None:
        self.model = None
        self.aligner = None
        self.loaded = False

    def transcribe(self, audio_path: Path, language: str) -> dict[str, Any]:
        from ... import qwen_service

        if not self.loaded:
            raise RuntimeError("ASR model is not loaded")
        if self.spec.engine == "mlx":
            if self.aligner is None:
                texts = []
                for audio, _offset in qwen_service._mlx_audio_chunks(audio_path):
                    result = self.model.generate(audio)
                    text = result if isinstance(result, str) else getattr(result, "text", "")
                    texts.append(str(text or ""))
                return {"text": " ".join(texts).strip(), "language": language, "segments": []}
            result = qwen_service._transcribe_mlx(
                self.model,
                self.aligner,
                audio_path,
                language,
            )
            return {
                "text": str(result.get("text") or ""),
                "language": language,
                "segments": qwen_service._mlx_segments(result),
            }
        result = self.model.transcribe(
            audio=str(audio_path),
            language=language or None,
            return_time_stamps=self.aligner is not None,
        )[0]
        return {
            "text": str(result.text),
            "language": str(result.language),
            "segments": qwen_service._timestamp_segments(result) if self.aligner is not None else [],
        }


class QwenTTSAdapter(RuntimeAdapter):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        self.model: Any = None
        self.args: Namespace | None = None
        self.request_count = 0

    def load(self, model_path: Path, dependencies: dict[str, Path], options: dict[str, Any]) -> None:
        from ... import qwen_service

        if self.spec.engine == "mlx":
            from mlx_audio.tts.utils import load_model

            self.model = load_model(str(model_path))
            backend = "mlx"
        else:
            from qwen_tts import Qwen3TTSModel

            self.model = Qwen3TTSModel.from_pretrained(
                str(model_path),
                **qwen_service._torch_options(),
            )
            backend = "hf"
        self.args = Namespace(
            model=str(model_path),
            backend=backend,
            variant=self.spec.variant or "custom_voice",
            speaker=str(options.get("speaker") or "Vivian"),
            reference_audio=str(options.get("reference_audio") or ""),
            reference_text=str(options.get("reference_text") or ""),
        )
        self.request_count = 0
        self.loaded = True

    def unload(self) -> None:
        self.model = None
        self.args = None
        self.request_count = 0
        self.loaded = False

    def synthesize(self, texts: list[str], language: str) -> list[bytes | RuntimeError]:
        from ... import qwen_service

        if not self.loaded or self.args is None:
            raise RuntimeError("TTS model is not loaded")
        request_start = self.request_count + 1
        self.request_count += len(texts)
        generated = qwen_service._generate_tts_batch(
            self.model,
            self.args,
            texts,
            language,
            request_start=request_start,
        )
        outputs: list[bytes | RuntimeError] = []
        for value in generated:
            if isinstance(value, RuntimeError):
                outputs.append(value)
                continue
            try:
                import base64

                outputs.append(base64.b64decode(qwen_service._audio_bytes(*value)))
            except (RuntimeError, ValueError) as exc:
                outputs.append(RuntimeError(str(exc)))
        return outputs
