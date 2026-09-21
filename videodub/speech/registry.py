from __future__ import annotations

from .models import CapabilityDependency, ModelSpec


MLX_ALIGNER = CapabilityDependency(
    "qwen3-forced-aligner-mlx",
    "Qwen3-ForcedAligner 0.6B 8bit",
    "huggingface",
    "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
    ("timestamps",),
)
TRANSFORMERS_ALIGNER = CapabilityDependency(
    "qwen3-forced-aligner-transformers",
    "Qwen3-ForcedAligner 0.6B",
    "modelscope",
    "Qwen/Qwen3-ForcedAligner-0.6B",
    ("timestamps",),
)
GGUF_ALIGNER = CapabilityDependency(
    "qwen3-forced-aligner-gguf",
    "Qwen3-ForcedAligner 0.6B Q8_0",
    "huggingface",
    "cstr/qwen3-forced-aligner-0.6b-GGUF",
    ("timestamps",),
    ("qwen3-forced-aligner-0.6b-q8_0.gguf",),
)
GGUF_VAD = CapabilityDependency(
    "silero-vad-gguf",
    "Silero VAD",
    "huggingface",
    "ggml-org/whisper-vad",
    ("transcription",),
    ("ggml-silero-v6.2.0.bin",),
    "vad_path",
)
GGUF_CODEC = CapabilityDependency(
    "qwen3-tts-codec-gguf",
    "Qwen3-TTS 12Hz codec",
    "huggingface",
    "cstr/qwen3-tts-tokenizer-12hz-GGUF",
    ("text_to_speech",),
    ("qwen3-tts-tokenizer-12hz.gguf",),
    "codec_path",
)


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        "qwen3-asr-mlx-0.6b-8bit",
        "Qwen3-ASR 0.6B 8bit",
        "asr",
        "qwen3-asr",
        "huggingface",
        "mlx-community/Qwen3-ASR-0.6B-8bit",
        "qwen3-asr-mlx",
        "mlx",
        ("transcription", "timestamps"),
        (MLX_ALIGNER,),
    ),
    ModelSpec(
        "qwen3-asr-transformers-0.6b",
        "Qwen3-ASR 0.6B",
        "asr",
        "qwen3-asr",
        "modelscope",
        "Qwen/Qwen3-ASR-0.6B",
        "qwen3-asr-transformers",
        "transformers",
        ("transcription", "timestamps"),
        (TRANSFORMERS_ALIGNER,),
    ),
    ModelSpec(
        "qwen3-asr-gguf-0.6b",
        "Qwen3-ASR 0.6B GGUF",
        "asr",
        "qwen3-asr",
        "huggingface",
        "cstr/qwen3-asr-0.6b-GGUF",
        "qwen3-asr-gguf",
        "gguf",
        ("transcription", "timestamps"),
        (GGUF_ALIGNER, GGUF_VAD),
    ),
    ModelSpec(
        "qwen3-tts-mlx-base-0.6b-8bit",
        "Qwen3-TTS 0.6B Base 8bit",
        "tts",
        "qwen3-tts",
        "huggingface",
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "qwen3-tts-mlx",
        "mlx",
        ("text_to_speech", "voice_clone", "zero_shot_voice"),
        variant="base",
    ),
    ModelSpec(
        "qwen3-tts-mlx-customvoice-0.6b-8bit",
        "Qwen3-TTS 0.6B CustomVoice 8bit",
        "tts",
        "qwen3-tts",
        "huggingface",
        "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        "qwen3-tts-mlx",
        "mlx",
        ("text_to_speech",),
        variant="custom_voice",
    ),
    ModelSpec(
        "qwen3-tts-transformers-base-0.6b",
        "Qwen3-TTS 0.6B Base",
        "tts",
        "qwen3-tts",
        "modelscope",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "qwen3-tts-transformers",
        "transformers",
        ("text_to_speech", "voice_clone", "zero_shot_voice"),
        variant="base",
    ),
    ModelSpec(
        "qwen3-tts-transformers-customvoice-0.6b",
        "Qwen3-TTS 0.6B CustomVoice",
        "tts",
        "qwen3-tts",
        "modelscope",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "qwen3-tts-transformers",
        "transformers",
        ("text_to_speech",),
        variant="custom_voice",
    ),
    ModelSpec(
        "qwen3-tts-gguf-base-0.6b",
        "Qwen3-TTS 0.6B Base GGUF",
        "tts",
        "qwen3-tts",
        "huggingface",
        "cstr/qwen3-tts-0.6b-base-GGUF",
        "qwen3-tts-gguf",
        "gguf",
        ("text_to_speech", "voice_clone", "zero_shot_voice"),
        (GGUF_CODEC,),
        "base",
    ),
    ModelSpec(
        "qwen3-tts-gguf-customvoice-0.6b",
        "Qwen3-TTS 0.6B CustomVoice GGUF",
        "tts",
        "qwen3-tts",
        "huggingface",
        "cstr/qwen3-tts-0.6b-customvoice-GGUF",
        "qwen3-tts-gguf",
        "gguf",
        ("text_to_speech",),
        (GGUF_CODEC,),
        "custom_voice",
    ),
)

_BY_ID = {spec.id.casefold(): spec for spec in MODEL_REGISTRY}
_BY_REPOSITORY = {spec.repo_id.casefold(): spec for spec in MODEL_REGISTRY}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return _BY_ID[model_id.casefold()]
    except KeyError as exc:
        raise KeyError(f"Unsupported speech model: {model_id}") from exc


def find_model_spec(repo_id: str) -> ModelSpec | None:
    return _BY_REPOSITORY.get(repo_id.strip().casefold())
