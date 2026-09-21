from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ...subtitles import read_srt
from .base import RuntimeAdapter


def _language_code(language: str) -> str:
    return {
        "english": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "cantonese": "yue",
        "japanese": "ja",
        "korean": "ko",
        "german": "de",
        "spanish": "es",
        "french": "fr",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
    }.get(language.strip().casefold(), language.strip().casefold() or "en")


class _GGUFAdapter(RuntimeAdapter):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        self.model_path = Path()
        self.dependencies: dict[str, Path] = {}
        self.options: dict[str, Any] = {}
        self.executable = Path()

    def load(self, model_path: Path, dependencies: dict[str, Path], options: dict[str, Any]) -> None:
        executable = Path(str(options.get("executable") or ""))
        if not executable.is_file():
            raise RuntimeError("CrispASR runtime is not installed")
        self.model_path = model_path
        self.dependencies = dependencies
        self.options = options
        self.executable = executable
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False
        self.dependencies = {}

    def _run(self, command: list[str | Path]) -> None:
        result = subprocess.run(
            [str(item) for item in command],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "CrispASR failed")[-2000:])


class QwenGGUFASRAdapter(_GGUFAdapter):
    def transcribe(self, audio_path: Path, language: str) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("ASR model is not loaded")
        aligner = next((self.dependencies[d.id] for d in self.spec.dependencies if "timestamps" in d.capabilities and d.id in self.dependencies), None)
        vad = next((self.dependencies[d.id] for d in self.spec.dependencies if d.manifest_field == "vad_path" and d.id in self.dependencies), None)
        if aligner is None or vad is None:
            raise RuntimeError("Qwen3-ASR GGUF companion dependencies are incomplete")
        with tempfile.TemporaryDirectory(prefix="scip-speech-asr-") as temp:
            prefix = Path(temp) / "recognized"
            self._run([
                self.executable, "--backend", "qwen3", "-m", self.model_path,
                "-f", audio_path, "-l", _language_code(language), "--vad",
                "-vm", vad, "-am", aligner, "--split-on-punct",
                "--strict-pipeline", "-osrt", "-of", prefix,
            ])
            srt = prefix.with_suffix(".srt")
            if not srt.is_file():
                candidates = list(Path(temp).glob("*.srt"))
                if not candidates:
                    raise RuntimeError("CrispASR did not produce SRT output")
                srt = candidates[0]
            cues = read_srt(srt)
        return {
            "text": " ".join(cue.text for cue in cues),
            "language": language,
            "segments": [
                {"text": cue.text, "start": cue.start_ms / 1000, "end": cue.end_ms / 1000}
                for cue in cues
            ],
        }


class QwenGGUFTTSAdapter(_GGUFAdapter):
    def synthesize(self, texts: list[str], language: str) -> list[bytes | RuntimeError]:
        if not self.loaded:
            raise RuntimeError("TTS model is not loaded")
        codec = next((self.dependencies[d.id] for d in self.spec.dependencies if d.manifest_field == "codec_path" and d.id in self.dependencies), None)
        if codec is None:
            raise RuntimeError("Qwen3-TTS GGUF codec dependency is incomplete")
        outputs: list[bytes | RuntimeError] = []
        for text in texts:
            try:
                with tempfile.TemporaryDirectory(prefix="scip-speech-tts-") as temp:
                    output = Path(temp) / "speech.wav"
                    command: list[str | Path] = [
                        self.executable,
                        "--backend",
                        "qwen3-tts" if self.spec.variant == "base" else "qwen3-tts-customvoice",
                        "-m", self.model_path, "--codec-model", codec, "--voice",
                    ]
                    if self.spec.variant == "base":
                        command.extend([
                            str(self.options.get("reference_audio") or ""),
                            "--ref-text", str(self.options.get("reference_text") or ""),
                        ])
                    else:
                        command.append(str(self.options.get("speaker") or "Vivian"))
                    command.extend(["-l", _language_code(language), "--tts", text, "--tts-output", output])
                    self._run(command)
                    outputs.append(output.read_bytes())
            except (OSError, RuntimeError) as exc:
                outputs.append(RuntimeError(str(exc)))
        return outputs
