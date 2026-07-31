import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videodub.config import AppConfig
from videodub.model_manager import InstalledModel
from videodub.qwen_speech import (
    _segments_to_cues,
    _validate_crispasr_asr_model,
    synthesize_qwen,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.command: list[str] = []

    def run(self, command, *, quiet: bool = False) -> None:
        self.command = [str(item) for item in command]


class QwenSpeechTests(unittest.TestCase):
    def test_asr_segments_become_srt_cues(self) -> None:
        cues = _segments_to_cues(
            [
                {"text": "Hello", "start": 0.25, "end": 1.5},
                {"text": "world", "start": 1.5, "end": 2.0},
            ],
            "",
            3.0,
        )
        self.assertEqual([cue.index for cue in cues], [1, 2])
        self.assertEqual(cues[0].start_ms, 250)
        self.assertEqual(cues[1].end_ms, 2000)

    def test_asr_text_without_segments_uses_video_duration(self) -> None:
        cues = _segments_to_cues(None, "Fallback", 4.2)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "Fallback")
        self.assertEqual(cues[0].end_ms, 4200)

    def test_old_gguf_conversion_is_rejected_before_crispasr_crashes(self) -> None:
        installed = InstalledModel(
            "asr",
            "gguf",
            "handy-computer/Qwen3-ASR-0.6B-gguf",
            "model.gguf",
        )
        with patch(
            "videodub.qwen_speech.read_installed_model",
            return_value=installed,
        ):
            with self.assertRaisesRegex(RuntimeError, "cstr/qwen3-asr-0.6b-GGUF"):
                _validate_crispasr_asr_model(Path("model.gguf"))

    def test_gguf_tts_uses_custom_voice_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.gguf"
            codec = root / "codec.gguf"
            output = root / "output.wav"
            model.touch()
            codec.touch()
            installed = InstalledModel(
                "tts",
                "gguf",
                "owner/customvoice",
                str(root),
                str(codec),
            )
            runner = RecordingRunner()
            config = AppConfig(
                tts_backend="gguf",
                tts_model_path=str(root),
                tts_speaker="Vivian",
            )
            with (
                patch(
                    "videodub.qwen_speech.crispasr_executable",
                    return_value=root / "crispasr.exe",
                ),
                patch(
                    "videodub.qwen_speech.read_installed_model",
                    return_value=installed,
                ),
            ):
                synthesize_qwen(config, "你好", output, runner)

        self.assertIn("qwen3-tts-customvoice", runner.command)
        self.assertEqual(
            runner.command[runner.command.index("--voice") + 1],
            "Vivian",
        )
        self.assertNotIn("--ref-text", runner.command)

    def test_gguf_base_tts_uses_reference_audio_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.gguf"
            codec = root / "codec.gguf"
            reference = root / "reference.wav"
            output = root / "output.wav"
            model.touch()
            codec.touch()
            reference.touch()
            installed = InstalledModel(
                "tts",
                "gguf",
                "owner/base",
                str(root),
                str(codec),
                variant="base",
            )
            runner = RecordingRunner()
            config = AppConfig(
                tts_backend="gguf",
                tts_model_path=str(root),
                tts_reference_audio=str(reference),
                tts_reference_text="参考文本",
            )
            with (
                patch(
                    "videodub.qwen_speech.crispasr_executable",
                    return_value=root / "crispasr.exe",
                ),
                patch(
                    "videodub.qwen_speech.read_installed_model",
                    return_value=installed,
                ),
            ):
                synthesize_qwen(config, "你好", output, runner)

        self.assertIn("qwen3-tts", runner.command)
        self.assertNotIn("qwen3-tts-customvoice", runner.command)
        self.assertEqual(
            runner.command[runner.command.index("--voice") + 1],
            str(reference),
        )
        self.assertEqual(
            runner.command[runner.command.index("--ref-text") + 1],
            "参考文本",
        )


if __name__ == "__main__":
    unittest.main()
