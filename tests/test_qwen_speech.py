import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videodub.config import AppConfig
from videodub.model_manager import InstalledModel
from videodub.qwen_speech import (
    _crispasr_asr_runtime_options,
    _crispasr_language_code,
    _segments_to_cues,
    _validate_crispasr_asr_model,
    resolve_tts_reference,
    synthesize_qwen,
)
from videodub.subtitles import Cue


class RecordingRunner:
    def __init__(self) -> None:
        self.command: list[str] = []

    def run(self, command, *, quiet: bool = False) -> None:
        self.command = [str(item) for item in command]


class QwenSpeechTests(unittest.TestCase):
    def test_crispasr_uses_explicit_language_codes(self) -> None:
        self.assertEqual(_crispasr_language_code("English"), "en")
        self.assertEqual(_crispasr_language_code("Chinese"), "zh")
        self.assertEqual(_crispasr_language_code("Japanese"), "ja")
        self.assertEqual(_crispasr_language_code("Russian"), "ru")
        self.assertEqual(_crispasr_language_code("yue"), "yue")

    def test_crispasr_asr_does_not_download_runtime_helpers(self) -> None:
        options = _crispasr_asr_runtime_options(
            "English",
            Path("vad.bin"),
            Path("aligner.gguf"),
        )
        self.assertEqual(
            options,
            [
                "-l",
                "en",
                "--vad",
                "-vm",
                "vad.bin",
                "-am",
                "aligner.gguf",
                "--split-on-punct",
                "--strict-pipeline",
            ],
        )

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
        self.assertEqual(cues[0].end_ms, 1500)
        self.assertEqual(cues[1].start_ms, 1500)
        self.assertEqual(cues[1].end_ms, 2000)
        self.assertEqual([cue.text for cue in cues], ["Hello", "world"])

    def test_asr_preserves_short_word_aligned_cue(self) -> None:
        cues = _segments_to_cues(
            [
                {
                    "text": "you know? And it's just one of those things whenever it gets interrupted",
                    "start": 59.52,
                    "end": 62.24,
                },
                {
                    "text": "from the power source, it has to reboot and it just totally wipes out the",
                    "start": 62.24,
                    "end": 65.76,
                },
                {"text": "history.", "start": 65.76, "end": 66.32},
            ],
            "",
            66.32,
        )

        self.assertEqual(
            [(cue.start_ms, cue.end_ms, cue.text) for cue in cues],
            [
                (
                    59_520,
                    62_240,
                    "you know? And it's just one of those things whenever it gets interrupted",
                ),
                (
                    62_240,
                    65_760,
                    "from the power source, it has to reboot and it just totally wipes out the",
                ),
                (65_760, 66_320, "history."),
            ],
        )

    def test_asr_does_not_invent_boundaries_for_long_cues(self) -> None:
        cues = _segments_to_cues(
            [
                {
                    "text": "one two three four five six seven eight",
                    "start": 0,
                    "end": 8,
                }
            ],
            "",
            8,
        )

        self.assertEqual(
            cues,
            [Cue(1, 0, 8000, "one two three four five six seven eight")],
        )

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
            reference_text = root / "reference.anything"
            output = root / "output.wav"
            model.touch()
            codec.touch()
            reference.touch()
            reference_text.write_text("参考文本", encoding="utf-8")
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
                tts_use_custom_voice=True,
                tts_reference_audio=str(reference),
                tts_reference_text_file=str(reference_text),
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
            str(reference.resolve()),
        )
        self.assertEqual(
            runner.command[runner.command.index("--ref-text") + 1],
            "参考文本",
        )
        self.assertEqual(runner.command[runner.command.index("-l") + 1], "zh")

    def test_custom_reference_text_file_has_no_extension_restriction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "voice.wav"
            text_file = root / "voice.data"
            audio.touch()
            text_file.write_text("实际读取的文本", encoding="utf-8")
            config = AppConfig(
                tts_use_custom_voice=True,
                tts_reference_audio=str(audio),
                tts_reference_text_file=str(text_file),
            )
            resolved_audio, resolved_text = resolve_tts_reference(config)
            self.assertEqual(Path(resolved_audio), audio.resolve())
            self.assertEqual(resolved_text, "实际读取的文本")

    def test_service_tts_request_contains_selected_language(self) -> None:
        captured: dict = {}

        def request(_url, *, payload=None, timeout=0):
            captured.update(payload or {})
            return {"audio_base64": ""}

        with tempfile.TemporaryDirectory() as temp, patch(
            "videodub.qwen_speech._json_request",
            side_effect=request,
        ), patch("videodub.qwen_speech._copy_qwen_audio"):
            synthesize_qwen(
                AppConfig(tts_backend="mlx", tts_language="Japanese"),
                "こんにちは",
                Path(temp) / "out.wav",
                RecordingRunner(),
            )
        self.assertEqual(captured["language"], "Japanese")


if __name__ == "__main__":
    unittest.main()
