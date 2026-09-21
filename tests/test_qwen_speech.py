from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videodub.config import AppConfig
from videodub.qwen_speech import (
    _segments_to_cues,
    check_qwen_service,
    resolve_tts_reference,
    synthesize_qwen,
    synthesize_qwen_batch,
)
from videodub.subtitles import Cue


class QwenSpeechCompatibilityTests(unittest.TestCase):
    def test_health_compatibility_reads_generic_service(self) -> None:
        with patch(
            "videodub.qwen_speech._json_request",
            return_value={
                "status": "ok",
                "type": "tts",
                "model": "qwen3-tts-mlx-customvoice-0.6b-8bit",
                "runtime": "qwen3-tts-mlx",
                "pid": 26762,
            },
        ):
            info = check_qwen_service("http://tts", "tts")

        self.assertTrue(info.available)
        self.assertEqual(info.pid, 26762)
        self.assertEqual(info.backend, "qwen3-tts-mlx")

    def test_asr_segments_preserve_acoustic_boundaries(self) -> None:
        cues = _segments_to_cues(
            [
                {"text": "Hello", "start": 0.25, "end": 1.5},
                {"text": "world", "start": 1.5, "end": 2.0},
            ],
            "",
        )
        self.assertEqual(
            cues,
            [Cue(1, 250, 1500, "Hello"), Cue(2, 1500, 2000, "world")],
        )

    def test_text_without_timestamps_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "没有声学时间戳"):
            _segments_to_cues(None, "Fallback")

    def test_runtime_calls_are_forwarded_to_generic_client(self) -> None:
        config = AppConfig()
        output = Path("speech.wav")
        with (
            patch("videodub.qwen_speech.synthesize_speech") as single,
            patch("videodub.qwen_speech.synthesize_speech_batch") as batch,
        ):
            synthesize_qwen(config, "一", output, object())
            synthesize_qwen_batch(config, ["一"], [output], object())
        single.assert_called_once()
        batch.assert_called_once()

    def test_reference_text_remains_a_desktop_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "voice.wav"
            text = root / "voice.data"
            audio.touch()
            text.write_text("实际读取的文本", encoding="utf-8")
            resolved_audio, resolved_text = resolve_tts_reference(
                AppConfig(
                    tts_use_custom_voice=True,
                    tts_reference_audio=str(audio),
                    tts_reference_text_file=str(text),
                )
            )
            self.assertEqual(Path(resolved_audio), audio.resolve())
            self.assertEqual(resolved_text, "实际读取的文本")


if __name__ == "__main__":
    unittest.main()
