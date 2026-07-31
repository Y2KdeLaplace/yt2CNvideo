from __future__ import annotations

import unittest
from pathlib import Path

from videodub.qwen_service import _transcribe_mlx


class QwenServiceTests(unittest.TestCase):
    def test_mlx_qwen_receives_audio_path_as_generate_input(self) -> None:
        calls: list[tuple[str, str | None]] = []

        class Model:
            def generate(self, audio: str, *, language: str | None = None):
                calls.append((audio, language))
                return {"text": "hello"}

        result = _transcribe_mlx(Model(), Path("audio.wav"), "English")

        self.assertEqual(result, {"text": "hello"})
        self.assertEqual(calls, [("audio.wav", "English")])


if __name__ == "__main__":
    unittest.main()
