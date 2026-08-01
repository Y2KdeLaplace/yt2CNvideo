from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from videodub.qwen_service import _timestamp_segments, _transcribe_mlx


class QwenServiceTests(unittest.TestCase):
    def test_official_qwen_preserves_spaces_and_uses_natural_segments(self) -> None:
        words = (
            ("Good", 1.6, 1.9),
            ("morning", 1.9, 2.4),
            ("ladies", 2.5, 2.9),
            ("and", 2.9, 3.1),
            ("gentlemen", 3.1, 3.7),
            ("We", 4.0, 4.2),
            ("are", 4.2, 4.4),
            ("about", 4.4, 4.7),
            ("to", 4.7, 4.8),
            ("begin", 4.8, 5.1),
            ("boarding", 5.1, 5.6),
        )
        result = SimpleNamespace(
            text=(
                "Good morning, ladies and gentlemen. "
                "We are about to begin boarding."
            ),
            time_stamps=[
                SimpleNamespace(
                    items=[
                        SimpleNamespace(text=text, start_time=start, end_time=end)
                        for text, start, end in words
                    ]
                )
            ],
        )

        segments = _timestamp_segments(result)

        self.assertEqual(
            segments,
            [
                {
                    "text": "Good morning, ladies and gentlemen.",
                    "start": 1.6,
                    "end": 3.7,
                },
                {
                    "text": "We are about to begin boarding.",
                    "start": 4.0,
                    "end": 5.6,
                },
            ],
        )

    def test_mlx_qwen_aligns_words_and_builds_natural_segments(self) -> None:
        calls: list[tuple[object, str | None]] = []

        class Model:
            def generate(
                self,
                audio: object,
                *,
                language: str | None = None,
            ):
                calls.append((audio, language))
                return {
                    "text": (
                        "Good morning, ladies and gentlemen. "
                        "We are about to begin boarding."
                    )
                }

        words = (
            ("Good", 0.1, 0.4),
            ("morning", 0.4, 0.9),
            ("ladies", 1.0, 1.4),
            ("and", 1.4, 1.6),
            ("gentlemen", 1.6, 2.2),
            ("We", 2.5, 2.7),
            ("are", 2.7, 2.9),
            ("about", 2.9, 3.2),
            ("to", 3.2, 3.3),
            ("begin", 3.3, 3.6),
            ("boarding", 3.6, 4.1),
        )

        class Aligner:
            def generate(self, audio, text, language):
                self.call = (audio, text, language)
                return SimpleNamespace(
                    items=[
                        SimpleNamespace(text=text, start_time=start, end_time=end)
                        for text, start, end in words
                    ]
                )

        aligner = Aligner()
        with patch(
            "videodub.qwen_service._mlx_audio_chunks",
            return_value=[("audio-chunk", 10.0)],
        ):
            result = _transcribe_mlx(
                Model(),
                aligner,
                Path("audio.wav"),
                "English",
            )

        self.assertEqual(calls, [("audio-chunk", "English")])
        self.assertEqual(aligner.call[0], "audio-chunk")
        self.assertEqual(
            result["segments"],
            [
                {
                    "text": "Good morning, ladies and gentlemen.",
                    "start": 10.1,
                    "end": 12.2,
                },
                {
                    "text": "We are about to begin boarding.",
                    "start": 12.5,
                    "end": 14.1,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
