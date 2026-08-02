from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from videodub.qwen_service import (
    _generate_tts_batch,
    _timestamp_segments,
    _transcribe_mlx,
)


class QwenServiceTests(unittest.TestCase):
    def test_mlx_base_tts_uses_shared_reference_batch_generation(self) -> None:
        model = SimpleNamespace()
        model.batch_generate = Mock(
            return_value=iter(
                [
                    SimpleNamespace(sequence_idx=1, audio="second", sample_rate=24000),
                    SimpleNamespace(sequence_idx=0, audio="first", sample_rate=24000),
                ]
            )
        )
        args = SimpleNamespace(
            backend="mlx",
            variant="base",
            reference_audio="voice.wav",
            reference_text="reference",
            speaker="Vivian",
        )

        outputs = _generate_tts_batch(model, args, ["一", "二"], "Chinese")

        self.assertEqual(outputs, [("first", 24000), ("second", 24000)])
        model.batch_generate.assert_called_once_with(
            ["一", "二"],
            lang_code="Chinese",
            ref_audio="voice.wav",
            ref_text="reference",
        )

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

    def test_official_qwen_does_not_force_split_without_natural_boundary(self) -> None:
        words = [
            SimpleNamespace(
                text=f"word{index}",
                start_time=index * 0.6,
                end_time=(index + 1) * 0.6,
            )
            for index in range(12)
        ]
        result = SimpleNamespace(
            text=" ".join(item.text for item in words) + ".",
            time_stamps=[SimpleNamespace(items=words)],
        )

        segments = _timestamp_segments(result)

        self.assertEqual(
            segments,
            [
                {
                    "text": " ".join(item.text for item in words) + ".",
                    "start": 0.0,
                    "end": 7.2,
                }
            ],
        )

    def test_official_qwen_splits_long_sentence_at_natural_small_pause(self) -> None:
        words = (
            ("This", 0.0, 0.4),
            ("continuous", 0.4, 0.9),
            ("explanation", 0.9, 1.5),
            ("reaches", 1.5, 2.0),
            ("a", 2.35, 2.45),
            ("natural", 2.45, 2.9),
            ("pause", 2.9, 3.3),
            ("here", 3.3, 3.7),
        )
        result = SimpleNamespace(
            text="This continuous explanation reaches a natural pause here.",
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
                    "text": "This continuous explanation reaches",
                    "start": 0.0,
                    "end": 2.0,
                },
                {
                    "text": "a natural pause here.",
                    "start": 2.35,
                    "end": 3.7,
                },
            ],
        )

    def test_official_qwen_uses_smaller_pause_after_soft_maximum(self) -> None:
        words = [
            SimpleNamespace(
                text=f"w{index}",
                start_time=index * 0.5 + (0.15 if index >= 10 else 0.0),
                end_time=(index + 1) * 0.5 + (0.15 if index >= 10 else 0.0),
            )
            for index in range(12)
        ]
        result = SimpleNamespace(
            text=" ".join(item.text for item in words) + ".",
            time_stamps=[SimpleNamespace(items=words)],
        )

        segments = _timestamp_segments(result)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["end"], 5.0)
        self.assertEqual(segments[1]["start"], 5.15)


if __name__ == "__main__":
    unittest.main()
