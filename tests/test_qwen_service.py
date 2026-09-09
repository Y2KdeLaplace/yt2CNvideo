from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from videodub.qwen_service import (
    _alignment_to_segments,
    _generate_tts_batch,
    _generate_tts_one,
    _timestamp_segments,
    _transcribe_mlx,
    _validate_chunk_alignment_bounds,
)
from videodub.sentences import TimelineError


class SizedAudio:
    def __init__(self, duration_seconds: float, name: str) -> None:
        self.sample_count = round(duration_seconds * 16000)
        self.name = name

    def __len__(self) -> int:
        return self.sample_count


class QwenServiceTests(unittest.TestCase):
    def test_zero_duration_word_inside_sentence_builds_segment_without_repair(self):
        words = [
            SimpleNamespace(text="I'm", start_time=10.0, end_time=10.08),
            SimpleNamespace(text="not", start_time=10.08, end_time=10.18),
            SimpleNamespace(text="ready", start_time=10.18, end_time=10.3),
            SimpleNamespace(text="yet", start_time=10.32, end_time=10.32),
        ]
        original_timestamps = [
            (word.start_time, word.end_time) for word in words
        ]

        segments = _alignment_to_segments(
            SimpleNamespace(items=words),
            "I'm not ready yet.",
            0.0,
        )

        self.assertEqual(
            segments,
            [{"text": "I'm not ready yet.", "start": 10.0, "end": 10.32}],
        )
        self.assertEqual(
            [(word.start_time, word.end_time) for word in words],
            original_timestamps,
        )

    def test_reversed_word_reports_absolute_time_without_repairing_it(self):
        word = SimpleNamespace(text="You", start_time=1.3, end_time=1.2)
        with self.assertRaisesRegex(
            TimelineError,
            "invalid_or_non_monotonic_timeline",
        ) as error:
            _alignment_to_segments(SimpleNamespace(items=[word]), "You", 240.0)
        self.assertIn("start=241.300s end=241.200s", str(error.exception))
        self.assertEqual((word.start_time, word.end_time), (1.3, 1.2))

    def test_non_monotonic_and_abnormal_overlap_remain_invalid(self):
        cases = [
            [("one", 0.14, 0.15), ("two", 0.13, 0.3)],
            [("one", 0.0, 0.5), ("two", 0.35, 0.8)],
        ]
        for words in cases:
            with self.subTest(words=words), self.assertRaisesRegex(
                TimelineError,
                "invalid_or_non_monotonic_timeline",
            ):
                _alignment_to_segments(
                    SimpleNamespace(
                        items=[
                            SimpleNamespace(
                                text=text,
                                start_time=start,
                                end_time=end,
                            )
                            for text, start, end in words
                        ]
                    ),
                    "one two",
                    0.0,
                )

    def test_zero_duration_word_at_segment_start_waits_for_positive_duration(self):
        words = [
            SimpleNamespace(text="Wait", start_time=10.0, end_time=10.0),
            SimpleNamespace(text="here", start_time=10.8, end_time=10.88),
        ]

        segments = _alignment_to_segments(
            SimpleNamespace(items=words),
            "Wait here.",
            0.0,
        )

        self.assertEqual(
            segments,
            [{"text": "Wait here.", "start": 10.0, "end": 10.88}],
        )
        self.assertEqual((words[0].start_time, words[0].end_time), (10.0, 10.0))

    def test_isolated_zero_duration_word_is_an_alignment_failure(self):
        word = SimpleNamespace(text="You", start_time=1.232, end_time=1.232)
        with self.assertRaisesRegex(
            TimelineError,
            "zero_duration_spoken_segment",
        ) as error:
            _alignment_to_segments(SimpleNamespace(items=[word]), "You", 240.0)

        self.assertIn("start=241.232s end=241.232s", str(error.exception))
        self.assertEqual((word.start_time, word.end_time), (1.232, 1.232))

    def test_chunk_bound_violation_is_not_clamped(self):
        within_tolerance = SimpleNamespace(
            text="Edge",
            start_time=119.92,
            end_time=120.0005,
        )
        _validate_chunk_alignment_bounds([within_tolerance], 120.0, 240.0)

        word = SimpleNamespace(text="Beyond", start_time=119.92, end_time=132.0)
        with self.assertRaisesRegex(
            TimelineError,
            "chunk_bound_violation",
        ) as error:
            _validate_chunk_alignment_bounds([word], 120.0, 240.0)

        self.assertIn("chunk=240.000s–360.000s", str(error.exception))
        self.assertEqual((word.start_time, word.end_time), (119.92, 132.0))

    def test_mlx_tts_limit_is_reported_for_caller_owned_retry(self) -> None:
        model = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda _text: list(range(100))),
            generate=Mock(
                side_effect=[
                    iter(
                        [
                            SimpleNamespace(
                                audio="bad",
                                sample_rate=24000,
                                token_count=600,
                            )
                        ]
                    ),
                    iter(
                        [
                            SimpleNamespace(
                                audio="good",
                                sample_rate=24000,
                                token_count=200,
                            )
                        ]
                    ),
                ]
            ),
        )
        args = SimpleNamespace(
            backend="mlx",
            variant="base",
            reference_audio="voice.wav",
            reference_text="reference",
            speaker="Vivian",
        )

        with self.assertRaisesRegex(RuntimeError, "生成长度上限"):
            _generate_tts_one(model, args, "完整文稿", "Chinese")
        self.assertEqual(model.generate.call_count, 1)

    def test_empty_iterator_is_a_clear_error(self) -> None:
        args = SimpleNamespace(backend="mlx", variant="base", reference_audio="voice.wav",
                               reference_text="ref", speaker="Vivian")
        model = SimpleNamespace(generate=Mock(return_value=iter([])))
        with self.assertRaisesRegex(RuntimeError, "未生成任何音频.*你好"):
            _generate_tts_one(model, args, "你好", "Chinese")

    def test_sequential_batch_preserves_success_when_peer_is_empty(self) -> None:
        args = SimpleNamespace(backend="mlx", variant="base", reference_audio="voice.wav",
                               reference_text="ref", speaker="Vivian")
        model = SimpleNamespace(generate=Mock(side_effect=[iter([SimpleNamespace(audio="good")]), iter([])]))
        result = _generate_tts_batch(model, args, ["一", "二"], "Chinese")
        self.assertEqual(result[0], ("good", 24000))
        self.assertIsInstance(result[1], RuntimeError)

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
            temperature=0.7,
            top_p=0.9,
            max_tokens=4096,
            ref_audio="voice.wav",
            ref_text="reference",
        )

    def test_empty_native_batch_does_not_add_hidden_generation_retries(self):
        args = SimpleNamespace(backend="mlx", variant="base", reference_audio="voice.wav",
                               reference_text="ref", speaker="Vivian")
        model = SimpleNamespace(batch_generate=Mock(return_value=iter([])), generate=Mock())
        results = _generate_tts_batch(model, args, ["一", "二"], "Chinese")
        self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
        model.generate.assert_not_called()
        model.batch_generate.assert_called_once()

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
        audio_chunk = SizedAudio(5.0, "audio-chunk")
        with patch(
            "videodub.qwen_service._mlx_audio_chunks",
            return_value=[(audio_chunk, 10.0)],
        ):
            result = _transcribe_mlx(
                Model(),
                aligner,
                Path("audio.wav"),
                "English",
            )

        self.assertEqual(calls, [(audio_chunk, "English")])
        self.assertIs(aligner.call[0], audio_chunk)
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

    def test_failed_parent_chunk_retries_at_120_seconds_with_absolute_offset(self):
        parent = SizedAudio(240.0, "parent")
        child = SizedAudio(120.0, "child")
        model = SimpleNamespace(
            generate=Mock(side_effect=lambda audio, **_kwargs: {"text": audio.name})
        )

        def align(audio, _text, _language):
            if audio is parent:
                start = end = 1.0
            else:
                start = 2.0
                end = 3.0
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        text=audio.name,
                        start_time=start,
                        end_time=end,
                    )
                ]
            )

        aligner = SimpleNamespace(generate=Mock(side_effect=align))
        with patch(
            "videodub.qwen_service._mlx_audio_chunks",
            return_value=[(parent, 240.0)],
        ), patch(
            "videodub.qwen_service._split_mlx_audio",
            return_value=[(child, 60.0)],
        ) as split:
            result = _transcribe_mlx(model, aligner, Path("audio.wav"), "English")

        split.assert_called_once_with(parent, 120.0)
        self.assertEqual(result["text"], "child")
        self.assertEqual(
            result["segments"],
            [{"text": "child", "start": 302.0, "end": 303.0}],
        )

    def test_alignment_failure_after_60_second_retry_reports_chunk_range(self):
        parent = SizedAudio(240.0, "parent")
        child = SizedAudio(120.0, "child")
        grandchild = SizedAudio(60.0, "grandchild")
        model = SimpleNamespace(
            generate=Mock(side_effect=lambda audio, **_kwargs: {"text": audio.name})
        )

        def align(audio, _text, _language):
            duration = len(audio) / 16000
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        text=audio.name,
                        start_time=0.0,
                        end_time=duration + 1.0,
                    )
                ]
            )

        def split(audio, max_duration):
            if audio is parent and max_duration == 120.0:
                return [(child, 0.0)]
            if audio is child and max_duration == 60.0:
                return [(grandchild, 0.0)]
            self.fail("unexpected subdivision")

        aligner = SimpleNamespace(generate=Mock(side_effect=align))
        with patch(
            "videodub.qwen_service._mlx_audio_chunks",
            return_value=[(parent, 240.0)],
        ), patch(
            "videodub.qwen_service._split_mlx_audio",
            side_effect=split,
        ):
            with self.assertRaisesRegex(
                TimelineError,
                "240→120→60.*chunk=240.000s–300.000s.*chunk_bound_violation",
            ):
                _transcribe_mlx(model, aligner, Path("audio.wav"), "English")

        self.assertEqual(model.generate.call_count, 3)
        self.assertEqual(aligner.generate.call_count, 3)

    def test_multiple_mlx_chunks_produce_global_monotonic_segments(self):
        first = SizedAudio(5.0, "First")
        second = SizedAudio(5.0, "Second")
        model = SimpleNamespace(
            generate=Mock(side_effect=lambda audio, **_kwargs: {"text": audio.name})
        )
        aligner = SimpleNamespace(
            generate=Mock(
                side_effect=lambda audio, _text, _language: SimpleNamespace(
                    items=[
                        SimpleNamespace(
                            text=audio.name,
                            start_time=0.1,
                            end_time=0.5,
                        )
                    ]
                )
            )
        )
        with patch(
            "videodub.qwen_service._mlx_audio_chunks",
            return_value=[(first, 0.0), (second, 5.0)],
        ):
            result = _transcribe_mlx(model, aligner, Path("audio.wav"), "English")

        self.assertEqual(
            result["segments"],
            [
                {"text": "First", "start": 0.1, "end": 0.5},
                {"text": "Second", "start": 5.1, "end": 5.5},
            ],
        )

    def test_global_segment_regression_is_rejected_inside_asr_service(self):
        first = SizedAudio(300.0, "First")
        second = SizedAudio(10.0, "Second")
        timestamps = {
            "First": (256.0, 256.72),
            "Second": (0.164, 0.564),
        }
        model = SimpleNamespace(
            generate=Mock(side_effect=lambda audio, **_kwargs: {"text": audio.name})
        )
        aligner = SimpleNamespace(
            generate=Mock(
                side_effect=lambda audio, _text, _language: SimpleNamespace(
                    items=[
                        SimpleNamespace(
                            text=audio.name,
                            start_time=timestamps[audio.name][0],
                            end_time=timestamps[audio.name][1],
                        )
                    ]
                )
            )
        )
        with patch(
            "videodub.qwen_service._mlx_audio_chunks",
            return_value=[(first, 0.0), (second, 244.0)],
        ):
            with self.assertRaisesRegex(
                TimelineError,
                "MLX ASR 全局时间轴.*non_monotonic_segment_timeline",
            ):
                _transcribe_mlx(model, aligner, Path("audio.wav"), "English")

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
