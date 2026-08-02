import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from videodub.config import AppConfig
from videodub.media import VideoJob
from videodub.runner import ProcessRunner
from videodub.subtitles import Cue
from videodub.tts import (
    AlignedSentence,
    AlignedToken,
    SentenceUnit,
    _TTSAlignmentQualityError,
    _atempo_chain,
    _align_full_audio,
    _aligned_tokens_from_items,
    _audio_chunk_ranges,
    _alignment_cue_ranges,
    _joined_speech_text,
    _map_sentences,
    _prepare_transcript,
    _render_aligned_audio,
    _scheduled_sentence_start,
    _synthesize_qwen,
    _tts_text_chunks,
    _wav_duration_ms,
    dub_video,
)


class TtsTests(unittest.TestCase):
    def test_atempo_chain_splits_large_factor(self) -> None:
        value = _atempo_chain(5.0)
        self.assertTrue(value.startswith("atempo=2.0,atempo=2.0,"))
        self.assertIn("atempo=1.250000", value)

    def test_tts_synthesizes_the_complete_video_text_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls: list[str] = []

            def synthesize(_config, text, output, _runner, **_kwargs):
                calls.append(text)
                output.write_bytes(b"audio")

            with (
                patch(
                    "videodub.tts.check_qwen_service",
                    return_value=SimpleNamespace(
                        available=True,
                        model="Qwen3-TTS",
                        error="",
                    ),
                ),
                patch(
                    "videodub.tts.synthesize_qwen",
                    side_effect=synthesize,
                ),
            ):
                output = _synthesize_qwen(
                    AppConfig(tts_backend="mlx"),
                    ProcessRunner(),
                    "嘿，宝贝，我回来了。嘿，亲爱的，怎么样？挺好的。工作怎么样？",
                    root,
                    "http://127.0.0.1:9955",
                )

        self.assertEqual(
            calls,
            ["嘿，宝贝，我回来了。嘿，亲爱的，怎么样？挺好的。工作怎么样？"],
        )
        self.assertEqual(output.name, "full_audio.wav")

    def test_long_tts_text_is_split_only_at_sentence_boundaries(self) -> None:
        first = "甲" * 350 + "。"
        second = "乙" * 350 + "。"
        third = "丙" * 100 + "。"

        chunks = _tts_text_chunks(first + second + third)

        self.assertEqual(chunks, [first, second + third])
        self.assertEqual("".join(chunks), first + second + third)

    def test_long_tts_uses_bounded_batches_and_joins_all_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = ("甲" * 350 + "。") + ("乙" * 350 + "。")
            batches: list[list[str]] = []

            def synthesize_batch(_config, texts, outputs, _runner, **_kwargs):
                batches.append(texts)
                for output in outputs:
                    with wave.open(str(output), "wb") as destination:
                        destination.setnchannels(1)
                        destination.setsampwidth(2)
                        destination.setframerate(24000)
                        destination.writeframes(b"\0\0" * 2400)

            with (
                patch(
                    "videodub.tts.check_qwen_service",
                    return_value=SimpleNamespace(
                        available=True,
                        model="Qwen3-TTS",
                        error="",
                    ),
                ),
                patch("videodub.tts.synthesize_qwen_batch", side_effect=synthesize_batch),
            ):
                output = _synthesize_qwen(
                    AppConfig(tts_backend="mlx"),
                    ProcessRunner(),
                    text,
                    root,
                    "http://127.0.0.1:9955",
                )

            self.assertEqual(batches, [["甲" * 350 + "。", "乙" * 350 + "。"]])
            self.assertGreater(_wav_duration_ms(output), 200)
            self.assertTrue((root / "tts-chunks.json").is_file())

    def test_alignment_reuses_exact_tts_chunk_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = ("甲" * 350 + "。") + ("乙" * 350 + "。")

            def synthesize_batch(_config, texts, outputs, _runner, **_kwargs):
                for output in outputs:
                    with wave.open(str(output), "wb") as destination:
                        destination.setnchannels(1)
                        destination.setsampwidth(2)
                        destination.setframerate(24000)
                        destination.writeframes(b"\0\0" * 2400)

            with (
                patch(
                    "videodub.tts.check_qwen_service",
                    return_value=SimpleNamespace(
                        available=True,
                        model="Qwen3-TTS",
                        error="",
                    ),
                ),
                patch("videodub.tts.synthesize_qwen_batch", side_effect=synthesize_batch),
            ):
                raw = _synthesize_qwen(
                    AppConfig(tts_backend="mlx"),
                    ProcessRunner(),
                    text,
                    root,
                    "http://127.0.0.1:9955",
                )

            aligned_texts: list[str] = []

            def align(_path, chunk_text, _language, **_kwargs):
                aligned_texts.append(chunk_text)
                return [{"text": chunk_text, "start": 0.0, "end": 0.08}]

            with (
                patch("videodub.tts.align_qwen", side_effect=align),
                patch(
                    "videodub.tts._audio_chunk_ranges",
                    side_effect=AssertionError("must use the TTS manifest"),
                ),
            ):
                tokens = _align_full_audio(
                    AppConfig(),
                    ProcessRunner(),
                    [Cue(1, 0, 1000, text)],
                    raw,
                    root,
                    "http://127.0.0.1:9957",
                )

            self.assertEqual(aligned_texts, _tts_text_chunks(text))
            self.assertEqual(len(tokens), 2)
            self.assertGreater(tokens[1].start_ms, tokens[0].end_ms)

    def test_joined_speech_text_keeps_western_words_separate(self) -> None:
        cues = [Cue(1, 0, 1000, "Hello"), Cue(2, 1000, 2000, "world.")]
        self.assertEqual(_joined_speech_text(cues), "Hello world.")

    def test_alignment_chunks_choose_largest_srt_gap_in_safe_window(self) -> None:
        cues = [
            Cue(1, 0, 180_000, "甲" * 240),
            Cue(2, 181_000, 200_000, "乙" * 20),
            Cue(3, 210_000, 220_000, "丙" * 30),
            Cue(4, 222_000, 233_000, "丁" * 20),
        ]
        self.assertEqual(
            _alignment_cue_ranges(cues, 310_000),
            [(0, 2), (2, 4)],
        )

    def test_short_video_splits_when_its_tts_audio_exceeds_five_minutes(self) -> None:
        cues = [
            Cue(1, 0, 100_000, "甲" * 230),
            Cue(2, 101_000, 170_000, "乙" * 20),
            Cue(3, 180_000, 210_000, "丙" * 30),
            Cue(4, 212_000, 233_000, "丁" * 30),
        ]

        ranges = _alignment_cue_ranges(cues, 327_500)

        self.assertEqual(ranges, [(0, 2), (2, 4)])

    def test_audio_chunk_search_relaxes_window_and_silence_threshold(self) -> None:
        config = AppConfig()
        runner = ProcessRunner()
        messages: list[str] = []
        runner.logger = messages.append
        cues = [
            Cue(1, 0, 290_000, "甲"),
            Cue(2, 291_000, 580_000, "乙"),
        ]

        def silences(_config, _runner, _path, *, noise_db, minimum_ms):
            if (noise_db, minimum_ms) == (-30, 40):
                return [(299_950, 300_050)]
            return []

        with patch("videodub.tts._silence_intervals", side_effect=silences):
            ranges = _audio_chunk_ranges(
                config,
                runner,
                Path("full.wav"),
                cues,
                [(0, 1), (1, 2)],
                580_000,
            )

        self.assertEqual(ranges, [(0, 300_000), (300_000, 580_000)])
        self.assertTrue(any("降级搜索" in message for message in messages))

    def test_audio_chunk_search_expands_before_relaxing_silence(self) -> None:
        config = AppConfig()
        runner = ProcessRunner()
        messages: list[str] = []
        runner.logger = messages.append
        cues = [
            Cue(1, 0, 180_000, "甲" * 388),
            Cue(2, 185_000, 233_000, "乙" * 120),
        ]

        def silences(_config, _runner, _path, *, noise_db, minimum_ms):
            if (noise_db, minimum_ms) == (-42, 120):
                return [(219_100, 219_600)]
            return []

        with patch("videodub.tts._silence_intervals", side_effect=silences):
            ranges = _audio_chunk_ranges(
                config,
                runner,
                Path("full.wav"),
                cues,
                [(0, 1), (1, 2)],
                327_500,
            )

        self.assertEqual(ranges, [(0, 219_350), (219_350, 327_500)])
        self.assertTrue(any("±60s，-42dB/120ms" in line for line in messages))

    def test_audio_chunk_search_never_hard_cuts_when_no_silence_exists(self) -> None:
        with patch("videodub.tts._silence_intervals", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "拒绝硬切语音"):
                _audio_chunk_ranges(
                    AppConfig(),
                    ProcessRunner(),
                    Path("full.wav"),
                    [
                        Cue(1, 0, 290_000, "甲"),
                        Cue(2, 291_000, 580_000, "乙"),
                    ],
                    [(0, 1), (1, 2)],
                    580_000,
                )

    def test_single_long_cue_without_safe_text_boundary_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "没有字幕块边界"):
            _alignment_cue_ranges(
                [Cue(1, 0, 239_000, "无法在一句中间硬切")],
                301_000,
            )

    def test_existing_translation_transcript_is_read_without_reexport(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            job = VideoJob(root / "Lecture.mp4", generated_dir=output)
            transcript_path = job.translated_transcript_path("Chinese")
            transcript_path.write_text("你好，世界。", encoding="utf-8")

            transcript = _prepare_transcript(
                job,
                "Chinese",
                [Cue(1, 0, 1000, "你好世界")],
            )

            self.assertEqual(transcript, "你好，世界。")
            self.assertEqual(transcript_path.read_text(encoding="utf-8"), transcript)

    def test_forced_alignment_tokens_map_to_complete_sentences(self) -> None:
        cues = [
            Cue(1, 0, 1000, "你好，"),
            Cue(2, 1000, 2000, "世界。"),
            Cue(3, 2500, 3500, "再见。"),
        ]
        tokens = [
            AlignedToken("你", 100, 300),
            AlignedToken("好", 300, 500),
            AlignedToken("世", 600, 800),
            AlignedToken("界", 800, 1000),
            AlignedToken("再", 1200, 1400),
            AlignedToken("见", 1400, 1600),
        ]

        aligned = _map_sentences(cues, tokens)

        self.assertEqual(
            [(item.source_start_ms, item.source_end_ms) for item in aligned],
            [(100, 1000), (1200, 1600)],
        )

    def test_zero_length_character_timestamp_is_kept_as_a_point(self) -> None:
        tokens = _aligned_tokens_from_items(
            [
                {"text": "感", "start": 1.0, "end": 1.16},
                {"text": "谢", "start": 1.16, "end": 1.16},
                {"text": "你", "start": 1.16, "end": 1.24},
            ],
            10_000,
            1,
        )

        self.assertEqual(
            tokens,
            [
                AlignedToken("感", 11_000, 11_160),
                AlignedToken("谢", 11_160, 11_160),
                AlignedToken("你", 11_160, 11_240),
            ],
        )

    def test_truly_backward_alignment_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "字符 '好' 的时间戳发生倒退"):
            _aligned_tokens_from_items(
                [
                    {"text": "你", "start": 1.0, "end": 1.2},
                    {"text": "好", "start": 0.9, "end": 1.1},
                ],
                0,
                1,
            )

    def test_point_only_sentence_borrows_audio_between_real_anchors(self) -> None:
        aligned = _map_sentences(
            [
                Cue(1, 0, 1000, "你好。"),
                Cue(2, 1500, 2500, "再见。"),
            ],
            [
                AlignedToken("你", 500, 500),
                AlignedToken("好", 500, 500),
                AlignedToken("再", 900, 1100),
                AlignedToken("见", 1100, 1300),
            ],
        )

        self.assertEqual(
            [(item.source_start_ms, item.source_end_ms) for item in aligned],
            [(500, 900), (900, 1300)],
        )

    def test_unrecoverable_point_only_sentence_becomes_silence(self) -> None:
        aligned = _map_sentences(
            [Cue(1, 0, 1000, "你好。")],
            [
                AlignedToken("你", 500, 500),
                AlignedToken("好", 500, 500),
            ],
        )

        self.assertEqual(aligned[0].source_start_ms, 500)
        self.assertEqual(aligned[0].source_end_ms, 500)

    def test_unrecoverable_point_sentence_does_not_abort_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            messages: list[str] = []
            runner = ProcessRunner(messages.append)
            runner.run = lambda *_args, **_kwargs: self.fail(
                "zero-length fallback must not invoke ffmpeg"
            )

            output = _render_aligned_audio(
                AppConfig(),
                runner,
                root / "unused.wav",
                [
                    AlignedSentence(
                        SentenceUnit(34, 34, 0, 1, "喂？"),
                        500,
                        500,
                    )
                ],
                root,
                1000,
            )

            self.assertEqual(_wav_duration_ms(output), 1000)
            self.assertTrue(any("保留目标时间窗静音" in line for line in messages))

    def test_short_cue_borrows_surrounding_silence_within_safety_limits(self) -> None:
        start = _scheduled_sentence_start(
            124_960,
            125_040,
            448,
            121_840,
            168_000,
            is_last=False,
        )

        self.assertEqual(start, 124_842)
        self.assertEqual(124_960 - start, 118)
        self.assertEqual(start + 448 - 125_040, 250)

    def test_wav_duration_is_read_without_starting_ffprobe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "voice.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(b"\0\0" * 12000)

            self.assertEqual(_wav_duration_ms(path), 500)

    def test_ffmpeg_input_trim_allows_atempo_to_shorten_sentence(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("ffmpeg is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.wav"
            with wave.open(str(raw), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(b"\0\0" * 24000)

            aligned = _render_aligned_audio(
                AppConfig(ffmpeg_path=ffmpeg),
                ProcessRunner(),
                raw,
                [
                    AlignedSentence(
                        SentenceUnit(1, 1, 0, 800, "测试。"),
                        0,
                        1000,
                    )
                ],
                root,
                800,
            )

            self.assertEqual(_wav_duration_ms(aligned), 800)

    def test_extreme_speed_ratio_is_rendered_instead_of_rejected(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("ffmpeg is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.wav"
            with wave.open(str(raw), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(b"\0\0" * 48000)
            messages: list[str] = []

            aligned = _render_aligned_audio(
                AppConfig(ffmpeg_path=ffmpeg),
                ProcessRunner(messages.append),
                raw,
                [
                    AlignedSentence(
                        SentenceUnit(1, 1, 0, 400, "测试。"),
                        0,
                        2000,
                    )
                ],
                root,
                400,
            )

            self.assertEqual(_wav_duration_ms(aligned), 400)
            self.assertTrue(any("强制变速" in message for message in messages))

    def test_subtitle_only_job_generates_audio_without_muxing_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            config = AppConfig(
                work_dir=str(root),
                cache_dir=str(root / "cache"),
                translation_language="Chinese",
            )
            job = VideoJob(
                root / "Lecture.mp4",
                generated_dir=output,
                source_subtitle_path=root / "Lecture.en.srt",
            )
            job.chinese_subtitle_path.write_text(
                (
                    "1\n00:00:00,000 --> 00:00:01,001\n你好。\n\n"
                    "2\n00:00:01,000 --> 00:00:02,000\n再见。\n"
                ),
                encoding="utf-8",
            )

            def render_track(*args, **kwargs):
                work_dir = args[4]
                voice = work_dir / "aligned_audio.wav"
                voice.write_bytes(b"audio")
                return voice

            messages: list[str] = []
            runner = ProcessRunner(messages.append)

            def run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"audio")
                return []

            runner.run = run

            with (
                patch(
                    "videodub.tts._synthesize_qwen",
                    return_value=Path("raw.wav"),
                ) as synthesize,
                patch(
                    "videodub.tts._align_full_audio",
                    side_effect=[
                        _TTSAlignmentQualityError("退化音频"),
                        [],
                    ],
                ),
                patch("videodub.tts._map_sentences", return_value=[]),
                patch("videodub.tts._render_aligned_audio", side_effect=render_track),
                patch(
                    "videodub.tts.check_qwen_service",
                    return_value=SimpleNamespace(
                        available=True,
                        model="Qwen3-ForcedAligner",
                        error="",
                    ),
                ),
                patch("videodub.tts.media_duration", side_effect=AssertionError),
                patch("videodub.tts._mux_video", side_effect=AssertionError),
            ):
                result = dub_video(config, runner, job)

            self.assertEqual(result, output / "Lecture.Chinese配音.m4a")
            self.assertEqual(result.read_bytes(), b"audio")
            self.assertEqual(synthesize.call_count, 2)
            self.assertTrue(any("时间轴重叠 1ms" in message for message in messages))

if __name__ == "__main__":
    unittest.main()
