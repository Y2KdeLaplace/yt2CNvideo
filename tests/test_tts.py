from __future__ import annotations

import inspect
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydub import AudioSegment
from pydub.generators import Sine

from videodub.config import AppConfig
from videodub.media import VideoJob
from videodub.qwen_speech import QwenServiceInfo
from videodub.runner import ProcessRunner
from videodub.subtitles import Cue, write_srt
from videodub.tts import (
    CROSSFADE_MS,
    SentenceAudio,
    SentenceUnit,
    _adjust_sentence_audio,
    _atempo_chain,
    _audio_duration_ms,
    _fit_sentence_durations,
    _global_duration_factor,
    _local_duration_factor,
    _render_sentence_track,
    _scheduled_sentence_start,
    _sentence_units,
    _synthesize_sentence,
    _synthesize_sentence_units,
    dub_video,
)


def write_tone(path: Path, duration_ms: int, frequency: int = 440) -> None:
    audio = (
        Sine(frequency)
        .to_audio_segment(duration=duration_ms, volume=-6)
        .set_frame_rate(24000)
        .set_channels(1)
        .set_sample_width(2)
    )
    with path.open("wb") as destination:
        audio.export(destination, format="wav")


class AudioRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.messages: list[str] = []

    def logger(self, message: str) -> None:
        self.messages.append(message)

    def check_cancelled(self) -> None:
        return None

    def run(self, command, *, quiet: bool = False) -> list[str]:
        values = [str(item) for item in command]
        self.commands.append(values)
        output = Path(values[-1])
        if "-af" in values:
            source = Path(values[values.index("-i") + 1])
            with source.open("rb") as input_file:
                audio = AudioSegment.from_wav(input_file)
            filters = values[values.index("-af") + 1]
            speed = 1.0
            for value in re.findall(r"atempo=([0-9.]+)", filters):
                speed *= float(value)
            fitted = audio[: max(1, round(len(audio) / speed))]
            if len(fitted) < round(len(audio) / speed):
                fitted += AudioSegment.silent(
                    duration=round(len(audio) / speed) - len(fitted),
                    frame_rate=24000,
                )
            with output.open("wb") as destination:
                fitted.export(destination, format="wav")
        else:
            output.write_bytes(b"audio")
        return []


def sentence_audio(
    root: Path,
    index: int,
    *,
    start_ms: int,
    end_ms: int,
    raw_ms: int,
    text: str = "一句话。",
    frequency: int = 440,
) -> SentenceAudio:
    path = root / f"raw-{index}.wav"
    write_tone(path, raw_ms, frequency)
    unit = SentenceUnit(index + 1, index + 1, start_ms, end_ms, text)
    return SentenceAudio(unit, path, raw_ms, raw_ms)


class SentenceUnitTests(unittest.TestCase):
    def test_natural_sentence_units_keep_cue_and_timeline_boundaries(self) -> None:
        units = _sentence_units(
            [
                Cue(1, 100, 700, "这是"),
                Cue(2, 800, 1400, "第一句。"),
                Cue(3, 2000, 2600, "第二句！"),
            ]
        )

        self.assertEqual(
            units,
            [
                SentenceUnit(1, 2, 100, 1400, "这是第一句。"),
                SentenceUnit(3, 3, 2000, 2600, "第二句！"),
            ],
        )

    def test_long_sentence_chunks_still_produce_one_sentence_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            unit = SentenceUnit(1, 3, 0, 3000, "第一段。第二段。第三段。")

            def synthesize(_config, _text, output, _runner, **_kwargs):
                write_tone(output, 100)

            def synthesize_batch(_config, texts, outputs, _runner, **_kwargs):
                for text, output in zip(texts, outputs, strict=True):
                    self.assertTrue(text)
                    write_tone(output, 100)

            with (
                patch("videodub.tts.TTS_CHUNK_MAX_CHARS", 4),
                patch("videodub.tts.synthesize_qwen", side_effect=synthesize),
                patch(
                    "videodub.tts.synthesize_qwen_batch",
                    side_effect=synthesize_batch,
                ),
            ):
                result = _synthesize_sentence(
                    AppConfig(tts_backend="mlx"),
                    AudioRunner(),
                    unit,
                    0,
                    root,
                    "http://tts",
                )

            self.assertEqual(result.unit, unit)
            self.assertEqual(result.path.name, "sentence-00000.raw.wav")
            self.assertTrue(result.path.is_file())
            self.assertGreater(result.duration_ms, 100)

    def test_sentence_units_keep_independent_outputs_in_bounded_tts_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            units = [
                SentenceUnit(
                    index + 1,
                    index + 1,
                    index * 1000,
                    index * 1000 + 800,
                    text,
                )
                for index, text in enumerate(("第一句。", "第二句。", "第三句。"))
            ]
            batch_calls: list[list[str]] = []

            def synthesize(_config, _text, output, _runner, **_kwargs):
                write_tone(output, 100)

            def synthesize_batch(_config, texts, outputs, _runner, **_kwargs):
                batch_calls.append(texts)
                for output in outputs:
                    write_tone(output, 100)

            with (
                patch("videodub.tts.synthesize_qwen", side_effect=synthesize),
                patch(
                    "videodub.tts.synthesize_qwen_batch",
                    side_effect=synthesize_batch,
                ),
            ):
                results = _synthesize_sentence_units(
                    AppConfig(tts_backend="mlx"),
                    AudioRunner(),
                    units,
                    root,
                    "http://tts",
                )

        self.assertEqual(batch_calls, [["第一句。", "第二句。"]])
        self.assertEqual(
            [item.path.name for item in results],
            [
                "sentence-00000.raw.wav",
                "sentence-00001.raw.wav",
                "sentence-00002.raw.wav",
            ],
        )
        self.assertEqual([item.unit for item in results], units)


class DurationFittingTests(unittest.TestCase):
    def test_short_sentences_are_not_stretched_to_fill_their_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = [
                sentence_audio(root, 0, start_ms=0, end_ms=1000, raw_ms=300),
                sentence_audio(root, 1, start_ms=2000, end_ms=3000, raw_ms=300),
            ]
            runner = AudioRunner()

            fitted = _fit_sentence_durations(AppConfig(), runner, raw, root)
            track_path = _render_sentence_track(runner, fitted, root, 3500)
            with track_path.open("rb") as source:
                track = AudioSegment.from_wav(source)

        self.assertEqual(fitted[0].global_duration_ratio, 1.20)
        self.assertLess(fitted[0].duration_ms, 500)
        self.assertLess(fitted[1].duration_ms, 500)
        self.assertEqual(track[600:1900].rms, 0)

    def test_overlong_sentence_uses_atempo_and_approaches_target_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = sentence_audio(root, 0, start_ms=0, end_ms=1000, raw_ms=1400)
            runner = AudioRunner()

            fitted = _fit_sentence_durations(AppConfig(), runner, [raw], root)[0]

        self.assertAlmostEqual(fitted.global_duration_ratio, 0.80)
        self.assertAlmostEqual(fitted.local_duration_ratio, 0.90)
        self.assertLess(abs(fitted.duration_ms - 1008), 25)
        self.assertIn("atempo=", runner.commands[0][runner.commands[0].index("-af") + 1])

    def test_global_factor_reflects_multiple_slow_sentences(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentences = [
                sentence_audio(root, 0, start_ms=0, end_ms=1000, raw_ms=1500),
                sentence_audio(root, 1, start_ms=1200, end_ms=2200, raw_ms=1500),
            ]
            self.assertEqual(_global_duration_factor(sentences), 0.80)

    def test_abnormally_long_sentence_receives_extra_local_compression(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentences = [
                sentence_audio(root, 0, start_ms=0, end_ms=1000, raw_ms=1000),
                sentence_audio(root, 1, start_ms=1200, end_ms=2200, raw_ms=2000),
            ]
            global_ratio = _global_duration_factor(sentences)

        self.assertEqual(global_ratio, 0.80)
        self.assertEqual(_local_duration_factor(1000, 1000, global_ratio), 1.00)
        self.assertEqual(_local_duration_factor(2000, 1000, global_ratio), 0.90)

    def test_atempo_chain_supports_speed_factors_above_two(self) -> None:
        self.assertEqual(_atempo_chain(4.5), "atempo=2.0,atempo=2.0,atempo=1.125000")

    def test_real_ffmpeg_atempo_produces_expected_duration(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("ffmpeg is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = sentence_audio(root, 0, start_ms=0, end_ms=1000, raw_ms=1400)

            fitted = _adjust_sentence_audio(
                AppConfig(ffmpeg_path=ffmpeg),
                ProcessRunner(),
                raw,
                0,
                0.80,
                root,
            )

        self.assertEqual(fitted.path.name, "sentence-00000.fit.wav")
        self.assertLess(abs(fitted.duration_ms - 1008), 50)


class TimelineSchedulingTests(unittest.TestCase):
    def test_overlapping_subtitle_windows_are_scheduled_serially(self) -> None:
        first = SentenceUnit(1, 1, 0, 1000, "一。")
        second = SentenceUnit(2, 2, 800, 1600, "二。")
        first_start = _scheduled_sentence_start(first, 800, 900, 0, 3000, is_last=False)
        second_start = _scheduled_sentence_start(
            second,
            3000,
            900,
            first_start + 900,
            3000,
            is_last=True,
        )

        self.assertEqual(first_start, 0)
        self.assertGreaterEqual(second_start, first_start + 900 - CROSSFADE_MS)

    def test_only_last_sentence_is_clipped_at_video_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = sentence_audio(
                root,
                0,
                start_ms=0,
                end_ms=1000,
                raw_ms=900,
                frequency=440,
            )
            last = sentence_audio(
                root,
                1,
                start_ms=1700,
                end_ms=2300,
                raw_ms=700,
                frequency=880,
            )
            runner = AudioRunner()

            output = _render_sentence_track(runner, [first, last], root, 2000)

            self.assertEqual(_audio_duration_ms(output), 2000)
            with output.open("rb") as source:
                track = AudioSegment.from_wav(source)
            self.assertGreater(track[100:800].rms, 0)
            self.assertTrue(any("最后一句超出视频结尾" in item for item in runner.messages))


class DubVideoFlowTests(unittest.TestCase):
    def test_dub_video_uses_only_tts_service_and_sentence_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            subtitle = output / "Lecture.zh-CN.srt"
            output.mkdir()
            write_srt(
                subtitle,
                [
                    Cue(1, 0, 900, "第一句。"),
                    Cue(2, 1200, 2000, "第二句。"),
                ],
            )
            job = VideoJob(root / "Lecture.mp4", generated_dir=output)
            config = AppConfig(
                cache_dir=str(root / "cache"),
                work_dir=str(root),
                tts_backend="mlx",
            )
            runner = AudioRunner()
            raw = [sentence_audio(root, 0, start_ms=0, end_ms=900, raw_ms=500)]
            voice = root / "voice.wav"
            write_tone(voice, 2000)

            with (
                patch(
                    "videodub.tts.check_qwen_service",
                    return_value=QwenServiceInfo(True, "tts", "Qwen3-TTS", "mlx"),
                ) as health,
                patch(
                    "videodub.tts._synthesize_sentence_units",
                    return_value=raw,
                ) as synthesize,
                patch("videodub.tts._fit_sentence_durations", return_value=raw) as fit,
                patch("videodub.tts._render_sentence_track", return_value=voice) as render,
            ):
                result = dub_video(
                    config,
                    runner,
                    job,
                    qwen_base_url="http://tts-only",
                )

        health.assert_called_once_with("http://tts-only", "tts")
        synthesize.assert_called_once()
        fit.assert_called_once()
        render.assert_called_once()
        self.assertEqual(result.name, "Lecture.Chinese配音.m4a")
        self.assertNotIn("aligner_base_url", inspect.signature(dub_video).parameters)

    def test_tts_module_no_longer_exposes_forced_aligner_call(self) -> None:
        import videodub.tts as tts

        self.assertFalse(hasattr(tts, "align_qwen"))
        self.assertFalse(hasattr(tts, "_align_full_audio"))


if __name__ == "__main__":
    unittest.main()
