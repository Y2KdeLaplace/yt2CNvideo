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
from videodub.sentences import build_sentence_units, write_units
from videodub.tts import (
    BLOCK_DURATION_RATIO_MIN,
    CROSSFADE_MS,
    SENTENCE_EDGE_FADE_MS,
    SentenceAudio,
    SentenceUnit,
    _adjust_sentence_audio,
    _atempo_chain,
    _audio_duration_ms,
    _build_speech_blocks,
    _fit_speech_blocks,
    _render_sentence_track,
    _required_block_duration_ratio,
    _scheduled_sentence_start,
    _smooth_block_ratios,
    _synthesize_sentence,
    _synthesize_sentence_units,
    _tts_batch_size,
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
    def test_tts_batch_size_is_backend_aware(self) -> None:
        self.assertEqual(_tts_batch_size(AppConfig(tts_backend="mlx")), 1)
        self.assertEqual(_tts_batch_size(AppConfig(tts_backend="hf")), 2)

    def test_natural_sentence_units_keep_cue_and_timeline_boundaries(self) -> None:
        units = build_sentence_units(
            [
                Cue(1, 100, 700, "这是"),
                Cue(2, 800, 1400, "第一句。"),
                Cue(3, 2000, 2600, "第二句！"),
            ]
        )

        self.assertEqual(
            units,
            [
                SentenceUnit(1, 2, 100, 1400, "这是第一句。", 1),
                SentenceUnit(3, 3, 2000, 2600, "第二句！", 2),
            ],
        )

    def test_long_sentence_chunks_still_produce_one_sentence_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            unit = SentenceUnit(1, 3, 0, 3000, "第一段。第二段。第三段。")

            requests: list[str] = []

            def synthesize(_config, text, output, _runner, **_kwargs):
                requests.append(text)
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
                    AppConfig(tts_backend="mlx", cache_dir=str(root / "cache")),
                    AudioRunner(),
                    unit,
                    0,
                    root,
                    "http://tts",
                )

            self.assertEqual(result.unit, unit)
            self.assertEqual(result.path.name, "sentence-00000.raw.wav")
            self.assertTrue(result.path.is_file())
            self.assertEqual(result.duration_ms, 480)
            self.assertEqual(requests, ["第一段。", "第二段。", "第三段。"])

    def test_mlx_sentence_units_are_generated_sequentially(self) -> None:
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
            single_calls: list[str] = []
            batch_calls: list[list[str]] = []

            def synthesize(_config, text, output, _runner, **_kwargs):
                single_calls.append(text)
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
                    AppConfig(tts_backend="mlx", cache_dir=str(root / "cache")),
                    AudioRunner(),
                    units,
                    root,
                    "http://tts",
                )

        self.assertEqual(single_calls, ["第一句。", "第二句。", "第三句。"])
        self.assertEqual(batch_calls, [])
        self.assertEqual(
            [item.path.name for item in results],
            [
                "sentence-00000.raw.wav",
                "sentence-00001.raw.wav",
                "sentence-00002.raw.wav",
            ],
        )
        self.assertEqual([item.unit for item in results], units)

    def test_non_mlx_sentence_units_keep_default_batch_size_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            units = [
                SentenceUnit(i + 1, i + 1, i * 1000, i * 1000 + 800, text)
                for i, text in enumerate(("一。", "二。", "三。"))
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
                patch("videodub.tts.synthesize_qwen_batch", side_effect=synthesize_batch),
            ):
                _synthesize_sentence_units(
                    AppConfig(tts_backend="hf", cache_dir=str(root / "cache")),
                    AudioRunner(),
                    units,
                    root,
                    "http://tts",
                )

        self.assertEqual(batch_calls, [["一。", "二。"]])


class DurationFittingTests(unittest.TestCase):
    def test_short_sentence_uses_following_block_slack_without_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = [
                sentence_audio(root, 0, start_ms=0, end_ms=600, raw_ms=2500),
                sentence_audio(root, 1, start_ms=4000, end_ms=5000, raw_ms=700),
            ]
            runner = AudioRunner()

            blocks = _fit_speech_blocks(AppConfig(), runner, raw, root, 6000)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].deadline_ms, 4000)
        self.assertEqual(blocks[0].duration_ratio, 1.0)
        self.assertGreater(blocks[0].sentences[0].duration_ms, 2400)

    def test_rapid_dialogue_forms_one_block_with_one_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = [
                sentence_audio(root, 0, start_ms=0, end_ms=500, raw_ms=1000),
                sentence_audio(root, 1, start_ms=580, end_ms=1000, raw_ms=1000),
                sentence_audio(root, 2, start_ms=1120, end_ms=1500, raw_ms=1000),
            ]
            runner = AudioRunner()

            blocks = _fit_speech_blocks(AppConfig(), runner, raw, root, 2800)

        self.assertEqual(len(blocks), 1)
        self.assertAlmostEqual(blocks[0].duration_ratio, 0.92)
        self.assertEqual(
            {sentence.duration_ratio for sentence in blocks[0].sentences},
            {blocks[0].duration_ratio},
        )

    def test_gap_at_break_threshold_creates_new_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentences = [
                sentence_audio(root, 0, start_ms=0, end_ms=500, raw_ms=400),
                sentence_audio(root, 1, start_ms=3000, end_ms=3500, raw_ms=400),
            ]
            blocks = _build_speech_blocks(sentences, 4000)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].deadline_ms, 3000)

    def test_block_that_fits_keeps_natural_speed_and_preferred_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentences = [
                sentence_audio(root, 0, start_ms=0, end_ms=500, raw_ms=400),
                sentence_audio(root, 1, start_ms=580, end_ms=1000, raw_ms=400),
            ]
            block = _fit_speech_blocks(
                AppConfig(), AudioRunner(), sentences, root, 1500
            )[0]

        self.assertEqual(block.duration_ratio, 1.0)
        self.assertEqual(block.scheduled_gaps_ms, (80,))

    def test_mild_compression_is_uniform_across_the_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentences = [
                sentence_audio(root, 0, start_ms=0, end_ms=500, raw_ms=1000),
                sentence_audio(root, 1, start_ms=580, end_ms=1000, raw_ms=1000),
            ]
            block = _fit_speech_blocks(
                AppConfig(), AudioRunner(), sentences, root, 1880
            )[0]

        self.assertAlmostEqual(block.required_duration_ratio, 0.93)
        self.assertAlmostEqual(block.duration_ratio, 0.93)
        self.assertEqual(
            [sentence.duration_ratio for sentence in block.sentences],
            [block.duration_ratio, block.duration_ratio],
        )

    def test_gaps_shrink_before_speech_speed_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentences = [
                sentence_audio(root, 0, start_ms=0, end_ms=500, raw_ms=800),
                sentence_audio(root, 1, start_ms=580, end_ms=1000, raw_ms=800),
            ]
            block = _fit_speech_blocks(
                AppConfig(), AudioRunner(), sentences, root, 1650
            )[0]

        self.assertEqual(block.duration_ratio, 1.0)
        self.assertEqual(block.scheduled_gaps_ms, (50,))

    def test_smoothing_only_adjusts_already_compressed_blocks(self) -> None:
        self.assertEqual(_smooth_block_ratios([0.82, 0.93, 0.82]), [0.82, 0.90, 0.82])
        self.assertEqual(_smooth_block_ratios([0.82, 1.0, 0.82]), [0.82, 1.0, 0.82])

    def test_ratio_floor_allows_spill_instead_of_extreme_speed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentences = [
                sentence_audio(root, 0, start_ms=0, end_ms=500, raw_ms=1000),
                sentence_audio(root, 1, start_ms=580, end_ms=1000, raw_ms=1000),
            ]
            raw_block = _build_speech_blocks(sentences, 1320)[0]
            runner = AudioRunner()
            block = _fit_speech_blocks(AppConfig(), runner, sentences, root, 1320)[0]

        self.assertAlmostEqual(_required_block_duration_ratio(raw_block), 0.65)
        self.assertEqual(block.duration_ratio, BLOCK_DURATION_RATIO_MIN)
        self.assertGreater(block.spill_ms, 0)
        self.assertLessEqual(1 / block.duration_ratio, 1.22)
        self.assertTrue(any("警告：TTS block" in message for message in runner.messages))

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
                0.82,
                root,
            )

        self.assertEqual(fitted.path.name, "sentence-00000.fit.wav")
        self.assertLess(abs(fitted.duration_ms - 1148), 50)


class TimelineSchedulingTests(unittest.TestCase):
    def test_overlapping_subtitle_windows_are_scheduled_serially(self) -> None:
        first = SentenceUnit(1, 1, 0, 1000, "一。")
        second = SentenceUnit(2, 2, 800, 1600, "二。")
        first_start = _scheduled_sentence_start(first, 0)
        second_start = _scheduled_sentence_start(second, first_start + 900)

        self.assertEqual(first_start, 0)
        self.assertGreaterEqual(second_start, first_start + 900)

    def test_original_gap_is_preserved_when_space_allows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentences = [
                sentence_audio(root, 0, start_ms=0, end_ms=500, raw_ms=500),
                sentence_audio(root, 1, start_ms=800, end_ms=1300, raw_ms=500),
            ]
            runner = AudioRunner()
            blocks = _fit_speech_blocks(AppConfig(), runner, sentences, root, 1800)
            output = _render_sentence_track(runner, blocks, root, 1800)
            with output.open("rb") as source:
                track = AudioSegment.from_wav(source)

        self.assertEqual(blocks[0].scheduled_gaps_ms, (300,))
        self.assertEqual(track[550:750].rms, 0)

    def test_sentence_edge_fade_is_shorter_than_chunk_crossfade(self) -> None:
        self.assertEqual(SENTENCE_EDGE_FADE_MS, 5)
        self.assertGreater(CROSSFADE_MS, SENTENCE_EDGE_FADE_MS)

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
            blocks = _fit_speech_blocks(
                AppConfig(), runner, [first, last], root, 2000
            )
            output = _render_sentence_track(runner, blocks, root, 2000)

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
            write_units(subtitle, build_sentence_units([Cue(1, 0, 900, "第一句。"), Cue(2, 1200, 2000, "第二句。")]))
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
                patch("videodub.tts._fit_speech_blocks", return_value=[]) as fit,
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
        self.assertFalse(hasattr(tts, "_global_duration_factor"))
        self.assertFalse(hasattr(tts, "_local_duration_factor"))

class SentenceCacheTests(unittest.TestCase):
    def test_first_empty_then_success_and_two_empty_fail_with_context(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from videodub.qwen_service import _generate_tts_one
        args = SimpleNamespace(backend="mlx", variant="base", reference_audio="ref.wav",
                               reference_text="ref", speaker="Vivian")
        for succeeds in [True, False]:
            with self.subTest(succeeds=succeeds), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                model = SimpleNamespace(generate=Mock(side_effect=[iter([]), iter([SimpleNamespace(audio="good")]) if succeeds else iter([])]))
                def request(config, text, output, runner, **kwargs):
                    _generate_tts_one(model, args, text, "Chinese")
                    write_tone(output, 200)
                units = [SentenceUnit(87, 88, 0, 5920, "我永远无法习惯那些临终挣扎。", 1)]
                config = AppConfig(tts_backend="mlx", tts_model_id="model-test", cache_dir=str(root / "cache"))
                with patch("videodub.tts.synthesize_qwen", side_effect=request):
                    if succeeds:
                        self.assertEqual(len(_synthesize_sentence_units(config, AudioRunner(), units, root, "http://tts")), 1)
                    else:
                        with self.assertRaises(RuntimeError) as error:
                            _synthesize_sentence_units(config, AudioRunner(), units, root, "http://tts")
                        for detail in ["1/1", "87–88", units[0].text, "model-test", "mlx", "连续 2 次", "未生成任何音频"]:
                            self.assertIn(detail, str(error.exception))
                self.assertEqual(model.generate.call_count, 2)

    def test_fifty_cached_sentences_survive_51_failure_and_fresh_work_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work = root / "tmp"
            work.mkdir()
            config = AppConfig(tts_backend="gguf", cache_dir=str(root / "cache"))
            units = [SentenceUnit(i+1, i+1, i*1000, i*1000+800, f"句子{i+1}。", i+1) for i in range(56)]
            calls = []
            fail = True
            def request(config, text, output, runner, **kwargs):
                calls.append(text)
                if text == "句子51。" and fail:
                    raise RuntimeError("empty")
                write_tone(output, 150)
            with patch("videodub.tts.synthesize_qwen", side_effect=request):
                with self.assertRaisesRegex(RuntimeError, "51/56"):
                    _synthesize_sentence_units(config, AudioRunner(), units, work, "http://tts")
                self.assertEqual(len(calls), 52)
                shutil.rmtree(work)
                work.mkdir()
                calls.clear()
                fail = False
                results = _synthesize_sentence_units(config, AudioRunner(), units, work, "http://tts")
            self.assertEqual(len(results), 56)
            self.assertEqual(calls, [f"句子{i}。" for i in range(51,57)])

    def test_partial_batch_success_cached_even_when_peer_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = AppConfig(tts_backend="hf", cache_dir=str(root / "cache"))
            units = [SentenceUnit(1, 1, 0, 900, "第一句。", 1), SentenceUnit(2, 2, 1000, 1900, "第二句。", 2)]
            def batch(config, texts, outputs, runner, **kwargs):
                write_tone(outputs[0], 150)
                raise RuntimeError("second empty")
            with patch("videodub.tts.synthesize_qwen_batch", side_effect=batch), patch("videodub.tts.synthesize_qwen", side_effect=RuntimeError("empty")):
                with self.assertRaises(RuntimeError):
                    _synthesize_sentence_units(config, AudioRunner(), units, root, "http://tts")
            def success(config, text, output, runner, **kwargs):
                self.assertEqual(text, "第二句。")
                write_tone(output, 150)
            with patch("videodub.tts.synthesize_qwen", side_effect=success) as request, patch("videodub.tts.synthesize_qwen_batch") as batch_request:
                _synthesize_sentence_units(config, AudioRunner(), units, root, "http://tts")
                request.assert_called_once()
                batch_request.assert_not_called()

    def test_cache_invalidates_model_language_speaker_and_reference_content(self):
        from dataclasses import replace
        from videodub.tts import _cache_key, _voice_cache_identity
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "ref.wav"
            text = root / "ref.txt"
            write_tone(audio, 150)
            text.write_text("reference", encoding="utf-8")
            config = AppConfig(tts_backend="mlx", tts_model_id="m1", tts_use_custom_voice=True,
                               tts_reference_audio=str(audio), tts_reference_text_file=str(text))
            def key(c): return _cache_key(_voice_cache_identity(c), "你好")
            original = key(config)
            for change in [dict(tts_model_id="m2"), dict(tts_backend="qwen"), dict(tts_language="English"), dict(tts_speaker="other")]:
                self.assertNotEqual(key(replace(config, **change)), original)
            text.write_text("new reference", encoding="utf-8")
            self.assertNotEqual(key(config), original)
            before_audio = key(config)
            write_tone(audio, 200)
            self.assertNotEqual(key(config), before_audio)

    def test_header_only_audio_is_retried_and_not_cached(self):
        import wave
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = AppConfig(tts_backend="mlx", cache_dir=str(root / "cache"))
            def empty(config, text, output, runner, **kwargs):
                with wave.open(str(output), "wb") as wav:
                    wav.setparams((1, 2, 24000, 0, "NONE", ""))
            with patch("videodub.tts.synthesize_qwen", side_effect=empty) as request:
                with self.assertRaisesRegex(RuntimeError, "连续 2 次"):
                    _synthesize_sentence_units(config, AudioRunner(), [SentenceUnit(1,1,0,1000,"你好",1)], root, "http://tts")
                self.assertEqual(request.call_count, 2)
            self.assertEqual(list((root / "cache" / "tts-sentences-v2").glob("*.wav")), [])


if __name__ == "__main__":
    unittest.main()
