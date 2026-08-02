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
    _atempo_chain,
    _joined_speech_text,
    _source_ranges,
    _synthesize_qwen,
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
            cues = [
                Cue(1, 640, 2320, "嘿，宝贝，我回来了。"),
                Cue(2, 2480, 4080, "嘿，亲爱的，怎么样？"),
                Cue(3, 4400, 4960, "挺好的。"),
                Cue(4, 5120, 6160, "工作怎么样？"),
            ]
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
                    cues,
                    root,
                    "http://127.0.0.1:9955",
                )

        self.assertEqual(
            calls,
            ["嘿，宝贝，我回来了。嘿，亲爱的，怎么样？挺好的。工作怎么样？"],
        )
        self.assertEqual(output.name, "raw-full.wav")

    def test_joined_speech_text_keeps_western_words_separate(self) -> None:
        cues = [Cue(1, 0, 1000, "Hello"), Cue(2, 1000, 2000, "world.")]
        self.assertEqual(_joined_speech_text(cues), "Hello world.")

    def test_source_ranges_prefer_nearby_silence_at_text_boundaries(self) -> None:
        cues = [
            Cue(1, 0, 1000, "一二三四。"),
            Cue(2, 1000, 2000, "五六。"),
            Cue(3, 2000, 3000, "七八九十。"),
        ]
        ranges = _source_ranges(cues, 3000, [1200, 1780])
        self.assertEqual(ranges, [(0, 1200), (1200, 1780), (1780, 3000)])

    def test_wav_duration_is_read_without_starting_ffprobe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "voice.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(b"\0\0" * 12000)

            self.assertEqual(_wav_duration_ms(path), 500)

    def test_subtitle_only_job_generates_audio_without_muxing_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            config = AppConfig(work_dir=str(root), translation_language="Chinese")
            job = VideoJob(
                root / "Lecture.mp4",
                generated_dir=output,
                source_subtitle_path=root / "Lecture.en.srt",
            )
            job.chinese_subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\n你好。\n",
                encoding="utf-8",
            )

            def prepare_track(*args, **kwargs):
                work_dir = args[4]
                voice = work_dir / "voice.m4a"
                voice.write_bytes(b"audio")
                return voice

            with (
                patch("videodub.tts._synthesize_qwen", return_value=Path("raw.wav")),
                patch("videodub.tts._prepare_timed_track", side_effect=prepare_track),
                patch("videodub.tts.media_duration", side_effect=AssertionError),
                patch("videodub.tts._mux_video", side_effect=AssertionError),
            ):
                result = dub_video(config, ProcessRunner(), job)

            self.assertEqual(result, output / "Lecture.Chinese配音.m4a")
            self.assertEqual(result.read_bytes(), b"audio")

if __name__ == "__main__":
    unittest.main()
