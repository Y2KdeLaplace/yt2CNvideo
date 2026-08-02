import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videodub.config import AppConfig
from videodub.media import VideoJob
from videodub.runner import ProcessRunner
from videodub.tts import _atempo_chain, dub_video


class TtsTests(unittest.TestCase):
    def test_atempo_chain_splits_large_factor(self) -> None:
        value = _atempo_chain(5.0)
        self.assertTrue(value.startswith("atempo=2.0,atempo=2.0,"))
        self.assertIn("atempo=1.250000", value)

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
                patch("videodub.tts._synthesize_qwen", return_value=[Path("raw.wav")]),
                patch("videodub.tts._prepare_timed_track", side_effect=prepare_track),
                patch("videodub.tts.media_duration", side_effect=AssertionError),
                patch("videodub.tts._mux_video", side_effect=AssertionError),
            ):
                result = dub_video(config, ProcessRunner(), job)

            self.assertEqual(result, output / "Lecture.Chinese配音.m4a")
            self.assertEqual(result.read_bytes(), b"audio")

if __name__ == "__main__":
    unittest.main()
