import tempfile
import unittest
from pathlib import Path

from videodub.media import discover_video_jobs


class MediaDiscoveryTests(unittest.TestCase):
    def test_downloaded_subtitle_without_video_becomes_a_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            source = root / "Lecture.en-orig.srt"
            source.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nHello.\n",
                encoding="utf-8",
            )

            jobs = discover_video_jobs(root, output)

            self.assertEqual(len(jobs), 1)
            self.assertFalse(jobs[0].has_video)
            self.assertEqual(jobs[0].source_subtitle_path, source)
            self.assertEqual(jobs[0].title, "Lecture")
            self.assertEqual(
                jobs[0].translated_subtitle_path("German").resolve(),
                (output / "Lecture.translated.de.srt").resolve(),
            )
            self.assertEqual(
                jobs[0].translated_transcript_path("German").resolve(),
                (output / "Lecture.translated.de.txt").resolve(),
            )

    def test_source_subtitle_is_not_duplicated_when_video_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "Lecture.mp4"
            video.touch()
            (root / "Lecture.en.srt").write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nHello.\n",
                encoding="utf-8",
            )

            jobs = discover_video_jobs(root)

            self.assertEqual(len(jobs), 1)
            self.assertTrue(jobs[0].has_video)


if __name__ == "__main__":
    unittest.main()
