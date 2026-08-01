import tempfile
import unittest
from pathlib import Path

from videodub.subtitles import (
    Cue,
    align_transcript_to_cues,
    extract_json_array,
    parse_srt_text,
    semantic_cues,
    subtitle_transcript,
    write_srt,
)


class SubtitleTests(unittest.TestCase):
    def test_parse_and_write_srt(self) -> None:
        content = """1
00:00:01,000 --> 00:00:02,500
Hello <i>world</i>

2
00:00:03.000 --> 00:00:04.100
Second line
"""
        cues = parse_srt_text(content)
        self.assertEqual(
            [(cue.start_ms, cue.end_ms, cue.text) for cue in cues],
            [(1000, 2500, "Hello world"), (3000, 4100, "Second line")],
        )
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "out.srt"
            write_srt(output, cues)
            self.assertIn(
                "00:00:01,000 --> 00:00:02,500",
                output.read_text(encoding="utf-8-sig"),
            )

    def test_extract_json_array_from_fence(self) -> None:
        data = extract_json_array(
            '说明\n```json\n[{"id": 1, "source_corrected": "Hi", "zh": "你好"}]\n```'
        )
        self.assertEqual(data[0]["zh"], "你好")

    def test_cue_duration_never_zero(self) -> None:
        self.assertEqual(Cue(1, 1000, 1000, "x").duration_ms, 1)

    def test_transcript_removes_timestamps_and_rolling_repeats(self) -> None:
        cues = [
            Cue(1, 0, 1000, "Hello brave"),
            Cue(2, 900, 2000, "brave new world."),
        ]
        self.assertEqual(subtitle_transcript(cues), "Hello brave new world.")

    def test_corrected_transcript_is_aligned_to_original_timestamps(self) -> None:
        source = [
            Cue(1, 100, 1000, "A speak train."),
            Cue(2, 1000, 2200, "Next sentence"),
        ]
        aligned = align_transcript_to_cues(
            source,
            "A spike train. Next sentence.",
        )
        self.assertEqual(len(aligned), 2)
        self.assertEqual((aligned[0].start_ms, aligned[0].end_ms), (100, 1000))
        self.assertEqual(aligned[0].text, "A spike train.")
        self.assertEqual(aligned[1].text, "Next sentence.")

    def test_semantic_cues_join_fragments_before_translation(self) -> None:
        grouped = semantic_cues(
            [
                Cue(1, 0, 800, "This is"),
                Cue(2, 700, 1600, "one sentence."),
                Cue(3, 1600, 2200, "Another one!"),
            ]
        )
        self.assertEqual([cue.text for cue in grouped], ["This is one sentence.", "Another one!"])
        self.assertEqual((grouped[0].start_ms, grouped[0].end_ms), (0, 1600))


if __name__ == "__main__":
    unittest.main()
