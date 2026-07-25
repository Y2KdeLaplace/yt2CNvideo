import tempfile
import unittest
from pathlib import Path

from videodub.subtitles import Cue, extract_json_array, parse_srt_text, write_srt


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


if __name__ == "__main__":
    unittest.main()
