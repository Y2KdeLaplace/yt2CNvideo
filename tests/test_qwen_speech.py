import unittest

from videodub.qwen_speech import _segments_to_cues


class QwenSpeechTests(unittest.TestCase):
    def test_asr_segments_become_srt_cues(self) -> None:
        cues = _segments_to_cues(
            [
                {"text": "Hello", "start": 0.25, "end": 1.5},
                {"text": "world", "start": 1.5, "end": 2.0},
            ],
            "",
            3.0,
        )
        self.assertEqual([cue.index for cue in cues], [1, 2])
        self.assertEqual(cues[0].start_ms, 250)
        self.assertEqual(cues[1].end_ms, 2000)

    def test_asr_text_without_segments_uses_video_duration(self) -> None:
        cues = _segments_to_cues(None, "Fallback", 4.2)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "Fallback")
        self.assertEqual(cues[0].end_ms, 4200)


if __name__ == "__main__":
    unittest.main()
