import importlib.util
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("fastapi", "httpx", "multipart"))


@unittest.skipUnless(AVAILABLE, "Requires the optional ASR service runtime")
class ASRHttpTests(unittest.TestCase):
    def _post_alignment(self, text, words):
        from fastapi.testclient import TestClient
        from videodub.qwen_service import create_asr_app
        model = SimpleNamespace(generate=lambda *a, **k: {"text": text})
        aligner = SimpleNamespace(
            generate=lambda *a, **k: SimpleNamespace(items=words)
        )
        module = ModuleType("mlx_audio.stt.utils")
        module.load_model = lambda path: model if path == "model" else aligner
        mocked = {"mlx_audio": ModuleType("mlx_audio"),
                  "mlx_audio.stt": ModuleType("mlx_audio.stt"),
                  "mlx_audio.stt.utils": module}
        args = SimpleNamespace(backend="mlx", model="model", aligner="aligner")
        audio_chunk = bytes(20 * 16000)
        with patch.dict(sys.modules, mocked), patch(
            "videodub.qwen_service._mlx_audio_chunks",
            return_value=[(audio_chunk, 0.0)],
        ), patch(
            "videodub.qwen_service._split_mlx_audio",
            side_effect=lambda audio, _duration: [(audio, 0.0)],
        ), TestClient(create_asr_app(args)) as client:
            return client.post("/v1/asr", files={"audio": ("a.wav", b"stub")})

    def test_zero_duration_word_inside_spoken_segment_is_accepted(self):
        response = self._post_alignment(
            "I'm not ready yet.",
            [
                SimpleNamespace(text="I'm", start_time=10.0, end_time=10.08),
                SimpleNamespace(text="not", start_time=10.08, end_time=10.18),
                SimpleNamespace(text="ready", start_time=10.18, end_time=10.3),
                SimpleNamespace(text="yet", start_time=10.32, end_time=10.32),
            ],
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["segments"],
            [{"text": "I'm not ready yet.", "start": 10.0, "end": 10.32}],
        )

    def test_invalid_alignments_are_readable_422_not_unhandled_500(self):
        cases = [
            (
                "You",
                [SimpleNamespace(text="You", start_time=1.3, end_time=1.2)],
            ),
            (
                "One two",
                [
                    SimpleNamespace(text="One", start_time=0.14, end_time=0.15),
                    SimpleNamespace(text="two", start_time=0.13, end_time=0.3),
                ],
            ),
        ]
        for text, words in cases:
            with self.subTest(text=text):
                response = self._post_alignment(text, words)
                self.assertEqual(response.status_code, 422)
                self.assertIn(words[-1].text, response.json()["detail"])
                self.assertIn(
                    "invalid_or_non_monotonic_timeline",
                    response.json()["detail"],
                )
                self.assertNotIn("Internal Server Error", response.text)
