import importlib.util
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("fastapi", "httpx", "multipart"))


@unittest.skipUnless(AVAILABLE, "Requires the optional ASR service runtime")
class ASRHttpTests(unittest.TestCase):
    def test_bad_alignment_is_readable_validation_error_not_unhandled_500(self):
        from fastapi.testclient import TestClient
        from videodub.qwen_service import create_asr_app
        word = SimpleNamespace(text="You", start_time=1.232, end_time=1.232)
        model = SimpleNamespace(generate=lambda *a, **k: {"text": "You"})
        aligner = SimpleNamespace(generate=lambda *a, **k: SimpleNamespace(items=[word]))
        module = ModuleType("mlx_audio.stt.utils")
        module.load_model = lambda path: model if path == "model" else aligner
        mocked = {"mlx_audio": ModuleType("mlx_audio"),
                  "mlx_audio.stt": ModuleType("mlx_audio.stt"),
                  "mlx_audio.stt.utils": module}
        args = SimpleNamespace(backend="mlx", model="model", aligner="aligner")
        with patch.dict(sys.modules, mocked), patch(
            "videodub.qwen_service._mlx_audio_chunks", return_value=[("audio", 0.0)]
        ), TestClient(create_asr_app(args)) as client:
            response = client.post("/v1/asr", files={"audio": ("a.wav", b"stub")})
        self.assertEqual(response.status_code, 422)
        self.assertIn("You", response.json()["detail"])
        self.assertIn("zero_duration_word", response.json()["detail"])
        self.assertNotIn("Internal Server Error", response.text)
