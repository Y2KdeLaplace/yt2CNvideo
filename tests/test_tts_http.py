"""Service-boundary regressions; also run in the installed Qwen TTS runtime."""
import importlib.util
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("fastapi", "httpx", "numpy", "soundfile"))


@unittest.skipUnless(AVAILABLE, "Requires the optional Qwen service runtime")
class TTSHttpTests(unittest.TestCase):
    def client(self, model):
        from fastapi.testclient import TestClient
        from videodub.qwen_service import create_tts_app
        args = SimpleNamespace(backend="mlx", variant="base", model="test-local-model",
                               reference_audio="ref.wav", reference_text="ref", speaker="Vivian")
        module = ModuleType("mlx_audio.tts.utils")
        module.load_model = lambda _: model
        mocked = {"mlx_audio": ModuleType("mlx_audio"),
                  "mlx_audio.tts": ModuleType("mlx_audio.tts"),
                  "mlx_audio.tts.utils": module}
        self.patch = patch.dict(sys.modules, mocked)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        return TestClient(create_tts_app(args))

    def test_empty_iterator_returns_descriptive_http_error_without_asgi_exception(self):
        model = SimpleNamespace(generate=Mock(side_effect=lambda **kwargs: iter([])))
        with self.client(model) as client:
            response = client.post("/v1/tts", json={"text": "我永远无法习惯。"})
        self.assertEqual(response.status_code, 500)
        for text in ["未生成任何音频", "我永远无法习惯", "test-local-model", "mlx"]:
            self.assertIn(text, response.json()["detail"])
        self.assertNotIn("StopIteration", response.text)
        self.assertEqual(model.generate.call_count, 1)

    def test_partial_batch_keeps_valid_audio_and_marks_empty_peer(self):
        import numpy as np
        model = SimpleNamespace(generate=Mock(side_effect=[
            iter([SimpleNamespace(audio=np.ones(2400), sample_rate=24000)]), iter([])]))
        with self.client(model) as client:
            response = client.post("/v1/tts", json={"texts": ["一", "二"]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["audio_base64_list"][0])
        self.assertIsNone(data["audio_base64_list"][1])
        self.assertIn("未生成任何音频", data["errors"][1])

    def test_empty_audio_array_rejected_and_sound_never_calls_model(self):
        import numpy as np
        model = SimpleNamespace(generate=Mock(return_value=iter([SimpleNamespace(audio=np.array([]))])))
        with self.client(model) as client:
            response = client.post("/v1/tts", json={"text": "你好"})
            self.assertEqual(response.status_code, 500)
            for text in ["。", "[fly buzzing]", "[wrapper crinkles]", "”"]:
                response = client.post("/v1/tts", json={"text": text})
                self.assertEqual(response.status_code, 400)
        self.assertEqual(model.generate.call_count, 1)
