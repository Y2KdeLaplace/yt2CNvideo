import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from videodub.config import AppConfig
from videodub.runner import ProcessRunner
from videodub.subtitles import Cue
from videodub.tts import _atempo_chain, _synthesize_edge


class TtsTests(unittest.TestCase):
    def test_atempo_chain_splits_large_factor(self) -> None:
        value = _atempo_chain(5.0)
        self.assertTrue(value.startswith("atempo=2.0,atempo=2.0,"))
        self.assertIn("atempo=1.250000", value)

    def test_edge_tts_uses_selected_voice_and_rate(self) -> None:
        calls: list[tuple[str, str, str]] = []

        class FakeCommunicate:
            def __init__(self, text: str, voice: str, *, rate: str):
                calls.append((text, voice, rate))

            async def save(self, output: str) -> None:
                Path(output).write_bytes(b"fake-mp3")

        fake_module = types.SimpleNamespace(Communicate=FakeCommunicate)
        config = AppConfig(
            tts_provider="edge",
            tts_voice="zh-CN-XiaoxiaoNeural",
            tts_rate=10,
        )
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"edge_tts": fake_module}
        ):
            outputs = _synthesize_edge(
                config,
                ProcessRunner(),
                [Cue(1, 0, 1000, "你好")],
                Path(temp),
            )
        self.assertEqual(
            calls,
            [("你好", "zh-CN-XiaoxiaoNeural", "+10%")],
        )
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
