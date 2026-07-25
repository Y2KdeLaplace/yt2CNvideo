import json
import sys
import tempfile
import unittest
from pathlib import Path

from videodub.config import AppConfig, load_config
from videodub.platform_utils import executable_exists, resolve_executable


class ConfigPortabilityTests(unittest.TestCase):
    def test_legacy_two_model_setting_migrates_to_one_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            settings = Path(temp) / "settings.json"
            settings.write_text(
                json.dumps(
                    {
                        "subtitle_text_model": "legacy-multimodal-model",
                        "subtitle_vision_model": "unused-vision-model",
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(settings)
            self.assertEqual(
                config.subtitle_model, "legacy-multimodal-model"
            )
            self.assertFalse(hasattr(config, "subtitle_vision_model"))

    def test_python_executable_resolution_is_platform_independent(self) -> None:
        resolved = resolve_executable(sys.executable, "python")
        self.assertTrue(Path(resolved).is_file())
        self.assertTrue(executable_exists(resolved))

    def test_default_tool_values_are_commands_not_windows_paths(self) -> None:
        config = AppConfig()
        self.assertEqual(config.yt_dlp_path, "yt-dlp")
        self.assertEqual(config.ffmpeg_path, "ffmpeg")
        self.assertEqual(config.ffprobe_path, "ffprobe")
        self.assertFalse(config.subtitle_use_vision)


if __name__ == "__main__":
    unittest.main()
