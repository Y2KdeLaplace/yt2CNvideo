import json
import sys
import tempfile
import unittest
from pathlib import Path

from videodub.config import AppConfig, load_config, migrate_work_directory
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
        self.assertEqual(Path(config.work_dir).name, "work")
        self.assertEqual(Path(config.output_dir), Path(config.work_dir) / "output")
        self.assertTrue(config.overwrite)

    def test_work_directory_migration_merges_and_overwrites_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "work"
            target = root / "chosen"
            (source / "output").mkdir(parents=True)
            target.mkdir()
            (source / "video.txt").write_text("new", encoding="utf-8")
            (target / "video.txt").write_text("old", encoding="utf-8")

            migrate_work_directory(source, target)

            self.assertFalse(source.exists())
            self.assertEqual(
                (target / "video.txt").read_text(encoding="utf-8"),
                "new",
            )
            self.assertTrue((target / "output").is_dir())


if __name__ == "__main__":
    unittest.main()
