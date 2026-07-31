import json
import sys
import tempfile
import unittest
from pathlib import Path

from videodub.config import (
    DEFAULT_WORK_DIR,
    PROJECT_ROOT,
    SETTINGS_FILE,
    AppConfig,
    api_key_from_runtime,
    encrypt_api_key,
    load_config,
    migrate_work_directory,
    save_config,
)
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
        self.assertEqual(config.tts_speaker, "Vivian")
        self.assertNotEqual(SETTINGS_FILE.parent, PROJECT_ROOT)

    def test_retired_tts_provider_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            settings = Path(temp) / "settings.json"
            settings.write_text(
                json.dumps(
                    {
                        "work_dir": temp,
                        "tts_provider": "retired-provider",
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(settings)

            self.assertFalse(hasattr(config, "tts_provider"))

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

            self.assertTrue(source.is_dir())
            self.assertEqual(list(source.iterdir()), [])
            self.assertEqual(
                (target / "video.txt").read_text(encoding="utf-8"),
                "new",
            )
            self.assertTrue((target / "output").is_dir())

    def test_saved_work_directory_persists_then_falls_back_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = root / "selected"
            selected.mkdir()
            settings = root / "settings.json"
            config = AppConfig(work_dir=str(selected))
            save_config(config, settings)

            self.assertEqual(Path(load_config(settings).work_dir), selected.resolve())
            (selected / "output").rmdir()
            selected.rmdir()
            self.assertEqual(
                Path(load_config(settings).work_dir),
                DEFAULT_WORK_DIR.resolve(),
            )

    def test_saved_api_key_is_encrypted_and_can_be_recovered(self) -> None:
        encrypted = encrypt_api_key("secret-value")
        self.assertNotIn("secret-value", encrypted)
        self.assertEqual(
            api_key_from_runtime(config=AppConfig(subtitle_api_key_encrypted=encrypted)),
            "secret-value",
        )


if __name__ == "__main__":
    unittest.main()
