import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videodub.config import (
    DEFAULT_WORK_DIR,
    PROJECT_ROOT,
    SETTINGS_FILE,
    AppConfig,
    api_key_from_runtime,
    configure_cache_directory,
    encrypt_api_key,
    load_language_model_info,
    load_config,
    migrate_work_directory,
    migrate_cache_directory,
    save_config,
    save_language_model_info,
)
from videodub.platform_utils import executable_exists, resolve_executable


class ConfigPortabilityTests(unittest.TestCase):
    def test_python_executable_resolution_is_platform_independent(self) -> None:
        resolved = resolve_executable(sys.executable, "python")
        self.assertTrue(Path(resolved).is_file())
        self.assertTrue(executable_exists(resolved))

    def test_default_tool_values_are_commands_not_windows_paths(self) -> None:
        config = AppConfig()
        self.assertEqual(config.yt_dlp_path, "yt-dlp")
        self.assertEqual(config.ffmpeg_path, "ffmpeg")
        self.assertEqual(config.ffprobe_path, "ffprobe")
        self.assertEqual(Path(config.work_dir).name, "work")
        self.assertEqual(Path(config.output_dir), Path(config.work_dir) / "output")
        self.assertTrue(config.overwrite)
        self.assertEqual(config.tts_speaker, "Vivian")
        self.assertEqual(config.asr_language, "English")
        self.assertEqual(config.translation_language, "Chinese")
        self.assertEqual(config.tts_language, "Chinese")
        self.assertEqual(config.tts_voice_preset, "")
        self.assertNotEqual(SETTINGS_FILE.parent, PROJECT_ROOT)

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

    def test_cache_directory_migration_moves_the_old_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "old-cache"
            target = root / "new-cache"
            (source / "runtimes").mkdir(parents=True)
            (source / "runtimes" / "crispasr").write_text("binary", encoding="utf-8")
            migrate_cache_directory(source, target)
            self.assertFalse(source.exists())
            self.assertEqual(
                (target / "runtimes" / "crispasr").read_text(encoding="utf-8"),
                "binary",
            )

    def test_configured_cache_directory_only_sets_the_application_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            root = configure_cache_directory(temp)
            self.assertEqual(Path(os.environ["VIDEODUB_CACHE_DIR"]), root)
            self.assertNotIn("HF_HUB_CACHE", os.environ)
            self.assertNotIn("MODELSCOPE_CACHE", os.environ)

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

    def test_language_model_info_is_encrypted_in_cache_and_moves_with_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_cache = root / "old-cache"
            new_cache = root / "new-cache"
            config = AppConfig(
                cache_dir=str(old_cache),
                subtitle_api_base_url="https://example.test/v1",
                subtitle_model="example-model",
                subtitle_api_key_encrypted=encrypt_api_key("secret-value"),
            )
            path = save_language_model_info(config)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotEqual(raw["subtitle_api_key_encrypted"], "secret-value")
            migrate_cache_directory(old_cache, new_cache)

            loaded = AppConfig(cache_dir=str(new_cache))
            load_language_model_info(loaded)

            self.assertEqual(loaded.subtitle_api_base_url, "https://example.test/v1")
            self.assertEqual(loaded.subtitle_model, "example-model")
            self.assertEqual(api_key_from_runtime(config=loaded), "secret-value")

    def test_saved_voice_preset_is_reused_and_first_use_is_empty(self) -> None:
        self.assertEqual(AppConfig().tts_voice_preset, "")
        with tempfile.TemporaryDirectory() as temp:
            settings = Path(temp) / "settings.json"
            save_config(AppConfig(tts_voice_preset="Diana"), settings)
            self.assertEqual(load_config(settings).tts_voice_preset, "Diana")


if __name__ == "__main__":
    unittest.main()
