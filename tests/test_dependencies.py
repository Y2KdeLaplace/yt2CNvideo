from __future__ import annotations

import unittest
from unittest.mock import patch

from videodub.config import AppConfig
from videodub.dependencies import (
    inspect_dependency_versions,
    update_dependencies,
)


class FakeRunner:
    def __init__(self, outputs: dict[str, list[str]] | None = None) -> None:
        self.outputs = outputs or {}
        self.commands: list[list[str]] = []
        self.messages: list[str] = []

    def run(self, command, **_kwargs):
        values = [str(item) for item in command]
        self.commands.append(values)
        return self.outputs.get(values[0], [])

    def logger(self, message: str) -> None:
        self.messages.append(message)


class DependencyTests(unittest.TestCase):
    def test_startup_check_reads_actual_tool_versions(self) -> None:
        config = AppConfig(
            yt_dlp_path="custom-yt-dlp",
            ffmpeg_path="custom-ffmpeg",
            ffprobe_path="custom-ffprobe",
        )
        runner = FakeRunner(
            {
                "custom-yt-dlp": ["2026.09.20"],
                "custom-ffmpeg": ["ffmpeg version 8.0 Copyright"],
                "custom-ffprobe": ["ffprobe version 8.0 Copyright"],
            }
        )
        with patch.object(config, "validate_core", return_value=[]):
            versions = inspect_dependency_versions(config, runner)  # type: ignore[arg-type]

        self.assertEqual(versions.yt_dlp, "2026.09.20")
        self.assertEqual(versions.ffmpeg, "8.0")
        self.assertEqual(versions.ffprobe, "8.0")
        self.assertEqual(
            runner.commands,
            [
                ["custom-yt-dlp", "--version"],
                ["custom-ffmpeg", "-version"],
                ["custom-ffprobe", "-version"],
            ],
        )

    def test_macos_startup_update_uses_homebrew_for_both_dependencies(self) -> None:
        runner = FakeRunner()
        with (
            patch(
                "videodub.dependencies.resolve_executable",
                return_value="/opt/homebrew/bin/brew",
            ),
            patch("videodub.dependencies.executable_exists", return_value=True),
        ):
            result = update_dependencies(runner, platform_name="darwin")  # type: ignore[arg-type]

        self.assertIn("自动更新完成", result)
        self.assertEqual(
            runner.commands[0],
            [
                "/opt/homebrew/bin/brew",
                "upgrade",
                "--formula",
                "--no-ask",
                "ffmpeg",
                "yt-dlp",
            ],
        )

    def test_windows_startup_updates_both_dependencies_with_winget(self) -> None:
        runner = FakeRunner()
        with (
            patch("videodub.dependencies.resolve_executable", return_value="winget.exe"),
            patch("videodub.dependencies.executable_exists", return_value=True),
        ):
            result = update_dependencies(runner, platform_name="windows")  # type: ignore[arg-type]

        self.assertEqual(len(runner.commands), 2)
        self.assertEqual(runner.commands[0][1:5], ["upgrade", "--id", "Gyan.FFmpeg", "--exact"])
        self.assertEqual(runner.commands[1][1:5], ["upgrade", "--id", "yt-dlp.yt-dlp", "--exact"])
        for command in runner.commands:
            self.assertIn("--silent", command)
            self.assertIn("--disable-interactivity", command)
        self.assertIn("自动更新完成", result)


if __name__ == "__main__":
    unittest.main()
