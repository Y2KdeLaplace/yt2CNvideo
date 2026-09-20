from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from videodub.config import AppConfig
from videodub.dependencies import (
    check_dependency_updates,
    inspect_dependency_versions,
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

    def test_macos_update_check_uses_homebrew_for_both_dependencies(self) -> None:
        runner = FakeRunner(
            {
                "/opt/homebrew/bin/brew": [
                    json.dumps(
                        {
                            "formulae": [
                                {"name": "ffmpeg", "current_version": "8.0"},
                                {"name": "yt-dlp", "current_version": "2026.09.20"},
                            ]
                        }
                    )
                ]
            }
        )
        with (
            patch(
                "videodub.dependencies.resolve_executable",
                return_value="/opt/homebrew/bin/brew",
            ),
            patch("videodub.dependencies.executable_exists", return_value=True),
        ):
            result = check_dependency_updates(runner, platform_name="darwin")  # type: ignore[arg-type]

        self.assertIn("ffmpeg", result)
        self.assertIn("yt-dlp", result)
        self.assertEqual(
            runner.commands[0],
            [
                "/opt/homebrew/bin/brew",
                "outdated",
                "--formula",
                "--json=v2",
                "yt-dlp",
                "ffmpeg",
            ],
        )

    def test_windows_update_check_lists_updates_without_installing(self) -> None:
        runner = FakeRunner({"winget.exe": ["yt-dlp yt-dlp.yt-dlp 1 2"]})
        with (
            patch("videodub.dependencies.resolve_executable", return_value="winget.exe"),
            patch("videodub.dependencies.executable_exists", return_value=True),
        ):
            result = check_dependency_updates(runner, platform_name="windows")  # type: ignore[arg-type]

        self.assertEqual(len(runner.commands), 1)
        self.assertNotIn("--id", runner.commands[0])
        self.assertEqual(runner.commands[0][1:3], ["list", "--upgrade-available"])
        self.assertNotIn("install", runner.commands[0])
        self.assertIn("yt-dlp", result)
        self.assertIn("winget", result)


if __name__ == "__main__":
    unittest.main()
