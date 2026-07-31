import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import videodub.downloader as downloader_module
from videodub.config import AppConfig
from videodub.downloader import (
    build_download_command,
    cleanup_new_download_directories,
    snapshot_download_directories,
)
from videodub.media import VideoJob
from videodub.runner import CommandError


class DownloaderTests(unittest.TestCase):
    def test_failed_download_cleanup_only_removes_new_directories(self) -> None:
        class LogRunner:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def logger(self, message: str) -> None:
                self.messages.append(message)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = root / "existing"
            output = root / "output"
            existing.mkdir()
            output.mkdir()
            config = AppConfig(work_dir=str(root))
            before = snapshot_download_directories(config)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "part.webm").write_bytes(b"partial")
            runner = LogRunner()

            cleanup_new_download_directories(
                config,
                before,
                runner,  # type: ignore[arg-type]
            )

            self.assertTrue(existing.is_dir())
            self.assertTrue(output.is_dir())
            self.assertFalse(incomplete.exists())
            self.assertTrue(any("incomplete" in item for item in runner.messages))

    def test_single_video_command_disables_playlist(self) -> None:
        command = build_download_command(
            AppConfig(link_type="single"), "https://youtu.be/x"
        )
        self.assertIn("--no-playlist", command)
        self.assertNotIn("--yes-playlist", command)
        self.assertEqual(command[-1], "https://youtu.be/x")

    def test_playlist_command_has_index_template(self) -> None:
        command = build_download_command(
            AppConfig(link_type="playlist"),
            "https://youtube.com/playlist?list=x",
        )
        self.assertIn("--yes-playlist", command)
        self.assertIn("--no-write-playlist-metafiles", command)
        output = command[command.index("-o") + 1]
        self.assertIn("%(playlist_index)03d", output)

    def test_manual_subtitles_are_requested_before_automatic_subtitles(self) -> None:
        config = AppConfig(link_type="single")
        manual = build_download_command(config, "https://youtu.be/x")
        automatic = build_download_command(
            config,
            "https://youtu.be/x",
            automatic_subtitles=True,
            skip_download=True,
        )
        self.assertIn("--write-subs", manual)
        self.assertNotIn("--write-auto-subs", manual)
        self.assertIn("--write-auto-subs", automatic)
        self.assertNotIn("--write-subs", automatic)
        self.assertIn("--skip-download", automatic)

    def test_video_only_fallback_omits_all_subtitle_options(self) -> None:
        command = build_download_command(
            AppConfig(link_type="playlist"),
            "https://youtube.com/playlist?list=x",
            include_subtitles=False,
        )
        self.assertNotIn("--write-subs", command)
        self.assertNotIn("--write-auto-subs", command)
        self.assertNotIn("--sub-langs", command)
        self.assertIn("--ignore-errors", command)

    def test_subtitle_429_retries_without_subtitles(self) -> None:
        class FakeRunner:
            def __init__(self) -> None:
                self.commands: list[list[str]] = []
                self.messages: list[str] = []

            def run(self, command: list[str]) -> list[str]:
                self.commands.append(command)
                if len(self.commands) == 1:
                    raise CommandError(
                        command,
                        1,
                        "ERROR: Unable to download video subtitles for 'zh-Hans': "
                        "HTTP Error 429: Too Many Requests",
                    )
                return ["VIDEODUB_ID:abc123"]

            def logger(self, message: str) -> None:
                self.messages.append(message)

        runner = FakeRunner()
        job = VideoJob(Path("video [abc123].mp4"), video_id="abc123")
        with patch.object(
            downloader_module,
            "discover_video_jobs",
            return_value=[job],
        ):
            result = downloader_module.download(
                AppConfig(link_type="playlist"),
                runner,  # type: ignore[arg-type]
                "https://youtube.com/playlist?list=x",
            )
        self.assertEqual(result, [job])
        self.assertEqual(len(runner.commands), 3)
        self.assertIn("--write-subs", runner.commands[0])
        self.assertNotIn("--write-subs", runner.commands[1])
        self.assertIn("--write-auto-subs", runner.commands[2])
        self.assertIn("--skip-download", runner.commands[2])
        self.assertTrue(any("自动跳过字幕" in item for item in runner.messages))


if __name__ == "__main__":
    unittest.main()
