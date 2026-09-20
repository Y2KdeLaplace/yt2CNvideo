from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from videodub.model_management.voices import (
    import_voice_sample,
    list_voice_samples,
)


class RecordingRunner:
    def __init__(self, create_output: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.logs: list[str] = []
        self.create_output = create_output

    def logger(self, message: str) -> None:
        self.logs.append(message)

    def run(self, command, **_kwargs) -> None:
        values = [str(item) for item in command]
        self.commands.append(values)
        if self.create_output:
            Path(values[-1]).write_bytes(b"converted wav")


class VoiceSampleTests(unittest.TestCase):
    def test_wav_and_text_are_imported_into_a_new_sample_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            audio = source / "speaker.wav"
            with wave.open(str(audio), "wb") as output:
                output.setparams((1, 2, 24000, 2400, "NONE", ""))
                output.writeframes(b"\0" * 4800)
            text = source / "speaker.md"
            text.write_text("测试声音。", encoding="utf-8")
            samples = root / "samples"
            runner = RecordingRunner()

            imported = import_voice_sample(
                audio,
                text,
                "ffmpeg",
                runner,
                root=samples,
            )

            self.assertEqual(imported.name, "speaker")
            self.assertTrue(imported.audio_path.is_file())
            self.assertEqual(imported.text_path.read_text(encoding="utf-8"), "测试声音。\n")
            self.assertEqual(runner.commands, [])
            self.assertEqual(list_voice_samples(samples), [imported])

    def test_video_import_extracts_audio_with_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            media = root / "clip.mp4"
            media.write_bytes(b"video")
            text = root / "clip.txt"
            text.write_text("reference", encoding="utf-8")
            runner = RecordingRunner(create_output=True)

            imported = import_voice_sample(
                media,
                text,
                "custom-ffmpeg",
                runner,
                root=root / "samples",
            )

            self.assertTrue(imported.audio_path.is_file())
            command = runner.commands[0]
            self.assertEqual(command[0], "custom-ffmpeg")
            self.assertIn("-map", command)
            self.assertIn("0:a:0", command)
            self.assertIn("-vn", command)
            self.assertTrue(any("提取音频" in message for message in runner.logs))


if __name__ == "__main__":
    unittest.main()
