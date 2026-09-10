from __future__ import annotations

import unittest
import tempfile
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from videodub.config import AppConfig
from videodub.model_manager import InstalledModel
from videodub.model_runtime import (
    ManagedModelService,
    _read_process_rss_kib,
    _reference_audio_summary,
    _terminate_process_tree,
)
from videodub.qwen_speech import QwenServiceInfo
from videodub.runner import ProcessRunner


class ManagedModelServiceTests(unittest.TestCase):
    def test_reference_audio_summary_reports_duration_size_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "Diana.wav"
            with wave.open(str(path), "wb") as output:
                output.setparams((1, 2, 24000, 24000, "NONE", ""))
                output.writeframes(b"\0" * 48000)

            summary = _reference_audio_summary(path, "reference")

        self.assertIsNotNone(summary)
        self.assertIn("Diana.wav duration=1.0s", str(summary))
        self.assertIn("text_chars=9", str(summary))

    def test_rss_read_failure_is_best_effort(self) -> None:
        with patch(
            "videodub.model_runtime.subprocess.run",
            side_effect=OSError("ps unavailable"),
        ):
            self.assertIsNone(_read_process_rss_kib(123))

    def test_tts_rss_sampler_stops_during_service_termination(self) -> None:
        runner = ProcessRunner()
        service = ManagedModelService(
            AppConfig(cache_dir="/unused"),
            runner,
            "tts",
        )
        process = Mock(pid=321)
        process.poll.return_value = None
        sampler = Mock()
        service.process = process
        service._rss_thread = sampler

        with (
            patch("videodub.model_runtime._read_process_rss_kib", return_value=None),
            patch("videodub.model_runtime._terminate_process_tree") as terminate,
        ):
            service._terminate()

        self.assertTrue(service._rss_stop.is_set())
        sampler.join.assert_called_once_with(timeout=2)
        terminate.assert_called_once_with(process)
        self.assertIsNone(service.process)

    def test_windows_termination_kills_the_complete_process_tree(self) -> None:
        process = Mock(pid=321)
        process.poll.return_value = None
        with (
            patch("videodub.model_runtime.os.name", "nt"),
            patch(
                "videodub.model_runtime.subprocess.CREATE_NO_WINDOW",
                0x08000000,
                create=True,
            ),
            patch("videodub.model_runtime.subprocess.run") as run,
        ):
            _terminate_process_tree(process)

        self.assertEqual(
            run.call_args.args[0],
            ["taskkill", "/PID", "321", "/T", "/F"],
        )
        process.wait.assert_called_once_with(timeout=10)

    def test_runner_cancellation_terminates_managed_service(self) -> None:
        process = Mock()
        process.poll.return_value = None
        runner = ProcessRunner()
        installed = InstalledModel(
            "asr",
            "mlx",
            "owner/model",
            "/model",
            aligner_path="/aligner",
        )
        checks = iter(
            (
                QwenServiceInfo(False, "asr"),
                QwenServiceInfo(True, "asr", "/model", "mlx"),
            )
        )
        config = AppConfig(asr_backend="mlx", asr_model_path="/model")

        with (
            patch(
                "videodub.model_runtime.read_installed_model",
                return_value=installed,
            ),
            patch(
                "videodub.model_runtime.check_qwen_service",
                side_effect=lambda *_args, **_kwargs: next(checks),
            ),
            patch("videodub.model_runtime.subprocess.Popen", return_value=process),
            patch("videodub.model_runtime.threading.Thread"),
            patch("videodub.model_runtime._terminate_process_tree") as terminate,
            ManagedModelService(config, runner, "asr", port=12000),
        ):
            runner.cancel()

        terminate.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
