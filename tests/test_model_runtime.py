from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from videodub.config import AppConfig
from videodub.model_manager import InstalledModel
from videodub.model_runtime import ManagedModelService, _terminate_process_tree
from videodub.qwen_speech import QwenServiceInfo
from videodub.runner import ProcessRunner


class ManagedModelServiceTests(unittest.TestCase):
    def test_aligner_service_uses_tts_companion_model(self) -> None:
        process = Mock()
        process.poll.return_value = None
        installed = InstalledModel(
            "tts",
            "mlx",
            "owner/tts",
            "/tts",
            aligner_path="/aligner",
        )
        checks = iter(
            (
                QwenServiceInfo(False, "aligner"),
                QwenServiceInfo(True, "aligner", "/aligner", "mlx"),
            )
        )
        runner = ProcessRunner()
        config = AppConfig(tts_backend="mlx", tts_model_path="/tts")

        with (
            patch(
                "videodub.model_runtime.read_installed_model",
                return_value=installed,
            ),
            patch(
                "videodub.model_runtime.check_qwen_service",
                side_effect=lambda *_args, **_kwargs: next(checks),
            ),
            patch("videodub.model_runtime.subprocess.Popen", return_value=process) as popen,
            patch("videodub.model_runtime.threading.Thread"),
            patch("videodub.model_runtime._terminate_process_tree"),
            ManagedModelService(config, runner, "aligner", port=13000),
        ):
            pass

        command = popen.call_args.args[0]
        self.assertIn("aligner", command)
        self.assertEqual(command[command.index("--model") + 1], "/aligner")

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
