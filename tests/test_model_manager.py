import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videodub.model_manager import (
    _download_huggingface,
    _download_modelscope,
    list_installed_models,
    model_choices,
    resolve_huggingface_model,
    resolve_modelscope_model,
    uv_runtime_prefix,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.command: list[str] = []
        self.logs: list[str] = []

    def logger(self, message: str) -> None:
        self.logs.append(message)

    def run(self, command) -> None:
        self.command = [str(item) for item in command]


def create_huggingface_snapshot(
    root: Path,
    repo_id: str,
    filename: str = "model.safetensors",
) -> Path:
    repository = root / ("models--" + repo_id.replace("/", "--"))
    revision = "abc123"
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text(revision, encoding="utf-8")
    snapshot = repository / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / filename).write_text("test", encoding="utf-8")
    return snapshot.resolve()


class ModelManagerTests(unittest.TestCase):
    def test_huggingface_cache_discovers_existing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"HF_HUB_CACHE": temp},
        ):
            expected = create_huggingface_snapshot(
                Path(temp),
                "mlx-community/Qwen3-ASR-0.6B-8bit",
            )

            actual = resolve_huggingface_model(
                "mlx-community/Qwen3-ASR-0.6B-8bit"
            )

        self.assertEqual(actual, expected)

    def test_incomplete_huggingface_snapshot_is_not_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"HF_HUB_CACHE": temp},
        ):
            create_huggingface_snapshot(
                Path(temp),
                "owner/incomplete",
                "config.json",
            )

            actual = resolve_huggingface_model("owner/incomplete")

        self.assertIsNone(actual)

    def test_huggingface_download_uses_official_cache_command(self) -> None:
        runner = RecordingRunner()
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"HF_HUB_CACHE": temp},
        ):
            expected = create_huggingface_snapshot(
                Path(temp),
                "owner/model",
                "model.gguf",
            )
            actual = _download_huggingface(
                "owner/model",
                runner,
                ("model.gguf",),
            )

        self.assertEqual(actual, expected)
        self.assertEqual(
            runner.command[:5],
            [
                "uvx",
                "--from",
                "huggingface-hub[hf_xet]",
                "hf",
                "download",
            ],
        )
        self.assertNotIn("--local-dir", runner.command)
        self.assertEqual(runner.command[-2:], ["--include", "model.gguf"])

    def test_modelscope_cache_discovers_official_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"MODELSCOPE_CACHE": temp},
        ):
            expected = (
                Path(temp)
                / "models"
                / "Qwen--Qwen3-ASR-0.6B"
                / "snapshots"
                / "master"
            )
            expected.mkdir(parents=True)
            (expected / "model.safetensors").write_text(
                "weights",
                encoding="utf-8",
            )

            actual = resolve_modelscope_model("Qwen/Qwen3-ASR-0.6B")

        self.assertEqual(actual, expected.resolve())

    def test_modelscope_download_uses_official_cache_command(self) -> None:
        runner = RecordingRunner()
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"MODELSCOPE_CACHE": temp},
        ):
            expected = (
                Path(temp)
                / "hub"
                / "models"
                / "Qwen"
                / "Qwen3-ASR-0.6B"
            )
            expected.mkdir(parents=True)
            (expected / "model.safetensors").write_text(
                "weights",
                encoding="utf-8",
            )

            actual = _download_modelscope(
                "Qwen/Qwen3-ASR-0.6B",
                runner,
            )

        self.assertEqual(actual, expected.resolve())
        self.assertEqual(
            runner.command,
            [
                "uvx",
                "--from",
                "modelscope-hub",
                "ms-hub",
                "download",
                "Qwen/Qwen3-ASR-0.6B",
            ],
        )

    def test_tts_choices_support_base_and_custom_voice(self) -> None:
        choices = model_choices("tts")
        variants = {choice.key for choice in choices}
        self.assertTrue(any("base" in key for key in variants))
        self.assertTrue(any("custom" in key for key in variants))

    def test_macos_discovers_existing_mlx_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"HF_HUB_CACHE": temp},
        ), patch(
            "videodub.model_manager.platform.system",
            return_value="Darwin",
        ), patch(
            "videodub.model_manager.platform.machine",
            return_value="arm64",
        ):
            asr = create_huggingface_snapshot(
                Path(temp),
                "mlx-community/Qwen3-ASR-0.6B-8bit",
            )
            base = create_huggingface_snapshot(
                Path(temp),
                "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
            )

            asr_models = list_installed_models("asr")
            tts_models = list_installed_models("tts")

        self.assertIn(str(asr), [item.path for item in asr_models])
        self.assertIn(str(base), [item.path for item in tts_models])
        self.assertEqual(tts_models[0].variant, "base")

    def test_runtime_is_managed_by_uv_without_project_environment(self) -> None:
        command = uv_runtime_prefix("asr", "hf")
        self.assertEqual(command[:3], ["uv", "run", "--no-project"])
        self.assertIn("--python", command)
        self.assertIn("qwen-asr", command)

if __name__ == "__main__":
    unittest.main()
