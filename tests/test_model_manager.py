import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videodub.config import AppConfig
from videodub.model_manager import (
    ModelChoice,
    _huggingface_downloaded_bytes,
    _download_huggingface,
    _download_modelscope,
    group_gguf_files,
    install_model,
    list_installed_models,
    list_huggingface_gguf_options,
    model_choices,
    resolve_huggingface_model,
    resolve_modelscope_model,
    uv_runtime_prefix,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.command: list[str] = []
        self.environment: dict[str, str] | None = None
        self.logs: list[str] = []

    def logger(self, message: str) -> None:
        self.logs.append(message)

    def run(self, command, **kwargs) -> None:
        self.command = [str(item) for item in command]
        self.environment = kwargs.get("env")

    def check_cancelled(self) -> None:
        return


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
    def test_gguf_files_are_grouped_by_quantization_and_shards(self) -> None:
        options = group_gguf_files(
            [
                "model-q4_k.gguf",
                "model-q8_0.gguf",
                "large-q5_k_m-00002-of-00002.gguf",
                "large-q5_k_m-00001-of-00002.gguf",
                "README.md",
            ]
        )

        self.assertEqual(len(options), 3)
        split = next(item for item in options if len(item.files) == 2)
        self.assertIn("2 个分片", split.label)
        self.assertEqual(
            split.files,
            (
                "large-q5_k_m-00001-of-00002.gguf",
                "large-q5_k_m-00002-of-00002.gguf",
            ),
        )

    def test_huggingface_file_check_returns_only_gguf_options(self) -> None:
        class FakeResponse:
            headers: dict[str, str] = {}

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    [
                        {"type": "file", "path": "README.md"},
                        {"type": "file", "path": "model-q4_k.gguf"},
                        {"type": "file", "path": "model-q8_0.gguf"},
                    ]
                ).encode("utf-8")

        runner = RecordingRunner()
        with patch(
            "videodub.model_manager.urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            options = list_huggingface_gguf_options(
                "owner/model",
                runner,
            )

        self.assertEqual(
            [item.files for item in options],
            [("model-q4_k.gguf",), ("model-q8_0.gguf",)],
        )
        self.assertIn("正在检查 Hugging Face", runner.logs[0])

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

    def test_huggingface_download_progress_counts_blob_and_xet_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blob = root / "hub" / "models--owner--model" / "blobs" / "partial"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"1234")
            xet = root / "xet" / "chunk"
            xet.parent.mkdir(parents=True)
            xet.write_bytes(b"123456")
            with patch.dict(
                os.environ,
                {
                    "HF_HUB_CACHE": str(root / "hub"),
                    "HF_XET_CACHE": str(root / "xet"),
                },
            ):
                downloaded = _huggingface_downloaded_bytes("owner/model")

        self.assertEqual(downloaded, 10)

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
        self.assertEqual(runner.environment, {"HF_ENDPOINT": "https://huggingface.co"})

    def test_huggingface_download_retries_with_sufy_mirror(self) -> None:
        class RetryRunner(RecordingRunner):
            def __init__(self, cache: Path) -> None:
                super().__init__()
                self.cache = cache
                self.endpoints: list[str] = []

            def run(self, command, **kwargs) -> None:
                super().run(command, **kwargs)
                endpoint = (kwargs.get("env") or {}).get("HF_ENDPOINT", "")
                self.endpoints.append(endpoint)
                if endpoint == "https://huggingface.co":
                    raise RuntimeError("official endpoint unavailable")
                create_huggingface_snapshot(self.cache, "owner/model", "model.gguf")

        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"HF_HUB_CACHE": temp},
        ):
            runner = RetryRunner(Path(temp))
            actual = _download_huggingface("owner/model", runner)

        self.assertTrue(actual.name == "abc123")
        self.assertEqual(
            runner.endpoints,
            ["https://huggingface.co", "https://hf-cdn.sufy.com"],
        )

    def test_selected_custom_gguf_file_becomes_runtime_model_path(self) -> None:
        runner = RecordingRunner()
        choice = ModelChoice(
            "other",
            "其他 Hugging Face 模型",
            "",
            "hf",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = root / "model-q8_0.gguf"
            selected.write_text("weights", encoding="utf-8")
            with (
                patch("videodub.model_manager._install_runtime"),
                patch(
                    "videodub.model_manager._download_choice",
                    return_value=root,
                ),
            ):
                installed = install_model(
                    AppConfig(),
                    "asr",
                    choice,
                    "owner/model",
                    runner,
                    ("model-q8_0.gguf",),
                )

        self.assertEqual(installed.backend, "gguf")
        self.assertEqual(Path(installed.path), selected)

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
