from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from videodub.speech.manager import SpeechModelManager
from videodub.speech.providers import build_download_command, check_provider_cli
from videodub.speech.registry import find_model_spec, get_model_spec
from videodub.speech.runtimes import create_runtime_adapter
from videodub.speech_client import check_speech_service
from videodub.dependencies import inspect_model_provider_dependency
from videodub.config import AppConfig
from videodub.model_management.backend import InstalledModel, repair_model_dependencies
from videodub.runner import ProcessRunner


class RegistryTests(unittest.TestCase):
    def test_qwen_asr_declares_forced_aligner_as_timestamp_dependency(self) -> None:
        spec = get_model_spec("qwen3-asr-mlx-0.6b-8bit")

        self.assertEqual(spec.kind, "asr")
        self.assertEqual(spec.source, "huggingface")
        self.assertEqual(spec.engine, "mlx")
        self.assertIn("transcription", spec.capabilities)
        self.assertEqual(len(spec.dependencies), 1)
        self.assertIn("timestamps", spec.dependencies[0].capabilities)
        self.assertIn("ForcedAligner", spec.dependencies[0].repo_id)

    def test_unknown_repository_is_not_supported(self) -> None:
        self.assertIsNone(find_model_spec("owner/random-safetensors"))

    def test_missing_aligner_only_removes_timestamp_capability(self) -> None:
        spec = get_model_spec("qwen3-asr-mlx-0.6b-8bit")
        manager = SpeechModelManager()
        with tempfile.TemporaryDirectory() as temp:
            status = manager.status(spec, temp, {})

        self.assertTrue(status.downloaded)
        self.assertFalse(status.dependency_complete)
        self.assertIn("transcription", status.available_capabilities)
        self.assertNotIn("timestamps", status.available_capabilities)
        self.assertEqual(status.missing_dependencies, ("qwen3-forced-aligner-mlx",))


class ProviderTests(unittest.TestCase):
    def test_provider_commands_use_preinstalled_cli(self) -> None:
        self.assertEqual(
            build_download_command("huggingface", "owner/model", ("*.gguf",)),
            ["hf", "download", "owner/model", "--include", "*.gguf"],
        )
        self.assertEqual(
            build_download_command("modelscope", "owner/model"),
            ["modelscope", "download", "--model", "owner/model"],
        )

    def test_cli_check_is_provider_specific(self) -> None:
        with patch("videodub.speech.providers.base.shutil.which", side_effect=lambda name: "/bin/hf" if name == "hf" else None):
            self.assertEqual(check_provider_cli("huggingface"), "/bin/hf")
            with self.assertRaisesRegex(RuntimeError, "uv tool install modelscope"):
                check_provider_cli("modelscope")

    def test_application_dependency_check_is_on_demand(self) -> None:
        with patch(
            "videodub.dependencies.check_provider_cli",
            return_value="/tools/hf",
        ) as check:
            self.assertEqual(
                inspect_model_provider_dependency("HuggingFace"),
                "/tools/hf",
            )
        check.assert_called_once_with("huggingface")


class RuntimeManagerTests(unittest.TestCase):
    def test_runtime_dispatch_uses_registry_runtime(self) -> None:
        adapter = create_runtime_adapter(get_model_spec("qwen3-tts-mlx-customvoice-0.6b-8bit"))
        self.assertEqual(adapter.__class__.__name__, "QwenTTSAdapter")

    def test_load_dispatch_and_unload_state(self) -> None:
        spec = get_model_spec("qwen3-tts-mlx-customvoice-0.6b-8bit")
        adapter = Mock(loaded=False)

        def load(*_args, **_kwargs):
            adapter.loaded = True

        def unload():
            adapter.loaded = False

        adapter.load.side_effect = load
        adapter.unload.side_effect = unload
        manager = SpeechModelManager()
        with tempfile.TemporaryDirectory() as temp, patch(
            "videodub.speech.manager.create_runtime_adapter",
            return_value=adapter,
        ) as factory:
            status = manager.load(spec.id, temp)
            self.assertTrue(status.loaded)
            manager.unload()

        factory.assert_called_once_with(spec)
        adapter.unload.assert_called_once_with()
        self.assertIsNone(manager.spec)

    def test_missing_companion_can_be_repaired_without_main_model_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "asr"
            aligner = root / "aligner"
            model.mkdir()
            aligner.mkdir()
            installed = InstalledModel(
                "asr",
                "mlx",
                "mlx-community/Qwen3-ASR-0.6B-8bit",
                str(model),
            )
            with (
                patch(
                    "videodub.model_management.backend._download_choice",
                    return_value=aligner,
                ) as download,
                patch("videodub.model_management.backend._record_installed_model"),
            ):
                repaired = repair_model_dependencies(
                    AppConfig(cache_dir=temp), installed, ProcessRunner()
                )

        self.assertEqual(repaired.path, installed.path)
        self.assertEqual(repaired.aligner_path, str(aligner))
        download.assert_called_once()


class ClientTests(unittest.TestCase):
    def test_health_check_reads_generic_speech_service(self) -> None:
        with patch(
            "videodub.speech_client._json_request",
            return_value={
                "status": "ok",
                "loaded": True,
                "model": "qwen3-asr-mlx-0.6b-8bit",
                "type": "asr",
                "runtime": "qwen3-asr-mlx",
                "pid": 123,
            },
        ):
            info = check_speech_service("http://127.0.0.1:9955")

        self.assertTrue(info.available)
        self.assertTrue(info.loaded)
        self.assertEqual(info.kind, "asr")
        self.assertEqual(info.pid, 123)


class APIRoutingTests(unittest.TestCase):
    def test_required_routes_are_registered(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from videodub.speech.api import create_speech_app

            app = create_speech_app(SpeechModelManager())
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"optional speech HTTP dependencies unavailable: {exc}")
        paths = {route.path for route in app.routes}
        self.assertTrue(
            {
                "/health",
                "/models",
                "/models/load",
                "/models/unload",
                "/models/download",
                "/models/{model_id}",
                "/audio/transcriptions",
                "/audio/speech",
            }.issubset(paths)
        )
        client = TestClient(app)
        self.assertEqual(client.get("/health").json()["status"], "ok")
        self.assertTrue(client.get("/models").json()["data"])
        self.assertEqual(client.post("/models/unload").json()["status"], "unloaded")


if __name__ == "__main__":
    unittest.main()
