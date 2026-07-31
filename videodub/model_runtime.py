from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from .config import AppConfig
from .model_manager import read_installed_model, uv_runtime_prefix
from .qwen_speech import check_qwen_service
from .runner import ProcessRunner


class ManagedModelService:
    """Start one locally installed model for one task and always stop it."""

    def __init__(
        self,
        config: AppConfig,
        runner: ProcessRunner,
        kind: str,
        *,
        port: int | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.kind = kind
        self.port = port or (9956 if kind == "asr" else 9955)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen[str] | None = None
        self._stop_lock = threading.Lock()

    def __enter__(self) -> "ManagedModelService":
        path = self.config.asr_model_path if self.kind == "asr" else self.config.tts_model_path
        backend = self.config.asr_backend if self.kind == "asr" else self.config.tts_backend
        if not path:
            raise RuntimeError(f"请先在“模型”菜单下载并选择{self.kind.upper()}模型")
        installed = read_installed_model(path)
        if installed is None:
            raise RuntimeError(f"模型未完整下载或已被移动：{path}")
        if backend == "gguf":
            return self
        existing = check_qwen_service(self.base_url, self.kind, timeout=1)
        if existing.available:
            raise RuntimeError(
                f"端口 {self.port} 已有 {self.kind.upper()} 服务运行，请先关闭后重试"
            )
        command = [
            *uv_runtime_prefix(self.kind, backend),
            "python",
            "-m",
            "videodub.qwen_service",
            self.kind,
            "--backend",
            backend,
            "--model",
            path,
            "--port",
            str(self.port),
        ]
        if self.kind == "asr" and installed.aligner_path:
            command.extend(["--aligner", installed.aligner_path])
        if self.kind == "tts":
            command.extend(
                [
                    "--variant",
                    installed.variant or "custom_voice",
                    "--speaker",
                    self.config.tts_speaker,
                    "--reference-audio",
                    self.config.tts_reference_audio,
                    "--reference-text",
                    self.config.tts_reference_text,
                ]
            )
        self.runner.logger(f"正在启动 {self.kind.upper()} 模型…")
        self.process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.runner.add_cancel_callback(self._terminate)
        threading.Thread(target=self._relay_output, daemon=True).start()
        try:
            for _ in range(180):
                self.runner.check_cancelled()
                if self.process.poll() is not None:
                    raise RuntimeError(f"{self.kind.upper()} 模型启动失败")
                info = check_qwen_service(self.base_url, self.kind, timeout=1)
                if info.available:
                    self.runner.logger(f"{self.kind.upper()} 模型已就绪：{info.model}")
                    return self
                time.sleep(1)
            raise RuntimeError(f"{self.kind.upper()} 模型启动超时")
        except Exception:
            self._terminate()
            raise

    def _relay_output(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            line = line.rstrip()
            if line:
                self.runner.logger(f"[{self.kind.upper()}] {line}")

    def __exit__(self, *_args: object) -> None:
        self._terminate()

    def _terminate(self) -> None:
        self.runner.remove_cancel_callback(self._terminate)
        with self._stop_lock:
            process = self.process
            if process is None or process.poll() is not None:
                return
            self.runner.logger(f"正在停止 {self.kind.upper()} 模型…")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            finally:
                self.process = None
