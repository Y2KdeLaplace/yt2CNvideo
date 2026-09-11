from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

from .config import AppConfig
from .model_manager import read_installed_model, uv_runtime_prefix
from .qwen_speech import check_qwen_service, resolve_tts_reference
from .runner import ProcessRunner


RUNTIME_DIAGNOSTICS_FILENAME = "runtime-diagnostics.log"
RSS_SAMPLE_INTERVAL_SECONDS = 2.0
RSS_REPORT_GROWTH_KIB = 256 * 1024
LONG_REFERENCE_WARNING_SECONDS = 20.0
_DIAGNOSTICS_LOCK = threading.Lock()


def append_runtime_diagnostic(cache_dir: str | Path, message: str) -> None:
    try:
        path = Path(cache_dir).expanduser() / RUNTIME_DIAGNOSTICS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with _DIAGNOSTICS_LOCK, path.open("a", encoding="utf-8") as output:
            output.write(f"{timestamp} {message}\n")
    except Exception:
        pass


def _reference_audio_summary(path: str | Path, text: str) -> str | None:
    try:
        reference = Path(path)
        with wave.open(str(reference), "rb") as source:
            duration = source.getnframes() / source.getframerate()
        size_mb = reference.stat().st_size / 1_000_000
        return (
            f"TTS reference: {reference.name} duration={duration:.1f}s "
            f"size={size_mb:.1f}MB text_chars={len(text)}"
        )
    except (OSError, ValueError, ZeroDivisionError, wave.Error):
        return None


def _reference_audio_duration_seconds(path: str | Path) -> float | None:
    try:
        with wave.open(str(Path(path)), "rb") as source:
            return source.getnframes() / source.getframerate()
    except (OSError, ValueError, ZeroDivisionError, wave.Error):
        return None


def _read_process_rss_kib(pid: int) -> int | None:
    if os.name == "nt":
        return None
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
            timeout=1,
        )
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return int(value) if value else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)


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
        default_port = 9956 if kind == "asr" else 9955
        self.port = port or default_port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen[str] | None = None
        self.service_pid: int | None = None
        self._stop_lock = threading.Lock()
        self.current_rss_kib = 0
        self.peak_rss_kib = 0
        self._last_reported_peak_kib = 0
        self._rss_lock = threading.Lock()
        self._rss_stop = threading.Event()
        self._rss_thread: threading.Thread | None = None

    def __enter__(self) -> "ManagedModelService":
        selected_path = (
            self.config.asr_model_path
            if self.kind == "asr"
            else self.config.tts_model_path
        )
        if not selected_path:
            raise RuntimeError(f"请先在“模型”菜单下载并选择{self.kind.upper()}模型")
        installed = read_installed_model(selected_path)
        if installed is None:
            raise RuntimeError(f"模型未完整下载或已被移动：{selected_path}")
        path = selected_path
        backend = installed.backend
        if self.kind == "asr" and backend == "mlx" and not installed.aligner_path:
            raise RuntimeError(
                "Mac ASR 缺少 MLX Forced Aligner，请在模型菜单中重新下载该模型。"
            )
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
            reference_audio = self.config.tts_reference_audio
            reference_text = self.config.tts_reference_text
            if installed.variant == "base":
                reference_audio, reference_text = resolve_tts_reference(self.config)
            if summary := _reference_audio_summary(reference_audio, reference_text):
                self.runner.logger(summary)
                append_runtime_diagnostic(self.config.cache_dir, summary)
            reference_duration = _reference_audio_duration_seconds(reference_audio)
            if (
                backend == "mlx"
                and installed.variant == "base"
                and reference_duration is not None
                and reference_duration >= LONG_REFERENCE_WARNING_SECONDS
            ):
                warning = (
                    "警告：当前参考音频较长，MLX Base voice cloning 每次生成都需要使用"
                    "完整 reference context，可能增加统一内存占用和生成时间。"
                    "如出现内存压力，可使用更短且文本精确匹配的参考音频。"
                )
                self.runner.logger(warning)
                append_runtime_diagnostic(self.config.cache_dir, warning)
            command.extend(
                [
                    "--variant",
                    installed.variant or "custom_voice",
                    "--speaker",
                    self.config.tts_speaker,
                    "--reference-audio",
                    reference_audio,
                    "--reference-text",
                    reference_text,
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
            creationflags=(
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0
            ),
            start_new_session=os.name != "nt",
        )
        self.runner.add_cancel_callback(self._terminate)
        threading.Thread(target=self._relay_output, daemon=True).start()
        if self.kind == "tts":
            self.runner.logger(f"TTS model subprocess: pid={self.process.pid}")
            append_runtime_diagnostic(
                self.config.cache_dir,
                f"TTS model subprocess: pid={self.process.pid} service=tts",
            )
        try:
            for _ in range(180):
                self.runner.check_cancelled()
                if self.process.poll() is not None:
                    raise RuntimeError(f"{self.kind.upper()} 模型启动失败")
                info = check_qwen_service(self.base_url, self.kind, timeout=1)
                if info.available:
                    if self.kind == "tts":
                        self.service_pid = info.pid
                        if self.service_pid is not None and os.name != "nt":
                            message = f"TTS service process: pid={self.service_pid}"
                            self.runner.logger(message)
                            append_runtime_diagnostic(self.config.cache_dir, message)
                            self._rss_stop.clear()
                            self._rss_thread = threading.Thread(
                                target=self._sample_rss,
                                name="tts-rss-sampler",
                                daemon=True,
                            )
                            self._rss_thread.start()
                        elif self.service_pid is None:
                            message = (
                                "TTS service RSS unavailable: /health did not report pid"
                            )
                            self.runner.logger(message)
                            append_runtime_diagnostic(self.config.cache_dir, message)
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
                prefix = f"[{self.kind.upper()}]"
                message = line if line.startswith(prefix) else f"{prefix} {line}"
                self.runner.logger(message)
                if self.kind == "tts" and line.startswith("[TTS] MLX memory:"):
                    append_runtime_diagnostic(self.config.cache_dir, line)

    def _record_rss_sample(self, pid: int) -> None:
        rss_kib = _read_process_rss_kib(pid)
        if rss_kib is None:
            return
        with self._rss_lock:
            self.current_rss_kib = rss_kib
            self.peak_rss_kib = max(self.peak_rss_kib, rss_kib)
            peak_rss_kib = self.peak_rss_kib
            should_report = (
                self._last_reported_peak_kib == 0
                or peak_rss_kib - self._last_reported_peak_kib
                >= RSS_REPORT_GROWTH_KIB
            )
            if should_report:
                self._last_reported_peak_kib = peak_rss_kib
        if should_report:
            append_runtime_diagnostic(
                self.config.cache_dir,
                f"TTS service RSS: pid={pid} "
                f"current={rss_kib / 1024 / 1024:.2f} GB "
                f"peak={peak_rss_kib / 1024 / 1024:.2f} GB",
            )

    def _sample_rss(self) -> None:
        try:
            while not self._rss_stop.is_set():
                process = self.process
                if process is None or process.poll() is not None:
                    return
                service_pid = self.service_pid
                if service_pid is None:
                    return
                self._record_rss_sample(service_pid)
                if self._rss_stop.wait(RSS_SAMPLE_INTERVAL_SECONDS):
                    return
        except Exception:
            return

    def _stop_rss_sampler(self, process: subprocess.Popen[str] | None) -> None:
        if process is not None and process.poll() is None and self.service_pid is not None:
            self._record_rss_sample(self.service_pid)
        self._rss_stop.set()
        thread = self._rss_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._rss_thread = None
        with self._rss_lock:
            peak_rss_kib = self.peak_rss_kib
        if peak_rss_kib:
            message = f"TTS service RSS peak: {peak_rss_kib / 1024 / 1024:.2f} GB"
            self.runner.logger(message)
            append_runtime_diagnostic(self.config.cache_dir, message)

    def __exit__(self, *_args: object) -> None:
        self._terminate()

    def _terminate(self) -> None:
        self.runner.remove_cancel_callback(self._terminate)
        with self._stop_lock:
            process = self.process
            if process is None:
                return
            if self.kind == "tts":
                self._stop_rss_sampler(process)
            if process.poll() is not None:
                self.process = None
                return
            self.runner.logger(f"正在停止 {self.kind.upper()} 模型…")
            try:
                _terminate_process_tree(process)
            finally:
                self.process = None
