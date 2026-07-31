from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import AppConfig, EXTERNAL_DIR
from .runner import ProcessRunner


MODELS_DIR = EXTERNAL_DIR / "models"
ENVS_DIR = EXTERNAL_DIR / "envs"
RUNTIMES_DIR = EXTERNAL_DIR / "runtimes"
CRISPASR_REPOSITORY = "CrispStrobe/CrispASR"


@dataclass(frozen=True)
class ModelChoice:
    key: str
    label: str
    repo_id: str
    backend: str
    include: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstalledModel:
    kind: str
    backend: str
    repo_id: str
    path: str
    codec_path: str = ""
    aligner_path: str = ""


def model_choices(kind: str) -> tuple[ModelChoice, ...]:
    mac = platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }
    if kind == "asr":
        if mac:
            return (
                ModelChoice(
                    "mlx",
                    "Qwen3-ASR 0.6B 8bit（MLX）",
                    "mlx-community/Qwen3-ASR-0.6B-8bit",
                    "mlx",
                ),
                ModelChoice("other", "其他模型", "", "mlx"),
            )
        return (
            ModelChoice(
                "official",
                "Qwen3-ASR 0.6B（官方）",
                "Qwen/Qwen3-ASR-0.6B",
                "hf",
            ),
            ModelChoice(
                "gguf",
                "Qwen3-ASR 0.6B（GGUF）",
                "handy-computer/Qwen3-ASR-0.6B-gguf",
                "gguf",
                ("*.gguf",),
            ),
            ModelChoice("other", "其他模型", "", "hf"),
        )
    if mac:
        return (
            ModelChoice(
                "mlx",
                "Qwen3-TTS 0.6B Base 8bit（MLX）",
                "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
                "mlx",
            ),
            ModelChoice("other", "其他模型", "", "mlx"),
        )
    return (
        ModelChoice(
            "official",
            "Qwen3-TTS 0.6B Base（官方）",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "hf",
        ),
        ModelChoice(
            "gguf",
            "Qwen3-TTS 0.6B Base Q8（GGUF）",
            "cstr/qwen3-tts-0.6b-base-GGUF",
            "gguf",
            ("qwen3-tts-12hz-0.6b-base-q8_0.gguf",),
        ),
        ModelChoice("other", "其他模型", "", "hf"),
    )


def choice_by_label(kind: str, label: str) -> ModelChoice:
    for item in model_choices(kind):
        if item.label == label:
            return item
    return model_choices(kind)[0]


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "--").replace("\\", "--").replace(":", "-")


def model_path(kind: str, backend: str, repo_id: str) -> Path:
    return MODELS_DIR / kind / backend / _safe_repo_name(repo_id)


def manifest_path(path: Path) -> Path:
    return path / ".videodub-model.json"


def read_installed_model(path: str | Path) -> InstalledModel | None:
    file = manifest_path(Path(path))
    if not file.is_file():
        return None
    try:
        return InstalledModel(**json.loads(file.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def is_model_installed(kind: str, backend: str, repo_id: str) -> bool:
    return read_installed_model(model_path(kind, backend, repo_id)) is not None


def list_installed_models(kind: str) -> list[InstalledModel]:
    root = MODELS_DIR / kind
    if not root.exists():
        return []
    result: list[InstalledModel] = []
    for file in root.rglob(".videodub-model.json"):
        installed = read_installed_model(file.parent)
        if installed:
            result.append(installed)
    return sorted(result, key=lambda item: (item.backend, item.repo_id.lower()))


def env_python(backend: str, kind: str) -> Path:
    name = "mlx" if backend == "mlx" else f"{kind}-hf"
    root = ENVS_DIR / name
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _install_runtime(kind: str, backend: str, runner: ProcessRunner) -> None:
    if backend == "gguf":
        _install_crispasr(runner)
        return
    python_path = env_python(backend, kind)
    if not python_path.is_file():
        version = "3.13" if backend == "mlx" else "3.12"
        runner.logger(f"正在创建 {kind.upper()} 模型环境（Python {version}）…")
        runner.run(["uv", "venv", python_path.parent.parent, "--python", version])
    if backend == "mlx":
        packages = [
            "fastapi>=0.128",
            "huggingface-hub[hf_xet]",
            "mlx-audio>=0.3",
            "numpy",
            "python-multipart",
            "soundfile",
            "uvicorn>=0.40",
        ]
    elif kind == "asr":
        packages = [
            "fastapi>=0.128",
            "python-multipart",
            "qwen-asr",
            "uvicorn>=0.40",
        ]
    else:
        packages = [
            "fastapi>=0.128",
            "numpy",
            "qwen-tts",
            "soundfile",
            "uvicorn>=0.40",
        ]
    runner.logger("正在安装模型运行环境…")
    runner.run(["uv", "pip", "install", "--python", python_path, *packages])


def _download_hf(
    repo_id: str,
    target: Path,
    runner: ProcessRunner,
    include: tuple[str, ...] = (),
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    command: list[str | Path] = [
        "uv",
        "tool",
        "run",
        "--from",
        "huggingface-hub[hf_xet]",
        "hf",
        "download",
        repo_id,
        "--local-dir",
        target,
    ]
    for pattern in include:
        command.extend(["--include", pattern])
    runner.logger(f"正在从 Hugging Face 下载：{repo_id}")
    runner.run(command)


def _download_file(
    url: str,
    target: Path,
    runner: ProcessRunner,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "youtube-video-localizer/2.1"},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as out:
        while True:
            runner.check_cancelled()
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _safe_extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            resolved = (target / item.filename).resolve()
            resolved.relative_to(target.resolve())
        package.extractall(target)


def _safe_extract_tar(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as package:
        for item in package.getmembers():
            resolved = (target / item.name).resolve()
            resolved.relative_to(target.resolve())
        package.extractall(target)


def _install_crispasr(runner: ProcessRunner) -> Path:
    existing = crispasr_executable()
    if existing:
        return existing
    api = (
        f"https://api.github.com/repos/{CRISPASR_REPOSITORY}/releases/latest"
    )
    request = urllib.request.Request(
        api,
        headers={"User-Agent": "youtube-video-localizer/2.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.loads(response.read().decode("utf-8"))
    system_token = "windows" if os.name == "nt" else "linux"
    machine = platform.machine().lower()
    arch_tokens = (
        ("arm64", "aarch64")
        if machine in {"arm64", "aarch64"}
        else ("x86_64", "x64", "amd64")
    )
    assets = release.get("assets") or []

    def score(asset: dict[str, object]) -> int:
        name = str(asset.get("name") or "").lower()
        if system_token not in name or not any(token in name for token in arch_tokens):
            return -100
        value = 10
        if "cpu" in name:
            value += 5
        if "cuda" in name or "vulkan" in name:
            value -= 3
        if name.endswith((".zip", ".tar.gz", ".tgz")):
            value += 2
        return value

    candidates = sorted(assets, key=score, reverse=True)
    if not candidates or score(candidates[0]) < 0:
        raise RuntimeError("CrispASR 最新版本没有适合当前系统的预编译文件")
    asset = candidates[0]
    name = str(asset["name"])
    url = str(asset["browser_download_url"])
    runner.logger(f"正在安装 CrispASR：{release.get('tag_name')} / {name}")
    with tempfile.TemporaryDirectory(prefix="videodub-crispasr-") as temp:
        archive = Path(temp) / name
        _download_file(url, archive, runner)
        RUNTIMES_DIR.mkdir(parents=True, exist_ok=True)
        if name.endswith(".zip"):
            _safe_extract_zip(archive, RUNTIMES_DIR)
        elif name.endswith((".tar.gz", ".tgz")):
            _safe_extract_tar(archive, RUNTIMES_DIR)
        else:
            target = RUNTIMES_DIR / name
            shutil.copy2(archive, target)
            target.chmod(target.stat().st_mode | stat.S_IEXEC)
    result = crispasr_executable()
    if not result:
        raise RuntimeError("CrispASR 已下载，但未找到可执行文件")
    return result


def crispasr_executable() -> Path | None:
    if not RUNTIMES_DIR.exists():
        return None
    names = {"crispasr.exe"} if os.name == "nt" else {"crispasr"}
    for item in RUNTIMES_DIR.rglob("*"):
        if item.is_file() and item.name.lower() in names:
            return item
    return None


def ensure_reference_audio(config: AppConfig, runner: ProcessRunner) -> Path:
    target = EXTERNAL_DIR / "voices" / "default-reference.wav"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    mp3 = target.with_suffix(".mp3")
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("基础环境缺少 edge-tts，无法建立默认参考声音") from exc
    runner.logger("正在生成默认中文参考声音…")
    asyncio.run(
        edge_tts.Communicate(
            config.tts_reference_text,
            config.tts_voice,
        ).save(str(mp3))
    )
    runner.run(
        [
            config.ffmpeg_path,
            "-y",
            "-i",
            mp3,
            "-ar",
            "24000",
            "-ac",
            "1",
            target,
        ]
    )
    mp3.unlink(missing_ok=True)
    return target


def install_model(
    config: AppConfig,
    kind: str,
    choice: ModelChoice,
    custom_repo: str,
    runner: ProcessRunner,
) -> InstalledModel:
    repo_id = custom_repo.strip() if choice.key == "other" else choice.repo_id
    if not repo_id or "/" not in repo_id:
        raise ValueError("请输入有效的 Hugging Face 模型名称，例如 owner/model")
    backend = choice.backend
    _install_runtime(kind, backend, runner)
    target = model_path(kind, backend, repo_id)
    _download_hf(repo_id, target, runner, choice.include)
    codec_path = ""
    aligner_path = ""
    if kind == "asr" and backend == "hf":
        aligner = MODELS_DIR / "asr" / "hf" / "Qwen--Qwen3-ForcedAligner-0.6B"
        _download_hf("Qwen/Qwen3-ForcedAligner-0.6B", aligner, runner)
        aligner_path = str(aligner)
    if kind == "tts":
        reference = ensure_reference_audio(config, runner)
        config.tts_reference_audio = str(reference)
        if backend == "gguf":
            codec = MODELS_DIR / "tts" / "gguf" / "qwen3-tts-tokenizer-12hz"
            _download_hf(
                "cstr/qwen3-tts-tokenizer-12hz-GGUF",
                codec,
                runner,
                ("qwen3-tts-tokenizer-12hz.gguf",),
            )
            codec_files = sorted(codec.rglob("*.gguf"))
            if not codec_files:
                raise RuntimeError("TTS GGUF 编解码器下载后未找到 .gguf 文件")
            codec_path = str(codec_files[0])
    installed = InstalledModel(
        kind,
        backend,
        repo_id,
        str(target),
        codec_path,
        aligner_path,
    )
    manifest_path(target).write_text(
        json.dumps(asdict(installed), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    runner.logger(f"模型安装完成：{target}")
    return installed


def uninstall_model(installed: InstalledModel, runner: ProcessRunner) -> None:
    target = Path(installed.path).resolve()
    target.relative_to(MODELS_DIR.resolve())
    if target.exists():
        shutil.rmtree(target)
    runner.logger(f"已卸载模型：{installed.repo_id}")


def first_model_file(path: str | Path, pattern: str = "*") -> Path:
    root = Path(path)
    if root.is_file():
        return root
    files = [
        item
        for item in root.rglob(pattern)
        if item.is_file() and item.name != ".videodub-model.json"
    ]
    if not files:
        raise RuntimeError(f"模型目录中没有找到 {pattern}：{root}")
    return sorted(files)[0]
