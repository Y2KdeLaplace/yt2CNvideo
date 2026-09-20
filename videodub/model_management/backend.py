from __future__ import annotations

import json
import os
import platform
import re
import shutil
import stat
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from ..config import AppConfig
from ..platform_utils import application_cache_dir
from ..runner import CancelledError, CommandError, ProcessRunner


CRISPASR_REPOSITORY = "CrispStrobe/CrispASR"
HF_FORCED_ALIGNER_REPOSITORY = "Qwen/Qwen3-ForcedAligner-0.6B"
GGUF_FORCED_ALIGNER_REPOSITORY = "cstr/qwen3-forced-aligner-0.6b-GGUF"
GGUF_FORCED_ALIGNER_FILENAME = "qwen3-forced-aligner-0.6b-q8_0.gguf"
MLX_FORCED_ALIGNER_REPOSITORY = (
    "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
)
HF_OFFICIAL_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINTS = (
    "https://hf-cdn.sufy.com",
    "https://hf-mirror.com",
)
MODEL_MANIFEST_VERSION = 1
MODEL_MANIFEST_RELATIVE_PATH = Path("model-management") / "models.json"
MODEL_SOURCES = ("huggingface", "modelscope")
_REPOSITORY_ID = re.compile(r"^[^/\s]+/[^/\s]+$")


def runtimes_dir() -> Path:
    return application_cache_dir() / "runtimes"


@dataclass(frozen=True)
class ModelChoice:
    key: str
    label: str
    repo_id: str
    backend: str
    source: str = "huggingface"


@dataclass(frozen=True)
class ModelFileOption:
    label: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class InstalledModel:
    kind: str
    backend: str
    repo_id: str
    path: str
    codec_path: str = ""
    aligner_path: str = ""
    source: str = "huggingface"
    variant: str = ""
    vad_path: str = ""

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.source.casefold(), self.repo_id.casefold(), self.path)


def model_manifest_path(config: AppConfig) -> Path:
    return Path(config.cache_dir).expanduser() / MODEL_MANIFEST_RELATIVE_PATH


def normalize_repository_id(repo_id: str) -> str:
    value = repo_id.strip().strip("/")
    if not _REPOSITORY_ID.fullmatch(value):
        raise ValueError("请输入有效的模型名称，例如 owner/model")
    return value


def _model_from_dict(value: object) -> InstalledModel | None:
    if not isinstance(value, dict):
        return None
    try:
        model = InstalledModel(
            kind=str(value.get("kind") or "unknown"),
            backend=str(value["backend"]),
            repo_id=normalize_repository_id(str(value["repo_id"])),
            path=str(Path(str(value["path"])).expanduser().resolve()),
            codec_path=str(value.get("codec_path") or ""),
            aligner_path=str(value.get("aligner_path") or ""),
            source=str(value["source"]).casefold(),
            variant=str(value.get("variant") or ""),
            vad_path=str(value.get("vad_path") or ""),
        )
    except (KeyError, TypeError, ValueError, OSError):
        return None
    if model.source not in MODEL_SOURCES or model.backend not in {
        "hf",
        "mlx",
        "gguf",
    }:
        return None
    return model


def _model_to_dict(model: InstalledModel) -> dict[str, str]:
    return {
        "kind": model.kind,
        "backend": model.backend,
        "repo_id": model.repo_id,
        "path": model.path,
        "codec_path": model.codec_path,
        "aligner_path": model.aligner_path,
        "source": model.source,
        "variant": model.variant,
        "vad_path": model.vad_path,
    }


def _write_manifest(config: AppConfig, models: Iterable[InstalledModel]) -> Path:
    path = model_manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = {model.identity: model for model in models}
    payload = {
        "version": MODEL_MANIFEST_VERSION,
        "models": [
            _model_to_dict(model)
            for model in sorted(
                unique.values(),
                key=lambda item: (item.repo_id.casefold(), item.source, item.path),
            )
        ],
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _read_manifest(config: AppConfig) -> list[InstalledModel] | None:
    path = model_manifest_path(config)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("version") != MODEL_MANIFEST_VERSION:
        return []
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return []
    return [
        model
        for value in raw_models
        if (model := _model_from_dict(value)) is not None
    ]


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def model_choices(kind: str) -> tuple[ModelChoice, ...]:
    if kind == "asr":
        if _is_apple_silicon():
            return (
                ModelChoice(
                    "mlx",
                    "Qwen3-ASR 0.6B 8bit（MLX）",
                    "mlx-community/Qwen3-ASR-0.6B-8bit",
                    "mlx",
                ),
                ModelChoice("other", "其他 Hugging Face 模型", "", "mlx"),
            )
        return (
            ModelChoice(
                "official",
                "Qwen3-ASR 0.6B（官方）",
                "Qwen/Qwen3-ASR-0.6B",
                "hf",
                "modelscope",
            ),
            ModelChoice(
                "gguf",
                "Qwen3-ASR 0.6B（GGUF）",
                "cstr/qwen3-asr-0.6b-GGUF",
                "gguf",
            ),
            ModelChoice("other", "其他 Hugging Face 模型", "", "hf"),
        )
    if _is_apple_silicon():
        return (
            ModelChoice(
                "mlx-base",
                "Qwen3-TTS 0.6B Base 8bit（MLX）",
                "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
                "mlx",
            ),
            ModelChoice(
                "mlx-custom",
                "Qwen3-TTS 0.6B CustomVoice 8bit（MLX）",
                "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
                "mlx",
            ),
            ModelChoice("other", "其他 Hugging Face 模型", "", "mlx"),
        )
    return (
        ModelChoice(
            "official-base",
            "Qwen3-TTS 0.6B Base（官方）",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "hf",
            "modelscope",
        ),
        ModelChoice(
            "official-custom",
            "Qwen3-TTS 0.6B CustomVoice（官方）",
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "hf",
            "modelscope",
        ),
        ModelChoice(
            "gguf-base",
            "Qwen3-TTS 0.6B Base（GGUF）",
            "cstr/qwen3-tts-0.6b-base-GGUF",
            "gguf",
        ),
        ModelChoice(
            "gguf-custom",
            "Qwen3-TTS 0.6B CustomVoice（GGUF）",
            "cstr/qwen3-tts-0.6b-customvoice-GGUF",
            "gguf",
        ),
        ModelChoice("other", "其他 Hugging Face 模型", "", "hf"),
    )


def choice_by_label(kind: str, label: str) -> ModelChoice:
    for item in model_choices(kind):
        if item.label == label:
            return item
    return model_choices(kind)[0]


def huggingface_cache_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def modelscope_cache_root() -> Path:
    explicit = os.environ.get("MODELSCOPE_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".cache" / "modelscope"


def _hf_repo_root(repo_id: str) -> Path:
    return huggingface_cache_root() / (
        "models--" + repo_id.replace("/", "--")
    )


def _has_model_content(path: Path) -> bool:
    """Confirm model weights with a bounded scan of the cache entry."""
    if not path.is_dir():
        return False
    weight_suffixes = {
        ".bin",
        ".gguf",
        ".mlx",
        ".npz",
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
    }
    pending = [(path, 0)]
    try:
        while pending:
            current, depth = pending.pop()
            for item in current.iterdir():
                if item.is_file() and item.suffix.casefold() in weight_suffixes:
                    return True
                if depth < 1 and item.is_dir():
                    pending.append((item, depth + 1))
    except OSError:
        return False
    return False


def resolve_huggingface_model(repo_id: str) -> Path | None:
    root = _hf_repo_root(repo_id)
    snapshots = root / "snapshots"
    candidates: list[Path] = []
    reference = root / "refs" / "main"
    if reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(snapshots / revision)
    for candidate in candidates:
        if _has_model_content(candidate):
            return candidate.resolve()
    return None


def resolve_modelscope_model(repo_id: str) -> Path | None:
    owner, name = normalize_repository_id(repo_id).split("/", 1)
    root = modelscope_cache_root()
    repository = root / "models" / f"{owner}--{name}"
    snapshots = repository / "snapshots"
    if snapshots.is_dir():
        for candidate in sorted(
            (item for item in snapshots.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            if _has_model_content(candidate):
                return candidate.resolve()
    legacy_candidates = (
        root / "models" / owner / name,
        root / owner / name,
        root / "hub" / owner / name,
        root / "hub" / "models" / owner / name,
        root / "models" / owner.lower() / name,
        root / owner.lower() / name,
    )
    for candidate in legacy_candidates:
        if _has_model_content(candidate):
            return candidate.resolve()
    return None


def _variant(repo_id: str) -> str:
    lowered = repo_id.casefold()
    if "customvoice" in lowered or "custom-voice" in lowered:
        return "custom_voice"
    if "tts" in lowered and "base" in lowered:
        return "base"
    return ""


_GGUF_SHARD = re.compile(
    r"^(?P<base>.+?)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


def group_gguf_files(files: Iterable[str]) -> tuple[ModelFileOption, ...]:
    groups: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    for raw in files:
        filename = str(raw).strip().replace("\\", "/")
        if not filename.casefold().endswith(".gguf"):
            continue
        name = filename.rsplit("/", 1)[-1]
        match = _GGUF_SHARD.match(name)
        key = filename.casefold()
        label = filename
        if match:
            prefix = filename[: -len(name)]
            key = (prefix + match.group("base")).casefold()
            label = (
                prefix
                + match.group("base")
                + f".gguf（{int(match.group('total'))} 个分片）"
            )
        groups.setdefault(key, []).append(filename)
        labels[key] = label
    return tuple(
        ModelFileOption(labels[key], tuple(sorted(values, key=str.casefold)))
        for key, values in sorted(groups.items(), key=lambda item: item[0])
    )


def _list_huggingface_gguf_options(
    repo_id: str,
    runner: ProcessRunner,
    endpoint: str,
) -> tuple[ModelFileOption, ...]:
    runner.logger(f"正在检查 Hugging Face 模型文件：{repo_id}")
    endpoint = endpoint.rstrip("/")
    safe_repo = urllib.parse.quote(repo_id, safe="/")
    url = (
        f"{endpoint}/api/models/{safe_repo}/tree/main"
        "?recursive=true&expand=false"
    )
    headers = {"User-Agent": f"scip/{__version__}"}
    token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    files: list[str] = []
    while url:
        runner.check_cancelled()
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
                link = response.headers.get("Link", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"无法读取 Hugging Face 模型文件（HTTP {exc.code}）："
                f"{detail[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"无法连接 Hugging Face：{exc.reason}"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("Hugging Face 返回的模型文件列表无效") from exc
        if not isinstance(payload, list):
            raise RuntimeError("Hugging Face 返回的模型文件列表无效")
        for item in payload:
            if isinstance(item, dict) and item.get("type") == "file":
                path = str(item.get("path") or "")
                if path.casefold().endswith(".gguf"):
                    files.append(path)
        url = ""
        for part in link.split(","):
            if 'rel="next"' not in part:
                continue
            start = part.find("<")
            end = part.find(">", start + 1)
            if start >= 0 and end > start:
                candidate = urllib.parse.urljoin(
                    endpoint + "/",
                    part[start + 1 : end],
                )
                expected = urllib.parse.urlsplit(endpoint)
                actual = urllib.parse.urlsplit(candidate)
                if (
                    actual.scheme == expected.scheme
                    and actual.netloc == expected.netloc
                ):
                    url = candidate
                break
    return group_gguf_files(files)


def list_huggingface_gguf_options(
    repo_id: str,
    runner: ProcessRunner,
) -> tuple[ModelFileOption, ...]:
    errors: list[Exception] = []
    endpoints = (HF_OFFICIAL_ENDPOINT, *HF_MIRROR_ENDPOINTS)
    for index, endpoint in enumerate(endpoints):
        try:
            return _list_huggingface_gguf_options(repo_id, runner, endpoint)
        except CancelledError:
            raise
        except RuntimeError as exc:
            errors.append(exc)
            if index + 1 < len(endpoints):
                runner.logger(
                    f"检查失败，正在使用镜像重试：{endpoints[index + 1]}"
                )
    detail = str(errors[-1]) if errors else "未知错误"
    raise RuntimeError(f"无法读取 Hugging Face 模型文件：{detail}") from (
        errors[-1] if errors else None
    )


def _resolve_choice(choice: ModelChoice) -> Path | None:
    if choice.source == "modelscope":
        return (
            resolve_modelscope_model(choice.repo_id)
            or resolve_huggingface_model(choice.repo_id)
        )
    return resolve_huggingface_model(choice.repo_id)


def _companion_paths(kind: str, backend: str) -> tuple[str, str]:
    codec_path = ""
    aligner_path = ""
    if kind == "asr" and backend in {"hf", "mlx"}:
        repository = (
            MLX_FORCED_ALIGNER_REPOSITORY
            if backend == "mlx"
            else HF_FORCED_ALIGNER_REPOSITORY
        )
        aligner = resolve_huggingface_model(repository)
        if backend == "hf":
            aligner = resolve_modelscope_model(repository) or aligner
        aligner_path = str(aligner or "")
    if kind == "asr" and backend == "gguf":
        repository = resolve_huggingface_model(
            GGUF_FORCED_ALIGNER_REPOSITORY
        )
        if repository:
            aligner = repository / GGUF_FORCED_ALIGNER_FILENAME
            aligner_path = str(aligner) if aligner.is_file() else ""
    if kind == "tts" and backend == "gguf":
        codec = resolve_huggingface_model(
            "cstr/qwen3-tts-tokenizer-12hz-GGUF"
        )
        if codec:
            files = sorted(codec.rglob("*.gguf"))
            codec_path = str(files[0]) if files else ""
    return codec_path, aligner_path


def _installed_vad_path(kind: str, backend: str) -> str:
    if kind != "asr" or backend != "gguf":
        return ""
    repository = resolve_huggingface_model("ggml-org/whisper-vad")
    if repository is None:
        return ""
    candidate = repository / "ggml-silero-v6.2.0.bin"
    return str(candidate) if candidate.is_file() else ""


def _installed_from_choice(
    kind: str,
    choice: ModelChoice,
) -> InstalledModel | None:
    if not choice.repo_id or choice.backend == "gguf":
        return None
    source = choice.source
    path = _resolve_choice(choice)
    if path is None:
        return None
    if (
        choice.source == "modelscope"
        and resolve_modelscope_model(choice.repo_id) is None
    ):
        source = "huggingface"
    codec_path, aligner_path = _companion_paths(kind, choice.backend)
    return InstalledModel(
        kind,
        choice.backend,
        choice.repo_id,
        str(path),
        codec_path,
        aligner_path,
        source,
        _variant(choice.repo_id),
        _installed_vad_path(kind, choice.backend),
    )


def _cached_huggingface_repositories(kind: str) -> list[InstalledModel]:
    root = huggingface_cache_root()
    if not root.is_dir():
        return []
    marker = "asr" if kind == "asr" else "tts"
    result: list[InstalledModel] = []
    for repo_root in root.glob("models--*--*"):
        raw = repo_root.name.removeprefix("models--")
        owner, name = raw.split("--", 1)
        repo_id = f"{owner}/{name}"
        if marker not in repo_id.casefold():
            continue
        path = resolve_huggingface_model(repo_id)
        if path is None:
            continue
        gguf_files = [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.casefold() == ".gguf"
        ]
        backend = (
            "mlx"
            if owner.casefold() == "mlx-community"
            else "gguf"
            if gguf_files
            else "hf"
        )
        codec_path, aligner_path = _companion_paths(kind, backend)
        if backend == "gguf":
            relative_files = [
                item.relative_to(path).as_posix() for item in gguf_files
            ]
            for option in group_gguf_files(relative_files):
                result.append(
                    InstalledModel(
                        kind,
                        backend,
                        repo_id,
                        str((path / option.files[0]).resolve()),
                        codec_path,
                        aligner_path,
                        "huggingface",
                        _variant(repo_id),
                        _installed_vad_path(kind, backend),
                    )
                )
        else:
            result.append(
                InstalledModel(
                    kind,
                    backend,
                    repo_id,
                    str(path),
                    codec_path,
                    aligner_path,
                    "huggingface",
                    _variant(repo_id),
                    _installed_vad_path(kind, backend),
                )
            )
    return result


def _discover_legacy_models() -> list[InstalledModel]:
    result = [
        installed
        for kind in ("asr", "tts")
        for choice in model_choices(kind)
        if (installed := _installed_from_choice(kind, choice)) is not None
    ]
    for kind in ("asr", "tts"):
        result.extend(_cached_huggingface_repositories(kind))
    unique: dict[tuple[str, str], InstalledModel] = {}
    for item in result:
        unique[(item.repo_id.casefold(), str(Path(item.path)))] = item
    return sorted(
        unique.values(),
        key=lambda item: (item.backend, item.repo_id.casefold()),
    )


def list_installed_models(
    kind: str | None = None,
    config: AppConfig | None = None,
) -> list[InstalledModel]:
    active_config = config or AppConfig()
    models = _read_manifest(active_config)
    if models is None:
        models = _discover_legacy_models()
        try:
            _write_manifest(active_config, models)
        except OSError:
            pass
    available = [model for model in models if Path(model.path).exists()]
    if len(available) != len(models):
        try:
            _write_manifest(active_config, available)
        except OSError:
            pass
    if kind is not None:
        available = [model for model in available if model.kind in {kind, "unknown"}]
    return sorted(
        available,
        key=lambda item: (item.repo_id.casefold(), item.source, item.path),
    )


def _record_installed_model(config: AppConfig, installed: InstalledModel) -> None:
    models = list_installed_models(config=config)
    models = [model for model in models if model.identity != installed.identity]
    models.append(installed)
    _write_manifest(config, models)


def read_installed_model(
    path: str | Path,
    config: AppConfig | None = None,
) -> InstalledModel | None:
    target = Path(path)
    for item in list_installed_models(config=config):
        if Path(item.path) == target:
            return item
    return None


def runtime_packages(kind: str, backend: str) -> tuple[str, tuple[str, ...]]:
    if backend == "mlx":
        return (
            "3.13",
            (
                "fastapi>=0.128",
                "huggingface-hub[hf_xet]",
                "mlx-audio>=0.3",
                "numpy",
                "python-multipart",
                "soundfile",
                "uvicorn>=0.40",
            ),
        )
    if kind == "asr":
        return (
            "3.12",
            (
                "fastapi>=0.128",
                "python-multipart",
                "qwen-asr",
                "uvicorn>=0.40",
            ),
        )
    return (
        "3.12",
        (
            "fastapi>=0.128",
            "numpy",
            "qwen-tts",
            "soundfile",
            "uvicorn>=0.40",
        ),
    )


def uv_runtime_prefix(kind: str, backend: str) -> list[str]:
    version, packages = runtime_packages(kind, backend)
    command = ["uv", "run", "--no-project", "--python", version]
    for package in packages:
        command.extend(["--with", package])
    return command


def _install_runtime(kind: str, backend: str, runner: ProcessRunner) -> None:
    if backend == "gguf":
        _install_crispasr(runner)
        return
    runner.logger("正在由 uv 准备模型运行环境…")
    imports = (
        "import fastapi, mlx_audio, uvicorn"
        if backend == "mlx"
        else "import fastapi, qwen_asr, uvicorn"
        if kind == "asr"
        else "import fastapi, qwen_tts, uvicorn"
    )
    runner.run([*uv_runtime_prefix(kind, backend), "python", "-c", imports])


def _cache_tree_size(root: Path) -> int:
    if not root.is_dir():
        return 0
    total = 0
    try:
        for item in root.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        pass
    return total


def _huggingface_downloaded_bytes(repo_id: str) -> int:
    repository = _hf_repo_root(repo_id)
    xet_root = Path(
        os.environ.get(
            "HF_XET_CACHE",
            str(
                Path(os.environ.get("HF_HOME", "") or Path.home() / ".cache" / "huggingface")
                / "xet"
            ),
        )
    ).expanduser()
    return _cache_tree_size(repository / "blobs") + _cache_tree_size(xet_root)


def _format_download_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def _run_huggingface_download(
    command: list[str],
    repo_id: str,
    endpoint: str,
    runner: ProcessRunner,
) -> None:
    finished = threading.Event()
    initial_size = _huggingface_downloaded_bytes(repo_id)

    def monitor() -> None:
        last_reported_size = 0
        last_reported_at = 0.0
        while not finished.wait(1.0):
            current_size = _huggingface_downloaded_bytes(repo_id)
            downloaded = max(0, current_size - initial_size)
            now = time.monotonic()
            if downloaded <= 0:
                continue
            if (
                last_reported_size
                and downloaded - last_reported_size < 1024 * 1024
                and now - last_reported_at < 10
            ):
                continue
            runner.logger(
                f"下载进度（缓存已写入）：{_format_download_size(downloaded)}"
            )
            last_reported_size = downloaded
            last_reported_at = now

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        environment = {
            "HF_ENDPOINT": endpoint,
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "NO_COLOR": "1",
            "PYTHONUNBUFFERED": "1",
        }
        if endpoint != HF_OFFICIAL_ENDPOINT:
            environment["HF_HUB_DISABLE_XET"] = "1"
        runner.run(command, env=environment)
    finally:
        finished.set()
        watcher.join(timeout=1)


def _download_with_hfd(
    repo_id: str,
    runner: ProcessRunner,
    endpoint: str,
    include: tuple[str, ...],
) -> Path:
    bash = shutil.which("bash")
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not bash or not curl:
        raise RuntimeError("hfd requires both Bash and curl")
    snapshot = _hf_repo_root(repo_id) / "snapshots" / "hfd"
    snapshot.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="videodub-hfd-") as temporary:
        script = Path(temporary) / "hfd.sh"
        runner.logger(f"正在通过 hfd 下载：{repo_id}（{endpoint}）")
        runner.run(
            [
                curl,
                "--fail",
                "--location",
                "--output",
                script,
                f"{endpoint}/hfd/hfd.sh",
            ]
        )
        command: list[str | Path] = [
            bash,
            script,
            repo_id,
            "--local-dir",
            snapshot,
        ]
        if include:
            command.extend(["--include", *include])
        runner.run(command, env={"HF_ENDPOINT": endpoint})
    reference = _hf_repo_root(repo_id) / "refs" / "main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("hfd\n", encoding="utf-8")
    path = resolve_huggingface_model(repo_id)
    if path is None:
        raise RuntimeError(f"hfd completed but no model was found: {repo_id}")
    return path


def _download_huggingface(
    repo_id: str,
    runner: ProcessRunner,
    include: tuple[str, ...] = (),
) -> Path:
    command: list[str] = [
        "uvx",
        "--from",
        "huggingface-hub[hf_xet]",
        "hf",
        "download",
        repo_id,
    ]
    for pattern in include:
        command.extend(["--include", pattern])
    errors: list[Exception] = []
    repository = _hf_repo_root(repo_id)
    had_installed_model = resolve_huggingface_model(repo_id) is not None
    if repository.exists() and not had_installed_model:
        shutil.rmtree(repository)

    try:
        endpoints = (HF_OFFICIAL_ENDPOINT, *HF_MIRROR_ENDPOINTS)
        for index, endpoint in enumerate(endpoints):
            runner.logger(f"正在从 Hugging Face 下载：{repo_id}（{endpoint}）")
            try:
                _run_huggingface_download(command, repo_id, endpoint, runner)
                path = resolve_huggingface_model(repo_id)
                if path is None:
                    raise RuntimeError(
                        f"Hugging Face completed but no model was found: {repo_id}"
                    )
                return path
            except CancelledError:
                raise
            except (CommandError, RuntimeError) as exc:
                errors.append(exc)
                if index + 1 < len(endpoints):
                    runner.logger(
                        f"下载失败，正在使用镜像重试：{endpoints[index + 1]}"
                    )
        for endpoint in HF_MIRROR_ENDPOINTS:
            try:
                return _download_with_hfd(repo_id, runner, endpoint, include)
            except CancelledError:
                raise
            except (CommandError, RuntimeError) as exc:
                errors.append(exc)
                runner.logger(f"hfd 下载失败：{endpoint}")
        detail = str(errors[-1]) if errors else "未知错误"
        raise RuntimeError(f"Hugging Face 下载失败：{detail}") from (
            errors[-1] if errors else None
        )
    except Exception:
        if not had_installed_model and repository.exists():
            try:
                shutil.rmtree(repository)
                runner.logger(f"已清理下载失败的模型文件：{repo_id}")
            except OSError as cleanup_error:
                runner.logger(f"未能完整清理下载失败的模型文件：{cleanup_error}")
        raise


def _download_modelscope(repo_id: str, runner: ProcessRunner) -> Path:
    repo_id = normalize_repository_id(repo_id)
    runner.logger(f"正在从 ModelScope 下载：{repo_id}")
    runner.run(
        [
            "uvx",
            "--from",
            "modelscope",
            "modelscope",
            "download",
            "--model",
            repo_id,
        ]
    )
    path = resolve_modelscope_model(repo_id)
    if path is None:
        raise RuntimeError(f"ModelScope 下载完成后未找到模型缓存：{repo_id}")
    return path


def _download_choice(
    choice: ModelChoice,
    repo_id: str,
    runner: ProcessRunner,
    selected_files: tuple[str, ...] = (),
) -> Path:
    if choice.source == "modelscope":
        return _download_modelscope(repo_id, runner)
    return _download_huggingface(repo_id, runner, selected_files)


def _download_file(url: str, target: Path, runner: ProcessRunner) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"scip/{__version__}"},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=60) as response, target.open(
        "wb"
    ) as out:
        while True:
            runner.check_cancelled()
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _safe_extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            (target / item.filename).resolve().relative_to(target.resolve())
        package.extractall(target)


def _safe_extract_tar(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as package:
        for item in package.getmembers():
            (target / item.name).resolve().relative_to(target.resolve())
        package.extractall(target)


def _install_crispasr(runner: ProcessRunner) -> Path:
    existing = crispasr_executable()
    if existing:
        return existing
    api = f"https://api.github.com/repos/{CRISPASR_REPOSITORY}/releases/latest"
    request = urllib.request.Request(
        api,
        headers={"User-Agent": f"scip/{__version__}"},
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

    def score(asset: dict[str, object]) -> int:
        name = str(asset.get("name") or "").casefold()
        if system_token not in name or not any(
            token in name for token in arch_tokens
        ):
            return -100
        value = 10
        if "cpu" in name:
            value += 5
        if "cuda" in name or "vulkan" in name:
            value -= 3
        if name.endswith((".zip", ".tar.gz", ".tgz")):
            value += 2
        return value

    assets = sorted(release.get("assets") or [], key=score, reverse=True)
    if not assets or score(assets[0]) < 0:
        raise RuntimeError("CrispASR 最新版本没有适合当前系统的预编译文件")
    asset = assets[0]
    name = str(asset["name"])
    runner.logger(f"正在安装 CrispASR：{release.get('tag_name')} / {name}")
    with tempfile.TemporaryDirectory(prefix="videodub-crispasr-") as temp:
        archive = Path(temp) / name
        _download_file(str(asset["browser_download_url"]), archive, runner)
        runtime_root = runtimes_dir()
        runtime_root.mkdir(parents=True, exist_ok=True)
        if name.endswith(".zip"):
            _safe_extract_zip(archive, runtime_root)
        elif name.endswith((".tar.gz", ".tgz")):
            _safe_extract_tar(archive, runtime_root)
        else:
            target = runtime_root / name
            shutil.copy2(archive, target)
            target.chmod(target.stat().st_mode | stat.S_IEXEC)
    result = crispasr_executable()
    if result is None:
        raise RuntimeError("CrispASR 已下载，但未找到可执行文件")
    return result


def crispasr_executable() -> Path | None:
    try:
        runtime_root = runtimes_dir()
        if not runtime_root.is_dir():
            return None
        candidates = list(runtime_root.rglob("*"))
    except OSError:
        return None
    names = {"crispasr.exe"} if os.name == "nt" else {"crispasr"}
    for item in candidates:
        if item.is_file() and item.name.casefold() in names:
            return item
    return None


def _kind_from_repository(repo_id: str) -> str:
    lowered = repo_id.casefold()
    if "tts" in lowered:
        return "tts"
    if "asr" in lowered:
        return "asr"
    return "unknown"


def _backend_from_repository(repo_id: str, selected_files: tuple[str, ...]) -> str:
    lowered = repo_id.casefold()
    if selected_files or "gguf" in lowered:
        return "gguf"
    if lowered.startswith("mlx-community/") or "-mlx" in lowered:
        return "mlx"
    return "hf"


def download_model(
    config: AppConfig,
    source: str,
    repo_id: str,
    runner: ProcessRunner,
    selected_files: tuple[str, ...] = (),
) -> InstalledModel:
    source = source.strip().casefold()
    if source not in MODEL_SOURCES:
        raise ValueError("模型平台必须是 huggingface 或 modelscope")
    repo_id = normalize_repository_id(repo_id)
    kind = _kind_from_repository(repo_id)
    backend = _backend_from_repository(repo_id, selected_files)
    choice = ModelChoice("download", "", repo_id, backend, source)
    return install_model(
        config,
        kind,
        choice,
        "",
        runner,
        selected_files,
    )


def install_model(
    config: AppConfig,
    kind: str,
    choice: ModelChoice,
    custom_repo: str,
    runner: ProcessRunner,
    selected_files: tuple[str, ...] = (),
) -> InstalledModel:
    repo_id = custom_repo.strip() if choice.key == "other" else choice.repo_id
    repo_id = normalize_repository_id(repo_id)
    backend = "gguf" if selected_files else choice.backend
    if kind in {"asr", "tts"}:
        _install_runtime(kind, backend, runner)
    target = _download_choice(
        choice,
        repo_id,
        runner,
        selected_files,
    )
    if backend != "gguf" and any(target.rglob("*.gguf")):
        backend = "gguf"
    model_path = target
    if backend == "gguf":
        candidates = [
            target / relative
            for relative in selected_files
            if (target / relative).is_file()
        ]
        if not candidates:
            candidates = sorted(target.rglob("*.gguf"))
        if not candidates:
            raise RuntimeError("GGUF 模型下载后未找到 .gguf 文件")
        model_path = sorted(candidates)[0]
    codec_path = ""
    aligner_path = ""
    vad_path = ""
    if kind == "asr" and backend in {"hf", "mlx"}:
        aligner_repo = (
            MLX_FORCED_ALIGNER_REPOSITORY
            if backend == "mlx"
            else HF_FORCED_ALIGNER_REPOSITORY
        )
        aligner_choice = ModelChoice(
            "aligner",
            "",
            aligner_repo,
            backend,
            "huggingface" if backend == "mlx" else choice.source,
        )
        aligner_path = str(
            _download_choice(aligner_choice, aligner_choice.repo_id, runner)
        )
    if kind == "tts" and backend == "gguf":
        codec = _download_huggingface(
            "cstr/qwen3-tts-tokenizer-12hz-GGUF",
            runner,
            ("qwen3-tts-tokenizer-12hz.gguf",),
        )
        codec_files = sorted(codec.rglob("*.gguf"))
        if not codec_files:
            raise RuntimeError("TTS GGUF 编解码器下载后未找到 .gguf 文件")
        codec_path = str(codec_files[0])
    if kind == "asr" and backend == "gguf":
        aligner = _download_huggingface(
            GGUF_FORCED_ALIGNER_REPOSITORY,
            runner,
            (GGUF_FORCED_ALIGNER_FILENAME,),
        )
        aligner_file = aligner / GGUF_FORCED_ALIGNER_FILENAME
        if not aligner_file.is_file():
            raise RuntimeError("ASR GGUF 的 Qwen3 Forced Aligner 下载后未找到")
        aligner_path = str(aligner_file)
        vad = _download_huggingface(
            "ggml-org/whisper-vad",
            runner,
            ("ggml-silero-v6.2.0.bin",),
        )
        vad_file = vad / "ggml-silero-v6.2.0.bin"
        if not vad_file.is_file():
            raise RuntimeError("ASR GGUF 的 Silero VAD 依赖下载后未找到")
        vad_path = str(vad_file)
    installed = InstalledModel(
        kind,
        backend,
        repo_id,
        str(model_path.resolve()),
        codec_path,
        aligner_path,
        choice.source,
        _variant(repo_id),
        vad_path,
    )
    _record_installed_model(config, installed)
    runner.logger(f"模型下载完成：{model_path}")
    return installed


def _repository_root(installed: InstalledModel) -> tuple[Path, Path]:
    path = Path(installed.path).resolve()
    if installed.source == "modelscope":
        root = modelscope_cache_root().resolve()
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) >= 2 and parts[0] not in {"hub", "models"}:
            return root.joinpath(*parts[:2]), root
        if len(parts) >= 2 and parts[0] == "models" and "--" in parts[1]:
            return root.joinpath(*parts[:2]), root
        if len(parts) >= 3 and parts[0] == "models":
            return root.joinpath(*parts[:3]), root
        if len(parts) >= 3 and parts[0] == "hub" and parts[1] != "models":
            return root.joinpath(*parts[:3]), root
        if len(parts) >= 4 and parts[:2] == ("hub", "models"):
            return root.joinpath(*parts[:4]), root
        raise RuntimeError("无法确定 ModelScope 模型缓存目录")
    root = huggingface_cache_root().resolve()
    current = path
    while current != root and not current.name.startswith("models--"):
        current = current.parent
    current.relative_to(root)
    if not current.name.startswith("models--"):
        raise RuntimeError("无法确定 Hugging Face 模型缓存目录")
    return current, root


def uninstall_model(
    config: AppConfig,
    installed: InstalledModel,
    runner: ProcessRunner,
) -> None:
    target, root = _repository_root(installed)
    target.relative_to(root)
    if target == root:
        raise RuntimeError("拒绝删除整个模型缓存目录")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    retained: list[InstalledModel] = []
    for model in list_installed_models(config=config):
        try:
            Path(model.path).resolve().relative_to(target)
        except ValueError:
            retained.append(model)
    _write_manifest(config, retained)
    runner.logger(f"已卸载模型：{installed.repo_id}")


def first_model_file(path: str | Path, pattern: str = "*") -> Path:
    root = Path(path)
    if root.is_file():
        return root
    files = [item for item in root.rglob(pattern) if item.is_file()]
    if not files:
        raise RuntimeError(f"模型目录中没有找到 {pattern}：{root}")
    return sorted(files)[0]
