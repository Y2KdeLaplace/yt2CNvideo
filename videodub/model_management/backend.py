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
from dataclasses import dataclass, replace
from pathlib import Path

from .. import __version__
from ..config import AppConfig
from ..platform_utils import application_cache_dir
from ..runner import CancelledError, CommandError, ProcessRunner
from ..speech.providers import build_download_command, check_provider_cli
from ..speech.registry import find_model_spec


CRISPASR_REPOSITORY = "CrispStrobe/CrispASR"
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


def _companion_paths(repo_id: str) -> dict[str, str]:
    paths = {"codec_path": "", "aligner_path": "", "vad_path": ""}
    spec = find_model_spec(repo_id)
    for dependency in spec.dependencies if spec else ():
        repository = (
            resolve_modelscope_model(dependency.repo_id)
            if dependency.source == "modelscope"
            else resolve_huggingface_model(dependency.repo_id)
        )
        if repository is None:
            continue
        path = repository
        if dependency.files:
            candidates = [repository / name for name in dependency.files]
            match = next((item for item in candidates if item.is_file()), None)
            if match is None:
                continue
            path = match
        paths[dependency.manifest_field] = str(path)
    return paths


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
    companions = _companion_paths(choice.repo_id)
    return InstalledModel(
        kind,
        choice.backend,
        choice.repo_id,
        str(path),
        companions["codec_path"],
        companions["aligner_path"],
        source,
        _variant(choice.repo_id),
        companions["vad_path"],
    )


def _cached_huggingface_repositories(kind: str) -> list[InstalledModel]:
    root = huggingface_cache_root()
    if not root.is_dir():
        return []
    result: list[InstalledModel] = []
    for repo_root in root.glob("models--*--*"):
        raw = repo_root.name.removeprefix("models--")
        owner, name = raw.split("--", 1)
        repo_id = f"{owner}/{name}"
        spec = find_model_spec(repo_id)
        if spec is None or spec.kind != kind:
            continue
        path = resolve_huggingface_model(repo_id)
        if path is None:
            continue
        gguf_files = [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.casefold() == ".gguf"
        ]
        backend = "hf" if spec.engine == "transformers" else spec.engine
        companions = _companion_paths(repo_id)
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
                        companions["codec_path"],
                        companions["aligner_path"],
                        "huggingface",
                        spec.variant,
                        companions["vad_path"],
                    )
                )
        else:
            result.append(
                InstalledModel(
                    kind,
                    backend,
                    repo_id,
                    str(path),
                    companions["codec_path"],
                    companions["aligner_path"],
                    "huggingface",
                    spec.variant,
                    companions["vad_path"],
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
        available = [model for model in available if model.kind == kind]
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


def ensure_gguf_runtime(runner: ProcessRunner) -> Path:
    """Prepare the external GGUF speech runtime when inference is requested."""
    return _install_crispasr(runner)


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


def _download_huggingface(
    repo_id: str,
    runner: ProcessRunner,
    include: tuple[str, ...] = (),
) -> Path:
    command = build_download_command("huggingface", repo_id, include)
    command[0] = check_provider_cli("huggingface")
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
    command = build_download_command("modelscope", repo_id)
    command[0] = check_provider_cli("modelscope")
    runner.run(command)
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
    spec = find_model_spec(repo_id)
    return spec.kind if spec else "unknown"


def _backend_from_repository(repo_id: str, selected_files: tuple[str, ...]) -> str:
    spec = find_model_spec(repo_id)
    if spec is not None:
        return "hf" if spec.engine == "transformers" else spec.engine
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
    spec = find_model_spec(repo_id)
    if spec is not None:
        kind = spec.kind
        backend = "hf" if spec.engine == "transformers" else spec.engine
    else:
        kind = "unknown"
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
    companion_paths = {"codec_path": "", "aligner_path": "", "vad_path": ""}
    for dependency in spec.dependencies if spec else ():
        dependency_choice = ModelChoice(
            dependency.id,
            dependency.display_name,
            dependency.repo_id,
            backend,
            dependency.source,
        )
        repository = _download_choice(
            dependency_choice,
            dependency.repo_id,
            runner,
            dependency.files,
        )
        dependency_path = repository
        if dependency.files:
            candidates = [repository / item for item in dependency.files if (repository / item).is_file()]
            if not candidates:
                raise RuntimeError(
                    f"Companion dependency 下载后缺少文件：{dependency.display_name}"
                )
            dependency_path = candidates[0]
        companion_paths[dependency.manifest_field] = str(dependency_path)
    installed = InstalledModel(
        kind,
        backend,
        repo_id,
        str(model_path.resolve()),
        companion_paths["codec_path"],
        companion_paths["aligner_path"],
        choice.source,
        spec.variant if spec else _variant(repo_id),
        companion_paths["vad_path"],
    )
    _record_installed_model(config, installed)
    runner.logger(f"模型下载完成：{model_path}")
    return installed


def repair_model_dependencies(
    config: AppConfig,
    installed: InstalledModel,
    runner: ProcessRunner,
) -> InstalledModel:
    spec = find_model_spec(installed.repo_id)
    if spec is None:
        raise RuntimeError(f"未知仓库没有可修复的 runtime dependency：{installed.repo_id}")
    values = {
        "codec_path": installed.codec_path,
        "aligner_path": installed.aligner_path,
        "vad_path": installed.vad_path,
    }
    for dependency in spec.dependencies:
        current = values[dependency.manifest_field]
        if current and Path(current).exists():
            continue
        choice = ModelChoice(
            dependency.id,
            dependency.display_name,
            dependency.repo_id,
            installed.backend,
            dependency.source,
        )
        repository = _download_choice(
            choice,
            dependency.repo_id,
            runner,
            dependency.files,
        )
        path = repository
        if dependency.files:
            match = next(
                (repository / name for name in dependency.files if (repository / name).is_file()),
                None,
            )
            if match is None:
                raise RuntimeError(f"依赖文件缺失：{dependency.display_name}")
            path = match
        values[dependency.manifest_field] = str(path)
    repaired = replace(installed, **values)
    _record_installed_model(config, repaired)
    runner.logger(f"模型依赖修复完成：{installed.repo_id}")
    return repaired


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
