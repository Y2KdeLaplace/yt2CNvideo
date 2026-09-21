from __future__ import annotations

import os
import shutil
from pathlib import Path

from ...runner import ProcessRunner


def build_download_command(
    source: str,
    repo_id: str,
    files: tuple[str, ...] = (),
) -> list[str]:
    if source == "huggingface":
        command = ["hf", "download", repo_id]
        for pattern in files:
            command.extend(["--include", pattern])
        return command
    if source == "modelscope":
        command = ["modelscope", "download", "--model", repo_id]
        if files:
            command.extend(["--include", *files])
        return command
    raise ValueError(f"Unknown model source: {source}")


def check_provider_cli(source: str) -> str:
    executable = "hf" if source == "huggingface" else "modelscope" if source == "modelscope" else ""
    if not executable:
        raise ValueError(f"Unknown model source: {source}")
    resolved = shutil.which(executable)
    if not resolved:
        install = (
            'uv tool install "huggingface-hub[hf_xet]"'
            if source == "huggingface"
            else "uv tool install modelscope"
        )
        raise RuntimeError(f"未找到 {executable} 命令。请先运行：{install}")
    return resolved


def download_repository(
    repo_id: str,
    source: str,
    runner: ProcessRunner,
    *,
    files: tuple[str, ...] = (),
) -> None:
    executable = check_provider_cli(source)
    command = build_download_command(source, repo_id, files)
    command[0] = executable
    runner.run(command)


def require_local_repository(path: str | Path) -> Path:
    repository = Path(path).expanduser().resolve()
    if not repository.exists():
        raise RuntimeError(f"模型缓存不存在：{repository}")
    return repository


def delete_repository(repo_id: str, source: str) -> Path:
    owner, separator, name = repo_id.strip().partition("/")
    if not separator or not owner or not name or "/" in name:
        raise ValueError(f"Invalid repository id: {repo_id}")
    if source == "huggingface":
        cache = Path(
            os.environ.get("HF_HUB_CACHE")
            or (Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else Path.home() / ".cache" / "huggingface" / "hub")
        ).expanduser().resolve()
        target = (cache / f"models--{owner}--{name}").resolve()
    elif source == "modelscope":
        cache = Path(os.environ.get("MODELSCOPE_CACHE") or Path.home() / ".cache" / "modelscope").expanduser().resolve()
        candidates = (
            cache / "models" / f"{owner}--{name}",
            cache / "models" / owner / name,
            cache / "hub" / "models" / owner / name,
            cache / owner / name,
        )
        target = next((item.resolve() for item in candidates if item.exists()), candidates[0].resolve())
    else:
        raise ValueError(f"Unknown model source: {source}")
    target.relative_to(cache)
    if target == cache:
        raise RuntimeError("Refusing to delete a model cache root")
    if target.exists():
        shutil.rmtree(target)
    return target
