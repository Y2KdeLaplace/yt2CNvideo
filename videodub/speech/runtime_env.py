from __future__ import annotations


def runtime_packages(kind: str, engine: str) -> tuple[str, tuple[str, ...]]:
    if engine == "gguf":
        return (
            "3.12",
            ("fastapi>=0.128", "python-multipart", "uvicorn>=0.40"),
        )
    if engine == "mlx":
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
            ("fastapi>=0.128", "python-multipart", "qwen-asr", "uvicorn>=0.40"),
        )
    return (
        "3.12",
        ("fastapi>=0.128", "numpy", "qwen-tts", "soundfile", "uvicorn>=0.40"),
    )


def uv_runtime_prefix(kind: str, engine: str) -> list[str]:
    version, packages = runtime_packages(kind, engine)
    command = ["uv", "run", "--no-project", "--python", version]
    for package in packages:
        command.extend(["--with", package])
    return command
