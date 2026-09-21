from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CapabilityDependency:
    id: str
    display_name: str
    source: str
    repo_id: str
    capabilities: tuple[str, ...]
    files: tuple[str, ...] = ()
    manifest_field: str = "aligner_path"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    kind: str
    family: str
    source: str
    repo_id: str
    runtime: str
    engine: str
    capabilities: tuple[str, ...]
    dependencies: tuple[CapabilityDependency, ...] = ()
    variant: str = ""


@dataclass(frozen=True)
class ModelStatus:
    id: str
    downloaded: bool
    supported: bool
    runtime_available: bool
    dependency_complete: bool
    loaded: bool = False
    available_capabilities: tuple[str, ...] = field(default_factory=tuple)
    missing_dependencies: tuple[str, ...] = field(default_factory=tuple)
