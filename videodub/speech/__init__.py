"""Local speech-service registry, providers, and runtime adapters."""

from .models import CapabilityDependency, ModelSpec, ModelStatus
from .registry import MODEL_REGISTRY, find_model_spec, get_model_spec

__all__ = [
    "CapabilityDependency",
    "MODEL_REGISTRY",
    "ModelSpec",
    "ModelStatus",
    "find_model_spec",
    "get_model_spec",
]
