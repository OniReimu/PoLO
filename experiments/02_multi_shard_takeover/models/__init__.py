"""Model helpers and registry exports."""

from .base import ModelBase, ModelMetadata
from .registry import MODEL_REGISTRY, available_models, build_model, register_model
from .model import create_model

# Import subpackages for side-effect registration
try:  # Optional deps (transformers) may be missing in minimal installs.
    from . import language as _language_models  # noqa: F401
except Exception:  # pragma: no cover - best-effort registration
    _language_models = None

try:  # Optional deps (torchvision) may be missing in minimal installs.
    from . import vision as _vision_models  # noqa: F401
except Exception:  # pragma: no cover - best-effort registration
    _vision_models = None

__all__ = [
    "ModelBase",
    "ModelMetadata",
    "MODEL_REGISTRY",
    "available_models",
    "build_model",
    "register_model",
    "create_model",
]
