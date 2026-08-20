"""Vision model registrations."""

from . import cnn  # noqa: F401

try:  # Optional torchvision-backed models.
    from . import efficientnet  # noqa: F401
except Exception:  # pragma: no cover - best-effort registration
    efficientnet = None

try:  # Optional torchvision-backed models.
    from . import resnet  # noqa: F401
except Exception:  # pragma: no cover - best-effort registration
    resnet = None

try:  # Optional torchvision-backed models.
    from . import vit  # noqa: F401
except Exception:  # pragma: no cover - best-effort registration
    vit = None

__all__ = [name for name in ("cnn", "efficientnet", "resnet", "vit") if globals().get(name) is not None]
