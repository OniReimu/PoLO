"""Simple registry implementation."""

from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    """Simple registry for classes and functions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Any], Any]:
        """Decorator to register a class or function."""

        def decorator(obj: Any) -> Any:
            self._registry[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        """Get a registered item by name."""
        if name not in self._registry:
            raise KeyError(f"{self.name} '{name}' not found in registry")
        return self._registry[name]

    def keys(self) -> list[str]:
        """Get all registered keys."""
        return list(self._registry.keys())

