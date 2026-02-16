"""Algorithm registry for discovering and instantiating alignment methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.contracts.algorithms import AlgorithmPlugin

_REGISTRY: dict[str, type[AlgorithmPlugin]] = {}


def register(name: str):
    """Decorator to register an algorithm implementation.

    Args:
        name: Identifier used in config files (e.g., "dpo", "ppo").
    """

    def decorator(cls: type[AlgorithmPlugin]) -> type[AlgorithmPlugin]:
        _REGISTRY[name] = cls
        return cls

    return decorator


class AlgorithmRegistry:
    """Lookup registered algorithm implementations by name."""

    @staticmethod
    def get(name: str) -> AlgorithmPlugin:
        """Instantiate a registered algorithm by config name.

        Args:
            name: Algorithm identifier from the training config.

        Raises:
            KeyError: If no algorithm is registered under `name`.
        """
        if name not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY)) or "(none)"
            msg = f"Unknown algorithm '{name}'. Available: {available}"
            raise KeyError(msg)
        return _REGISTRY[name]()

    @staticmethod
    def available() -> list[str]:
        """Return sorted list of registered algorithm names."""
        return sorted(_REGISTRY)
