"""Small registries used by datasets and model adapters."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        key = self._normalize(name)

        def decorator(item: T) -> T:
            if key in self._items:
                raise ValueError(f"{self.namespace} '{name}' is already registered")
            self._items[key] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        key = self._normalize(name)
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"Unknown {self.namespace} '{name}'. Available: {available}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower().replace("_", "-")


DATASETS: Registry[type] = Registry("dataset")
MODEL_ADAPTERS: Registry[type] = Registry("model adapter")
MODEL_TRAINERS: Registry[type] = Registry("model trainer")
