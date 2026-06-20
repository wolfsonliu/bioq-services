"""Process-wide registry of MethodAdapter instances, keyed by (task_kind, method)."""

from __future__ import annotations

from .base import MethodAdapter
from ..task_kind import TaskKind


class MethodRegistry:
    """In-memory registry.  Populated at app startup; queried per request."""

    def __init__(self) -> None:
        self._adapters: dict[tuple[TaskKind, str], MethodAdapter] = {}

    def register(self, adapter: MethodAdapter) -> None:
        key = (adapter.task_kind, adapter.name)
        if key in self._adapters:
            raise ValueError(f"adapter already registered: {key}")
        self._adapters[key] = adapter

    def get(self, task_kind: TaskKind, method: str) -> MethodAdapter:
        if (task_kind, method) not in self._adapters:
            raise KeyError(f"no adapter for ({task_kind.value}, {method})")
        return self._adapters[(task_kind, method)]

    def list_methods(self, task_kind: TaskKind) -> list[str]:
        return sorted(name for (tk, name) in self._adapters if tk == task_kind)


# Single process-wide registry; populated in app.py at startup.
registry = MethodRegistry()
