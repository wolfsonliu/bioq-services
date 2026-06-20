"""MethodAdapter ABC — translates normalized Input → method-specific FC payload,
and back from FC outputs → normalized Output.

One adapter per (TaskKind, method) pair.  Each adapter owns an FCDispatcher
pointing at the underlying GPU service.

See engineering/decisions/2026-06-20-ensemble-service-design.md for details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from ..task_kind import TaskKind

if TYPE_CHECKING:
    from pipelines.framework.fc_dispatcher import FCDispatcher

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class MethodAdapter(Generic[InputT, OutputT], ABC):
    """Per-(TaskKind, method) translator.

    Subclasses set the class attributes `name`, `task_kind`, and
    `method_options_schema`, then implement `build_request` and
    `normalize_output`.
    """

    name: str                                 # "alphafold" / "esmfold2" / ...
    task_kind: TaskKind
    method_options_schema: type[BaseModel]    # method-specific knobs

    def __init__(self, fc_dispatcher: "FCDispatcher") -> None:
        self.fc = fc_dispatcher

    @abstractmethod
    def build_request(
        self,
        input: InputT,
        options: BaseModel,
    ) -> tuple[str, dict[str, Any], dict[str, Path | list[Path]]]:
        """Return (endpoint_path, payload, files_dict) for FCDispatcher.submit."""

    @abstractmethod
    def normalize_output(
        self,
        sub_task_id: str,
        downloaded_dir: Path,
    ) -> OutputT:
        """Parse FC outputs into normalized Output."""

    def estimate_runtime_seconds(self, input: InputT | None) -> int:
        """Best-effort runtime estimate.  Used for client UI only."""
        return 600  # default 10 min
