"""TaskKind enum — folding / design / scoring / ...

Each TaskKind owns:
  - Its own Input / Output Pydantic schema
  - A registry of MethodAdapter instances
  - (optionally) a TaskKind-specific aggregator (ranking function)

See engineering/decisions/2026-06-20-ensemble-service-design.md for the
abstraction rationale.
"""

from enum import Enum


class TaskKind(str, Enum):
    FOLDING = "folding"
    DESIGN = "design"
    SCORING = "scoring"
