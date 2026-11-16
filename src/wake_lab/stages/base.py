from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..artifacts import ArtifactStore
from ..paths import PathGuard


class StageName(str, Enum):
    PROMOTE = "promote"


@dataclass
class StageContext:
    config: dict[str, Any]
    paths: PathGuard
    artifacts: ArtifactStore
    logger: logging.Logger
    dry_run: bool = False
    device: str = "cpu"


@dataclass
class StageResult:
    stage: StageName
    status: str
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


@runtime_checkable
class Stage(Protocol):
    name: StageName

    def run(self, ctx: StageContext) -> StageResult:
        ...
