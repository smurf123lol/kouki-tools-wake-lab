from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .base import StageContext, StageResult
from .harvest import HarvestStage
from .prepare import PrepareStage
from .promote import PromoteStage
from .report import ReportStage
from .train import TrainStage

StageRun = Callable[[StageContext], StageResult]


@dataclass(frozen=True)
class StageSpec:
    """Запись реестра стадий: имя, описание, доступность и функция запуска."""

    name: str
    description: str
    available: bool
    run: StageRun
    unavailable_reason: str | None = None


_HARVEST = HarvestStage()
_PROMOTE = PromoteStage()
_PREPARE = PrepareStage()
_REPORT = ReportStage()
_TRAIN = TrainStage()

# Единственное место, где перечислены имена стадий: явный dict, без импорт-магии.
STAGES: dict[str, StageSpec] = {
    "harvest": StageSpec(
        name="harvest",
        description="сбор размеченных сэмплов через API Kouki",
        available=True,
        run=_HARVEST.run,
    ),
    "promote": StageSpec(
        name="promote",
        description="дедуп, holdout-сплит и запись корпуса в CAS",
        available=True,
        run=_PROMOTE.run,
    ),
    "prepare": StageSpec(
        name="prepare",
        description="расчет Log-Mel спектрограмм L1/L2 и экспорт в features.safetensors",
        available=True,
        run=_PREPARE.run,
    ),
    "report": StageSpec(
        name="report",
        description="сводка по корпусу и выкачанным данным (JSON)",
        available=True,
        run=_REPORT.run,
    ),
    "train": StageSpec(
        name="train",
        description="обучение модели wake-word",
        available=False,
        run=_TRAIN.run,
        unavailable_reason="не входит в объём T159 (см. спеку T159 §3)",
    ),
}


def stage_names() -> list[str]:
    """Имена стадий в стабильном (алфавитном) порядке."""
    return sorted(STAGES)


def get_stage(name: str) -> StageSpec | None:
    """Запись реестра по имени или ``None`` для неизвестной стадии."""
    return STAGES.get(name)
