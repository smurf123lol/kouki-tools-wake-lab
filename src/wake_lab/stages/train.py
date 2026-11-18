from __future__ import annotations

from .base import StageContext, StageName, StageResult, StageUnavailableError


class TrainStage:
    """Обучение модели wake-word — вне объёма T159 (честная ошибка, не no-op)."""

    name = StageName.TRAIN

    def run(self, ctx: StageContext) -> StageResult:
        raise StageUnavailableError(
            "обучение модели wake-word не входит в T159: стадия train зарегистрирована, "
            "но не реализована; см. спеку T159 §3"
        )
