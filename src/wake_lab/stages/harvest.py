from __future__ import annotations

from .base import StageContext, StageName, StageResult, StageUnavailableError


class HarvestStage:
    """Сбор размеченных сэмплов через API Kouki (реализуется в П3, T159)."""

    name = StageName.HARVEST

    def run(self, ctx: StageContext) -> StageResult:
        raise StageUnavailableError(
            "стадия harvest реализуется в П3 (T159): сбор данных через API Kouki; "
            "сейчас в wake-lab есть только promote и report"
        )
