from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..schema import KNOWN_CATEGORIES
from .base import StageContext, StageName, StageResult

_DATASET_NAME = re.compile(r"dataset_v(\d+)\.jsonl$")


class ReportStage:
    """Сводка по данным на диске и последнему прогону ``promote`` (JSON)."""

    name = StageName.REPORT

    def run(self, ctx: StageContext) -> StageResult:
        started = time.monotonic()
        warnings: list[str] = []
        artifacts = self._artifacts_summary(ctx, warnings)
        harvested = self._harvested_summary(ctx)
        outputs: dict[str, Any] = {
            "artifacts": artifacts,
            "harvested": harvested,
        }
        metrics = {
            "records": artifacts.get("records", 0),
            "train": artifacts.get("train", 0),
            "holdout": artifacts.get("holdout", 0),
            "harvested_total": harvested.get("total", 0),
        }
        ctx.logger.info(
            "report: records=%s train=%s holdout=%s harvested=%s",
            metrics["records"],
            metrics["train"],
            metrics["holdout"],
            metrics["harvested_total"],
        )
        return StageResult(
            stage=self.name,
            status="ok",
            outputs=outputs,
            metrics=metrics,
            warnings=warnings,
            elapsed_s=round(time.monotonic() - started, 3),
        )

    def _artifacts_summary(
        self, ctx: StageContext, warnings: list[str]
    ) -> dict[str, Any]:
        candidates: list[tuple[int, Any]] = []
        for ref in ctx.artifacts.list_named("dataset_v"):
            match = _DATASET_NAME.search(ref.name or "")
            if match:
                candidates.append((int(match.group(1)), ref))
        if not candidates:
            warnings.append(
                "манифест dataset_v*.jsonl не найден — сначала выполните stage promote"
            )
            return {"latest": None, "records": 0}
        version, ref = max(candidates, key=lambda item: item[0])
        path = ctx.artifacts.resolve(ref)
        records = self._read_jsonl(path, warnings)
        by_category: Counter[str] = Counter()
        by_split: Counter[str] = Counter()
        for record in records:
            by_category[str(record.get("category") or "unknown")] += 1
            by_split[str(record.get("split") or "unknown")] += 1
        corpus_sha256 = hashlib.sha256(
            "\n".join(str(record.get("sha256") or "") for record in records).encode("utf-8")
        ).hexdigest()
        promoted_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        return {
            "latest": ref.name,
            "version": version,
            "records": len(records),
            "by_category": dict(sorted(by_category.items())),
            "train": by_split.get("train", 0),
            "holdout": by_split.get("holdout", 0),
            "corpus_sha256": corpus_sha256,
            "promoted_at": promoted_at,
            "size_bytes": path.stat().st_size,
        }

    def _harvested_summary(self, ctx: StageContext) -> dict[str, Any]:
        root = ctx.paths.telemetry_labeled_dir
        by_category: dict[str, int] = {}
        if root.exists():
            for directory in sorted(root.iterdir()):
                if not directory.is_dir():
                    continue
                name = directory.name.upper()
                if name not in KNOWN_CATEGORIES:
                    continue
                by_category[name] = sum(1 for _ in directory.rglob("*.wav"))
        return {
            "root": root.as_posix(),
            "by_category": by_category,
            "total": sum(by_category.values()),
        }

    @staticmethod
    def _read_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"битая строка манифеста {path}: {exc}")
                continue
            if isinstance(record, dict):
                records.append(record)
        return records
