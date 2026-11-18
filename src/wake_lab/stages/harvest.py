"""Стадия ``harvest``: тянет размеченные сэмплы из API Kouki (T159 П3)."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..api.client import ApiError, KoukiApiClient
from .base import StageContext, StageName, StageResult

_PAGE_LIMIT = 200
_DEFAULT_CATEGORIES = ("TP", "CONFUSER", "SPEECH_NEG", "NOISE")


class HarvestStage:
    """Скачивает ``/api/v1/telemetry/labeled`` по категориям и пишет манифест.

    Идемпотентна: сэмплы с известным sha256 (из ``manifest.jsonl`` или уже
    лежащего файла) повторно не качаются. ``--dry-run`` показывает план, ничего
    не записывая; без ``--limit`` стадия отказывается работать, чтобы не тянуть
    весь корпус без явного указания.
    """

    name = StageName.HARVEST

    def run(self, ctx: StageContext) -> StageResult:
        started = time.monotonic()
        warnings: list[str] = []
        limit = ctx.params.get("limit")
        if limit is None:
            return self._failed(
                "укажите --limit N (например, --limit 5): harvest не тянет все "
                "данные без явного указания",
                started,
            )
        limit = int(limit)
        if limit <= 0:
            return self._failed("--limit должен быть больше нуля", started)

        raw_categories = ctx.config.get("promote", {}).get("categories")
        categories = (
            list(raw_categories)
            if isinstance(raw_categories, dict) and raw_categories
            else list(_DEFAULT_CATEGORIES)
        )
        api_cfg = ctx.config.get("api", {})
        base_url = str(api_cfg.get("base_url") or "")
        client = KoukiApiClient(
            base_url=base_url,
            timeout_s=float(api_cfg.get("timeout_s", 10)),
            retries=int(api_cfg.get("retries", 2)),
        )
        root = ctx.paths.telemetry_labeled_dir
        manifest_path = root / "manifest.jsonl"

        try:
            stats = client.stats()
        except ApiError as exc:
            return self._failed(str(exc), started)

        try:
            if ctx.dry_run:
                return self._dry_run(
                    client, categories, limit, root, stats, base_url, started
                )
            return self._harvest(
                client, categories, limit, root, manifest_path, stats, warnings, started
            )
        except ApiError as exc:
            return self._failed(str(exc), started)

    # --- dry-run ----------------------------------------------------------

    def _dry_run(
        self,
        client: KoukiApiClient,
        categories: list[str],
        limit: int,
        root: Path,
        stats: dict[str, Any],
        base_url: str,
        started: float,
    ) -> StageResult:
        plan: dict[str, int] = {}
        total = 0
        for category in categories:
            remaining = limit - total
            if remaining <= 0:
                plan[category] = 0
                continue
            records = _fetch_pages(client, category, remaining)
            plan[category] = len(records)
            total += len(records)
        outputs = {
            "mode": "dry-run",
            "base_url": base_url,
            "target": root.as_posix(),
            "limit": limit,
            "planned": plan,
            "planned_total": total,
            "api_stats": stats,
        }
        return StageResult(
            stage=self.name,
            status="dry_run",
            outputs=outputs,
            metrics={"planned": total},
            elapsed_s=round(time.monotonic() - started, 3),
        )

    # --- боевой сбор ------------------------------------------------------

    def _harvest(
        self,
        client: KoukiApiClient,
        categories: list[str],
        limit: int,
        root: Path,
        manifest_path: Path,
        stats: dict[str, Any],
        warnings: list[str],
        started: float,
    ) -> StageResult:
        known = _read_manifest(manifest_path)
        known_shas = {str(entry.get("sha256")) for entry in known if entry.get("sha256")}
        per_category: dict[str, int] = {}
        harvested = 0
        skipped = 0
        invalid = 0
        new_entries: list[dict[str, Any]] = []

        remaining_total = limit
        for category in categories:
            if remaining_total <= 0:
                per_category[category] = 0
                continue
            records = _fetch_pages(client, category, remaining_total)
            remaining_total -= len(records)
            per_category[category] = 0
            category_dir = root / category
            category_dir.mkdir(parents=True, exist_ok=True)
            for record in records:
                sample_id = str(record.get("sampleId") or "")
                sha256 = str(record.get("sha256") or "")
                if not sample_id:
                    warnings.append(f"{category}: сэмпл без sampleId пропущен")
                    invalid += 1
                    continue
                target = category_dir / f"{sample_id}.wav"
                if sha256 and sha256 in known_shas and target.exists():
                    skipped += 1
                    continue
                if target.exists():
                    file_digest = _sha256_file(target)
                    if not sha256 or file_digest == sha256 or file_digest in known_shas:
                        known_shas.add(file_digest)
                        skipped += 1
                        continue
                try:
                    payload = client.audio(sample_id)
                except ApiError as exc:
                    warnings.append(f"{category}: {sample_id} не скачан: {exc}")
                    invalid += 1
                    continue
                if not _looks_like_wav(payload):
                    warnings.append(f"{category}: {sample_id} не WAV — пропущен")
                    invalid += 1
                    continue
                _write_atomic(target, payload)
                digest = hashlib.sha256(payload).hexdigest()
                known_shas.add(digest)
                entry = {
                    "id": sample_id,
                    "label": str(record.get("manualLabel") or category),
                    "sha256": digest,
                    "duration_s": _duration_s(payload, record),
                    "source": "kouki-api",
                    "harvested_at": datetime.now(tz=UTC).isoformat(),
                }
                new_entries.append(entry)
                harvested += 1
                per_category[category] += 1

        if new_entries:
            _append_manifest(manifest_path, new_entries)

        outputs = {
            "mode": "harvest",
            "target": root.as_posix(),
            "manifest": manifest_path.as_posix(),
            "per_category": per_category,
            "api_stats": stats,
            "harvested": harvested,
            "skipped": skipped,
            "invalid": invalid,
        }
        metrics = {
            "harvested": harvested,
            "skipped": skipped,
            "invalid": invalid,
            "manifest_records": len(known) + len(new_entries),
        }
        ctx_warnings = list(warnings)
        return StageResult(
            stage=self.name,
            status="ok" if harvested or skipped else "failed",
            outputs=outputs,
            metrics=metrics,
            warnings=ctx_warnings
            if harvested or skipped
            else ctx_warnings + ["ничего не скачано и не пропущено"],
            elapsed_s=round(time.monotonic() - started, 3),
        )

    def _failed(self, message: str, started: float) -> StageResult:
        return StageResult(
            stage=self.name,
            status="failed",
            outputs={"error": message},
            warnings=[message],
            elapsed_s=round(time.monotonic() - started, 3),
        )


def _fetch_pages(
    client: KoukiApiClient, category: str, limit: int
) -> list[dict[str, Any]]:
    """Дочитывает ``labeled`` страницами по 200, пока не наберёт ``limit``."""
    records: list[dict[str, Any]] = []
    offset = 0
    while len(records) < limit:
        page_size = min(_PAGE_LIMIT, limit - len(records))
        page = client.labeled(category, limit=page_size, offset=offset)
        if not page:
            break
        records.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)
    return records[:limit]


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _append_manifest(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _looks_like_wav(payload: bytes) -> bool:
    return len(payload) > 12 and payload[0:4] == b"RIFF" and payload[8:12] == b"WAVE"


def _write_atomic(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _duration_s(payload: bytes, record: dict[str, Any]) -> float:
    sample_rate = int(record.get("sampleRate") or 16000)
    pcm_bytes = record.get("pcmByteLength")
    if isinstance(pcm_bytes, int) and pcm_bytes > 0 and sample_rate > 0:
        return round(pcm_bytes / (sample_rate * 2), 3)
    if len(payload) > 44 and sample_rate > 0:
        return round((len(payload) - 44) / (sample_rate * 2), 3)
    return 0.0
