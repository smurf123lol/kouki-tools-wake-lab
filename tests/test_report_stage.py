from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from wake_lab.artifacts import ArtifactStore
from wake_lab.paths import PathGuard
from wake_lab.stages.base import StageContext
from wake_lab.stages.report import ReportStage


def _config(tmp_path: Path) -> dict:
    return {
        "project_root": str(tmp_path),
        "paths": {
            "baseline": "data/baseline",
            "telemetry_labeled": "data/telemetry_harvested/labeled",
            "artifacts": "artifacts",
        },
        "api": {"base_url": "http://127.0.0.1:5300"},
    }


def _context(tmp_path: Path) -> tuple[StageContext, ArtifactStore]:
    config = _config(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    ctx = StageContext(
        config=config,
        paths=PathGuard(tmp_path, config),
        artifacts=store,
        logger=logging.getLogger("test"),
    )
    return ctx, store


def _record(index: int, category: str, split: str) -> dict:
    return {
        "id": f"{category}:{index:04d}",
        "path": f"records/{index:04d}.wav",
        "sha256": hashlib.sha256(f"{category}-{index}".encode()).hexdigest(),
        "source": "telemetry",
        "category": category,
        "label": "wake" if category == "TP" else "noise",
        "split": split,
    }


def test_report_summarises_latest_manifest_and_harvested_data(tmp_path: Path) -> None:
    ctx, store = _context(tmp_path)
    records = [
        _record(1, "TP", "train"),
        _record(2, "TP", "holdout"),
        _record(3, "NOISE", "train"),
        _record(4, "CONFUSER", "train"),
    ]
    store.commit_manifest(records, version=1)

    labeled = tmp_path / "data" / "telemetry_harvested" / "labeled"
    for category, count in (("TP", 2), ("NOISE", 1)):
        directory = labeled / category
        directory.mkdir(parents=True)
        for i in range(count):
            (directory / f"{category.lower()}-{i}.wav").write_bytes(b"RIFF")

    result = ReportStage().run(ctx)

    assert result.status == "ok"
    assert result.metrics == {
        "records": 4,
        "train": 3,
        "holdout": 1,
        "harvested_total": 3,
    }
    artifacts = result.outputs["artifacts"]
    assert artifacts["latest"] == "dataset_v1.jsonl"
    assert artifacts["version"] == 1
    assert artifacts["records"] == 4
    assert artifacts["by_category"] == {"CONFUSER": 1, "NOISE": 1, "TP": 2}
    assert artifacts["train"] == 3
    assert artifacts["holdout"] == 1
    expected_digest = hashlib.sha256(
        "\n".join(record["sha256"] for record in records).encode("utf-8")
    ).hexdigest()
    assert artifacts["corpus_sha256"] == expected_digest
    assert artifacts["promoted_at"]

    harvested = result.outputs["harvested"]
    assert harvested["by_category"] == {"NOISE": 1, "TP": 2}
    assert harvested["total"] == 3
    assert result.warnings == []


def test_report_uses_highest_dataset_version(tmp_path: Path) -> None:
    ctx, store = _context(tmp_path)
    store.commit_manifest([_record(1, "TP", "train")], version=1)
    store.commit_manifest([_record(2, "TP", "train"), _record(3, "NOISE", "train")], version=2)

    result = ReportStage().run(ctx)

    assert result.outputs["artifacts"]["latest"] == "dataset_v2.jsonl"
    assert result.outputs["artifacts"]["version"] == 2
    assert result.outputs["artifacts"]["records"] == 2


def test_report_without_manifest_is_ok_with_warning(tmp_path: Path) -> None:
    ctx, _store = _context(tmp_path)

    result = ReportStage().run(ctx)

    assert result.status == "ok"
    assert result.outputs["artifacts"]["latest"] is None
    assert result.metrics["records"] == 0
    assert any("promote" in warning for warning in result.warnings)
