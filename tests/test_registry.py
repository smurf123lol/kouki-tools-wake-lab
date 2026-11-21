from __future__ import annotations

import logging

from wake_lab.artifacts import ArtifactStore
from wake_lab.paths import PathGuard
from wake_lab.stages.base import StageContext
from wake_lab.stages.registry import STAGES, get_stage, stage_names


def _context(tmp_path, cfg: dict | None = None) -> StageContext:
    config = cfg or {
        "project_root": str(tmp_path),
        "paths": {
            "baseline": "data/baseline",
            "telemetry_labeled": "data/telemetry_harvested/labeled",
            "artifacts": "artifacts",
        },
        "api": {"base_url": "http://127.0.0.1:5300"},
    }
    return StageContext(
        config=config,
        paths=PathGuard(tmp_path, config),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        logger=logging.getLogger("test"),
    )


def test_registry_lists_all_stages_in_stable_order() -> None:
    assert stage_names() == ["calibrate", "harvest", "prepare", "promote", "report", "train"]
    assert set(STAGES) == {"calibrate", "harvest", "prepare", "promote", "report", "train"}


def test_every_stage_entry_has_required_fields() -> None:
    for name in stage_names():
        spec = STAGES[name]
        assert spec.name == name
        assert isinstance(spec.description, str) and spec.description
        assert isinstance(spec.available, bool)
        assert callable(spec.run)
        if not spec.available:
            assert spec.unavailable_reason


def test_train_is_available() -> None:
    spec = get_stage("train")
    assert spec is not None
    assert spec.available is True
    assert spec.unavailable_reason is None


def test_all_stages_are_available() -> None:
    for name in ("harvest", "prepare", "promote", "report", "train"):
        spec = get_stage(name)
        assert spec is not None
        assert spec.available is True
        assert spec.unavailable_reason is None


def test_unknown_stage_lookup_returns_none() -> None:
    assert get_stage("rg") is None
