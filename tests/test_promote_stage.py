import json
import logging

import numpy as np
import soundfile as sf

from wake_lab.artifacts import ArtifactStore
from wake_lab.paths import PathGuard
from wake_lab.stages.base import StageContext
from wake_lab.stages.promote import PromoteStage


def _write_wav(path, val=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.full(1600, val, dtype=np.float32), 16000, subtype="PCM_16")


def test_promote_stage_writes_versioned_manifest(tmp_path):
    root = tmp_path
    for d in [
        "data/baseline",
        "data/telemetry_harvested/labeled/TP",
        "data/telemetry_harvested/labeled/CONFUSER",
        "data/telemetry_harvested/labeled/SPEECH_NEG",
        "data/telemetry_harvested/labeled/NOISE",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)

    _write_wav(root / "data/telemetry_harvested/labeled/TP/a.wav", val=0.01)
    _write_wav(root / "data/telemetry_harvested/labeled/CONFUSER/b.wav", val=0.02)

    cfg = {
        "project_root": str(root),
        "paths": {
            "baseline": "data/baseline",
            "telemetry_labeled": "data/telemetry_harvested/labeled",
            "artifacts": "artifacts",
        },
        "baseline": {"manifest": "manifest.jsonl", "sha256": None},
        "promote": {
            "categories": {
                "TP": "positive",
                "CONFUSER": "negative_confuser",
                "SPEECH_NEG": "negative_speech",
                "NOISE": "negative_noise",
            },
            "dedup_iou_threshold": 0.8,
            "test_size": 0.5,
            "random_state": 42,
        },
    }

    ctx = StageContext(
        config=cfg,
        paths=PathGuard(root, cfg),
        artifacts=ArtifactStore(root / "artifacts"),
        logger=logging.getLogger("test"),
    )

    result = PromoteStage().run(ctx)
    assert result.status == "ok"
    assert result.metrics["total_records"] == 2

    manifest = ctx.artifacts.get(result.outputs["manifest"]).decode("utf-8")
    records = [json.loads(line) for line in manifest.splitlines() if line.strip()]
    assert len(records) == 2
    assert {r["split"] for r in records} == {"train", "holdout"}
    assert all(r["sha256"] for r in records)
    confusers = [r for r in records if r["subtype"] == "confuser"]
    assert len(confusers) == 1


def test_promote_stage_lowercase_dirs_and_confuser_ratio(tmp_path):
    root = tmp_path
    # Create lowercase directory 'confuser' and 'noise'
    (root / "data/baseline").mkdir(parents=True, exist_ok=True)
    (root / "data/telemetry_harvested/labeled/tp").mkdir(parents=True, exist_ok=True)
    (root / "data/telemetry_harvested/labeled/confuser").mkdir(parents=True, exist_ok=True)
    (root / "data/telemetry_harvested/labeled/noise").mkdir(parents=True, exist_ok=True)

    _write_wav(root / "data/telemetry_harvested/labeled/tp/wake1.wav", val=0.01)
    _write_wav(root / "data/telemetry_harvested/labeled/noise/noise1.wav", val=0.02)
    _write_wav(root / "data/telemetry_harvested/labeled/noise/noise2.wav", val=0.03)
    _write_wav(root / "data/telemetry_harvested/labeled/confuser/c1.wav", val=0.04)
    _write_wav(root / "data/telemetry_harvested/labeled/confuser/c2.wav", val=0.05)

    cfg = {
        "project_root": str(root),
        "paths": {
            "baseline": "data/baseline",
            "telemetry_labeled": "data/telemetry_harvested/labeled",
            "artifacts": "artifacts",
        },
        "baseline": {"manifest": "manifest.jsonl", "sha256": None},
        "promote": {
            "categories": {
                "TP": "positive",
                "CONFUSER": "negative_confuser",
                "NOISE": "negative_noise",
            },
            "confuser_ratio": 0.2,
            "dedup_iou_threshold": 0.8,
            "test_size": 0.5,
            "random_state": 42,
        },
    }

    ctx = StageContext(
        config=cfg,
        paths=PathGuard(root, cfg),
        artifacts=ArtifactStore(root / "artifacts"),
        logger=logging.getLogger("test"),
    )

    result = PromoteStage().run(ctx)
    assert result.status == "ok"
    manifest = ctx.artifacts.get(result.outputs["manifest"]).decode("utf-8")
    records = [json.loads(line) for line in manifest.splitlines() if line.strip()]
    confusers = [r for r in records if r["subtype"] == "confuser"]
    # With 2 noise samples and ratio 0.2, max confusers = round(2 * 0.2 / 0.8) = 1
    assert len(confusers) == 1
    assert any("confusers capped" in w for w in result.warnings)
