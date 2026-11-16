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
