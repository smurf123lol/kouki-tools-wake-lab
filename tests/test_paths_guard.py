import pytest

from wake_lab.paths import PathGuard, PathGuardError


def test_path_guard_blocks_baseline_mutation(tmp_path):
    baseline_dir = tmp_path / "data/baseline"
    baseline_dir.mkdir(parents=True)
    sample = baseline_dir / "sample.txt"
    sample.write_text("x")

    cfg = {
        "paths": {
            "baseline": "data/baseline",
            "telemetry_labeled": "data/telemetry_harvested/labeled",
            "artifacts": "artifacts",
        }
    }
    guard = PathGuard(tmp_path, cfg)
    digest = guard.baseline_digest()
    guard.verify_baseline_unmodified(digest)

    with pytest.raises(PathGuardError):
        guard.ensure_writable(sample)

    sample.write_text("y")
    with pytest.raises(PathGuardError):
        guard.verify_baseline_unmodified(digest)
