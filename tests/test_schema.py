from __future__ import annotations

import pytest

from wake_lab.schema import ConfigError, validate_config


def _config() -> dict:
    return {
        "project_root": ".",
        "paths": {
            "baseline": "data/baseline",
            "telemetry_labeled": "data/telemetry_harvested/labeled",
            "artifacts": "artifacts",
        },
        "api": {"base_url": "http://127.0.0.1:5300", "timeout_s": 10, "retries": 2},
        "baseline": {"manifest": "manifest.jsonl", "sha256": None},
        "promote": {
            "categories": {
                "TP": "positive",
                "CONFUSER": "negative_confuser",
                "SPEECH_NEG": "negative_speech",
                "NOISE": "negative_noise",
            },
            "dedup_iou_threshold": 0.8,
            "confuser_ratio": 0.15,
            "test_size": 0.2,
            "random_state": 42,
        },
    }


def test_valid_config_is_accepted() -> None:
    cfg = _config()
    assert validate_config(cfg) is cfg


def test_unknown_top_level_key_reports_full_path() -> None:
    cfg = _config()
    cfg["nope"] = True
    with pytest.raises(ConfigError, match=r"^nope:"):
        validate_config(cfg)


def test_unknown_nested_key_reports_full_path() -> None:
    cfg = _config()
    cfg["promote"]["unknown_option"] = 1
    with pytest.raises(ConfigError, match=r"^promote\.unknown_option:"):
        validate_config(cfg)


def test_missing_api_section_reports_required_key() -> None:
    cfg = _config()
    del cfg["api"]
    with pytest.raises(ConfigError, match=r"^api:"):
        validate_config(cfg)


def test_missing_paths_key_reports_full_path() -> None:
    cfg = _config()
    del cfg["paths"]["artifacts"]
    with pytest.raises(ConfigError, match=r"^paths\.artifacts:"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [0, 1, 1.5, -0.1])
def test_test_size_must_be_strictly_inside_unit_interval(value: float) -> None:
    cfg = _config()
    cfg["promote"]["test_size"] = value
    with pytest.raises(ConfigError, match=r"^promote\.test_size:"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [0, 1, 1.2])
def test_dedup_iou_threshold_must_be_strictly_inside_unit_interval(value: float) -> None:
    cfg = _config()
    cfg["promote"]["dedup_iou_threshold"] = value
    with pytest.raises(ConfigError, match=r"^promote\.dedup_iou_threshold:"):
        validate_config(cfg)


def test_unknown_category_is_rejected_with_path() -> None:
    cfg = _config()
    cfg["promote"]["categories"]["FOO"] = "positive"
    with pytest.raises(ConfigError, match=r"^promote\.categories\.FOO:"):
        validate_config(cfg)
