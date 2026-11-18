from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from wake_lab.cli import app

runner = CliRunner()

_VALID_CONFIG: dict = {
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


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_VALID_CONFIG, allow_unicode=True), encoding="utf-8")
    return path


def test_stage_list_prints_four_stages_with_availability() -> None:
    result = runner.invoke(app, ["stage", "list"])
    assert result.exit_code == 0
    output = result.output
    assert "AVAILABLE" in output
    for name in ("harvest", "promote", "report", "train"):
        assert name in output
    assert "no" in output
    assert "yes" in output


def test_unknown_stage_exits_with_code_two_and_lists_stages(config_path: Path) -> None:
    result = runner.invoke(app, ["stage", "rg", "--config", str(config_path)])
    assert result.exit_code == 2
    assert "неизвестная стадия: rg" in result.output
    assert "Доступны" in result.output
    assert "promote" in result.output


def test_train_stage_exits_with_code_three_and_explains_reason(config_path: Path) -> None:
    result = runner.invoke(app, ["stage", "train", "--config", str(config_path)])
    assert result.exit_code == 3
    assert "недоступна" in result.output
    assert "T159" in result.output


def test_harvest_stage_exits_with_code_three_until_p3(config_path: Path) -> None:
    result = runner.invoke(app, ["stage", "harvest", "--config", str(config_path)])
    assert result.exit_code == 3
    assert "П3" in result.output


def test_report_command_prints_json_summary(config_path: Path, tmp_path: Path) -> None:
    config = dict(_VALID_CONFIG)
    config["project_root"] = str(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    result = runner.invoke(app, ["report", "--config", str(path)])

    assert result.exit_code == 0
    assert '"artifacts"' in result.output
    assert '"harvested"' in result.output


def test_invalid_config_exits_with_code_two_and_prints_key_path(tmp_path: Path) -> None:
    config = dict(_VALID_CONFIG)
    config = {**config, "nope": True}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    result = runner.invoke(app, ["stage", "promote", "--config", str(path)])

    assert result.exit_code == 2
    assert "nope:" in result.output


def test_missing_config_file_exits_with_code_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["stage", "promote", "--config", str(tmp_path / "absent.yaml")])
    assert result.exit_code == 2
    assert "не найден" in result.output
