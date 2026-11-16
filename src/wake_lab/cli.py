from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import typer
import yaml

from .artifacts import ArtifactStore
from .paths import PathGuard
from .stages.base import StageContext, StageName, StageResult
from .stages.promote import PromoteStage

app = typer.Typer(help="Wake Lab: wake word data and model pipeline CLI")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_context(config_path: Path, dry_run: bool) -> StageContext:
    cfg = _load_config(config_path)
    project_root = Path(cfg.get("project_root", ".")).resolve()
    paths = PathGuard(project_root, cfg)
    artifacts = ArtifactStore(paths.artifacts_dir)
    logger = logging.getLogger("wake_lab")
    return StageContext(
        config=cfg,
        paths=paths,
        artifacts=artifacts,
        logger=logger,
        dry_run=dry_run,
    )


def _run_stage(name: str, config_path: Path, dry_run: bool) -> StageResult:
    ctx = _build_context(config_path, dry_run)
    if name == StageName.PROMOTE.value:
        return PromoteStage().run(ctx)
    raise typer.BadParameter(f"unknown stage: {name}")


@app.command()
def stage(
    name: str = typer.Argument(..., help="Stage name to run (e.g. promote)"),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c", help="Config file path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run without committing artifacts"),
) -> None:
    result = _run_stage(name, config, dry_run)
    typer.echo(json.dumps(asdict(result), default=str, indent=2))
    if result.status == "failed":
        raise typer.Exit(1)


@app.command()
def run(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c", help="Config file path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run without committing artifacts"),
) -> None:
    result = _run_stage(StageName.PROMOTE.value, config, dry_run)
    typer.echo(json.dumps(asdict(result), default=str, indent=2))
    if result.status == "failed":
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
