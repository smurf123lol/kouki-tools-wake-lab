from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
import yaml

from .artifacts import ArtifactStore
from .paths import PathGuard
from .schema import ConfigError, validate_config
from .stages.base import StageContext, StageResult, StageUnavailableError
from .stages.registry import STAGES, get_stage, stage_names

app = typer.Typer(help="Wake Lab: wake word data and model pipeline CLI")

_DEFAULT_CONFIG = Path("configs/default.yaml")


def _load_config(path: Path) -> dict:
    if not path.exists():
        typer.echo(f"файл конфигурации не найден: {path}", err=True)
        raise typer.Exit(2)
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        typer.echo(f"ошибка конфига {path}: {exc}", err=True)
        raise typer.Exit(2) from None
    try:
        validate_config(cfg)
    except ConfigError as exc:
        typer.echo(f"ошибка конфига {path}: {exc}", err=True)
        raise typer.Exit(2) from None
    return cfg


def _build_context(
    config_path: Path, dry_run: bool, params: dict[str, Any] | None = None
) -> StageContext:
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
        params=dict(params or {}),
    )


def _print_stage_list() -> None:
    lines = [f"{'NAME':<9}{'AVAILABLE':<11}DESCRIPTION"]
    for name in stage_names():
        spec = STAGES[name]
        available = "yes" if spec.available else "no"
        description = spec.description
        if spec.unavailable_reason:
            description = f"{description} — {spec.unavailable_reason}"
        lines.append(f"{name:<9}{available:<11}{description}")
    typer.echo("\n".join(lines))


def _print_result(result: StageResult) -> None:
    typer.echo(json.dumps(asdict(result), default=str, indent=2))


@app.command()
def stage(
    name: str = typer.Argument(
        "list",
        help="Имя стадии или 'list' для реестра стадий",
    ),
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c", help="Config file path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run without committing artifacts"),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Сколько сэмплов скачивать за стадию (обязателен для harvest)",
    ),
    theories: bool = typer.Option(
        False,
        "--theories",
        "-t",
        help="Запустить бенчмарк пула теорий при обучении (train stage)",
    ),
) -> None:
    if name == "list":
        _print_stage_list()
        return
    spec = get_stage(name)
    if spec is None:
        typer.echo(
            "неизвестная стадия: "
            f"{name}. Доступны: {', '.join(stage_names())} (список — 'stage list')",
            err=True,
        )
        raise typer.Exit(2)
    if not spec.available:
        typer.echo(f"стадия {name} недоступна: {spec.unavailable_reason}", err=True)
        raise typer.Exit(3)
    params: dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if theories:
        params["sweep_theories"] = True
    ctx = _build_context(config, dry_run, params)
    try:
        result = spec.run(ctx)
    except StageUnavailableError as exc:
        typer.echo(f"стадия {name} недоступна: {exc}", err=True)
        raise typer.Exit(3) from None
    _print_result(result)
    if result.status == "failed":
        raise typer.Exit(1)


@app.command()
def report(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c", help="Config file path"),
) -> None:
    spec = get_stage("report")
    if spec is None or not spec.available:
        typer.echo("стадия report недоступна", err=True)
        raise typer.Exit(3)
    ctx = _build_context(config, dry_run=False)
    result = spec.run(ctx)
    typer.echo(json.dumps(result.outputs, ensure_ascii=False, indent=2, default=str))
    if result.status == "failed":
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
