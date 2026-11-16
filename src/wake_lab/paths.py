from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class PathGuardError(RuntimeError):
    pass


class PathGuard:
    def __init__(self, project_root: str | Path, config: dict[str, Any]):
        self.project_root = Path(project_root).resolve()
        paths = config["paths"]
        self.baseline_dir = self.resolve_path(paths["baseline"])
        self.telemetry_labeled_dir = self.resolve_path(paths["telemetry_labeled"])
        self.artifacts_dir = self.resolve_path(paths["artifacts"])

    def resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p
        return p.resolve()

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def baseline_digest(self) -> str:
        h = hashlib.sha256()
        if not self.baseline_dir.exists():
            return h.hexdigest()
        for path in sorted(self.baseline_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(self.baseline_dir).as_posix()
                h.update(rel.encode("utf-8"))
                h.update(b"\0")
                h.update(self._sha256_file(path).encode("ascii"))
                h.update(b"\n")
        return h.hexdigest()

    def verify_baseline_unmodified(self, expected_sha256: str | None = None) -> str:
        actual = self.baseline_digest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise PathGuardError(
                f"baseline digest mismatch: expected {expected_sha256}, got {actual}"
            )
        return actual

    def _is_relative_to(self, path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def ensure_writable(self, path: str | Path) -> Path:
        p = self.resolve_path(path)
        if self._is_relative_to(p, self.baseline_dir):
            raise PathGuardError(f"refusing to mutate read-only baseline path: {p}")
        return p

    def safe_open(self, path: str | Path, mode: str = "r", **kwargs):
        p = self.resolve_path(path)
        if any(m in mode for m in "wax+"):
            self.ensure_writable(p)
        return p.open(mode, **kwargs)
