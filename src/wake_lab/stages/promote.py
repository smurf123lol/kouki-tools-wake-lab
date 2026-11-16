from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit

from .base import StageContext, StageName, StageResult
from ..artifacts import ArtifactRef


class PromoteStage:
    name = StageName.PROMOTE

    def run(self, ctx: StageContext) -> StageResult:
        started = time.monotonic()
        warnings: list[str] = []
        cfg = ctx.config.get("promote", {})
        categories = cfg.get(
            "categories",
            {
                "TP": "positive",
                "CONFUSER": "negative_confuser",
                "SPEECH_NEG": "negative_speech",
                "NOISE": "negative_noise",
            },
        )
        baseline_cfg = ctx.config.get("baseline", {})
        baseline_manifest = ctx.paths.baseline_dir / baseline_cfg.get("manifest", "manifest.jsonl")
        baseline_records = self._read_jsonl(baseline_manifest, warnings)
        baseline_hashes = {r.get("sha256") for r in baseline_records if r.get("sha256")}
        baseline_digest = ctx.paths.verify_baseline_unmodified(baseline_cfg.get("sha256"))

        telemetry_records: list[dict[str, Any]] = []
        seen_hashes = set(baseline_hashes)
        iou_threshold = float(cfg.get("dedup_iou_threshold", 0.8))

        for dirname, label in categories.items():
            category_dir = ctx.paths.telemetry_labeled_dir / dirname
            if not category_dir.exists():
                warnings.append(f"missing category directory: {category_dir}")
                continue
            for wav in sorted(category_dir.rglob("*.wav")):
                try:
                    self._validate_wav(wav)
                    digest = self._sha256_file(wav)
                    if digest in seen_hashes:
                        warnings.append(f"duplicate hash skipped: {wav}")
                        continue
                    rec = self._make_record(wav, dirname, label, digest, ctx.paths.project_root)
                    if self._iou_duplicate(rec, telemetry_records, iou_threshold):
                        warnings.append(f"duplicate IOU skipped: {wav}")
                        continue
                    seen_hashes.add(digest)
                    telemetry_records.append(rec)
                except Exception as exc:
                    warnings.append(f"invalid sample skipped {wav}: {exc}")

        combined = baseline_records + telemetry_records
        if not combined:
            return StageResult(
                stage=self.name,
                status="failed",
                metrics={"records": 0},
                warnings=warnings + ["no records to promote"],
                elapsed_s=time.monotonic() - started,
            )

        groups = [self._group_key(r) for r in combined]
        split_labels = self._split(combined, groups, cfg)
        records = []
        for rec, split in zip(combined, split_labels):
            out = dict(rec)
            out["split"] = split
            records.append(out)

        version = ctx.artifacts.next_version("dataset_v", ".jsonl")
        manifest_ref: ArtifactRef | None = None
        provenance_ref: ArtifactRef | None = None

        if not ctx.dry_run:
            manifest_ref = ctx.artifacts.commit_manifest(records, version=version)
            provenance = {
                "stage": self.name.value,
                "baseline_digest": baseline_digest,
                "manifest": {
                    "version": version,
                    "cas_hash": manifest_ref.cas_hash,
                    "name": manifest_ref.name,
                },
                "counts": {
                    "baseline": len(baseline_records),
                    "telemetry": len(telemetry_records),
                    "total": len(records),
                    "train": split_labels.count("train"),
                    "holdout": split_labels.count("holdout"),
                },
                "warnings": warnings,
                "config": cfg,
            }
            provenance_ref = ctx.artifacts.write_json("provenance.json", provenance)

        return StageResult(
            stage=self.name,
            status="dry_run" if ctx.dry_run else "ok",
            outputs={
                "manifest": manifest_ref,
                "provenance": provenance_ref,
            },
            metrics={
                "baseline_records": len(baseline_records),
                "telemetry_records": len(telemetry_records),
                "total_records": len(records),
                "train_records": split_labels.count("train"),
                "holdout_records": split_labels.count("holdout"),
            },
            warnings=warnings,
            elapsed_s=time.monotonic() - started,
        )

    def _read_jsonl(self, path: Path, warnings: list[str]) -> list[dict[str, Any]]:
        if not path.exists():
            warnings.append(f"baseline manifest not found: {path}")
            return []
        out: list[dict[str, Any]] = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"bad baseline jsonl {path}:{lineno}: {exc}")
                continue
            rec.setdefault("source", "baseline")
            out.append(rec)
        return out

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _validate_wav(self, path: Path) -> None:
        info = sf.info(str(path))
        if info.format != "WAV":
            raise ValueError(f"expected WAV, got {info.format}")
        if info.samplerate != 16000:
            raise ValueError(f"expected 16000 Hz, got {info.samplerate}")
        if info.channels != 1:
            raise ValueError(f"expected mono, got {info.channels} channels")
        if info.subtype != "PCM_16":
            raise ValueError(f"expected PCM_16, got {info.subtype}")
        if info.frames <= 0:
            raise ValueError("empty audio file")

    def _make_record(
        self,
        path: Path,
        category: str,
        label: str,
        digest: str,
        project_root: Path,
    ) -> dict[str, Any]:
        try:
            rel = path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            rel = path.resolve().as_posix()
        metadata = self._read_sidecar(path)
        return {
            "id": f"{category}:{digest[:16]}",
            "path": rel,
            "source": "telemetry",
            "category": category,
            "label": label,
            "sha256": digest,
            "metadata": metadata,
        }

    def _read_sidecar(self, wav: Path) -> dict[str, Any]:
        sidecar = wav.with_suffix(".json")
        if not sidecar.exists():
            return {}
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _group_key(self, record: dict[str, Any]) -> str:
        meta = record.get("metadata") or {}
        parts = []
        for key in ("speaker", "station", "date"):
            value = record.get(key) or meta.get(key)
            if value:
                parts.append(f"{key}:{value}")
        if parts:
            return "|".join(parts)
        return f"file:{record.get('path', '')}"

    def _split(
        self,
        records: list[dict[str, Any]],
        groups: list[str],
        cfg: dict[str, Any],
    ) -> list[str]:
        n = len(records)
        if n < 2 or len(set(groups)) < 2:
            return ["train"] * n
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=float(cfg.get("test_size", 0.2)),
            random_state=int(cfg.get("random_state", 42)),
        )
        train_idx, holdout_idx = next(splitter.split(records, groups=groups))
        out = ["train"] * n
        for i in holdout_idx:
            out[int(i)] = "holdout"
        return out

    def _interval(self, record: dict[str, Any]) -> tuple[float, float] | None:
        meta = record.get("metadata") or {}
        start = meta.get("start_s", record.get("start_s"))
        end = meta.get("end_s", record.get("end_s"))
        if start is None or end is None:
            return None
        return float(start), float(end)

    def _iou(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union > 0 else 0.0

    def _iou_duplicate(
        self,
        rec: dict[str, Any],
        existing: list[dict[str, Any]],
        threshold: float,
    ) -> bool:
        ri = self._interval(rec)
        if ri is None:
            return False
        for other in existing:
            if other.get("label") != rec.get("label"):
                continue
            oi = self._interval(other)
            if oi is None:
                continue
            if self._iou(ri, oi) >= threshold:
                return True
        return False
