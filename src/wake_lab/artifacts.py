from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ArtifactRef:
    cas_hash: str
    name: str | None
    size_bytes: int


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.cas_dir = self.root / "cas"
        self.cas_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _cas_path(self, digest: str) -> Path:
        return self.cas_dir / digest

    def _append_index(self, ref: ArtifactRef) -> None:
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ref), sort_keys=True, ensure_ascii=False) + "\n")

    def put_bytes(self, data: bytes, name: str | None = None) -> ArtifactRef:
        digest = self._hash_bytes(data)
        path = self._cas_path(digest)
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        ref = ArtifactRef(cas_hash=digest, name=name, size_bytes=len(data))
        self._append_index(ref)
        return ref

    def put_file(self, path: str | Path, name: str | None = None) -> ArtifactRef:
        p = Path(path)
        return self.put_bytes(p.read_bytes(), name=name or p.name)

    def resolve(self, ref: str | ArtifactRef) -> Path:
        digest = ref.cas_hash if isinstance(ref, ArtifactRef) else ref
        p = self._cas_path(digest)
        if not p.exists():
            raise FileNotFoundError(f"artifact not found: {digest}")
        return p

    def get(self, ref: str | ArtifactRef) -> bytes:
        return self.resolve(ref).read_bytes()

    def list_named(self, name_prefix: str | None = None) -> list[ArtifactRef]:
        if not self.index_path.exists():
            return []
        out: list[ArtifactRef] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            name = d.get("name")
            if name is None:
                continue
            if name_prefix is not None and not name.startswith(name_prefix):
                continue
            out.append(ArtifactRef(cas_hash=d["cas_hash"], name=name, size_bytes=d["size_bytes"]))
        return out

    def next_version(self, prefix: str = "dataset_v", suffix: str = ".jsonl") -> int:
        pattern = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$")
        max_version = 0
        for ref in self.list_named(prefix):
            m = pattern.match(ref.name or "")
            if m:
                max_version = max(max_version, int(m.group(1)))
        return max_version + 1

    def commit_manifest(self, records: Iterable[dict[str, Any]], version: int | None = None) -> ArtifactRef:
        if version is None:
            version = self.next_version()
        name = f"dataset_v{version}.jsonl"
        content = "".join(
            json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records
        )
        return self.put_bytes(content.encode("utf-8"), name=name)

    def write_json(self, name: str, obj: Any) -> ArtifactRef:
        data = json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
        return self.put_bytes(data, name=name)
