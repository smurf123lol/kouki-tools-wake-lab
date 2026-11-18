from __future__ import annotations

import json
import logging
import struct
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Self
from urllib.parse import parse_qs, urlparse

import pytest

from wake_lab.api.client import (
    ApiAuthError,
    ApiError,
    ApiNotFoundError,
    KoukiApiClient,
)
from wake_lab.artifacts import ArtifactStore
from wake_lab.paths import PathGuard
from wake_lab.stages.base import StageContext
from wake_lab.stages.harvest import HarvestStage

# --- фейковый HTTP-сервер --------------------------------------------------


@dataclass
class FakeRequest:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes


@dataclass
class FakeApi:
    routes: dict[tuple[str, str], object] = field(default_factory=dict)
    requests: list[FakeRequest] = field(default_factory=list)
    attempts: int = 0

    def __post_init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.fake = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def handle(self, request: FakeRequest) -> tuple[int, bytes, str]:
        self.attempts += 1
        self.requests.append(request)
        route = self.routes.get((request.method, request.path))
        if route is None:
            return 404, b'{"error":"not found"}', "application/json"
        if callable(route):
            return route(request)
        return route  # type: ignore[return-value]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        request = FakeRequest(
            method=method,
            path=parsed.path,
            query=parse_qs(parsed.query),
            headers={k.lower(): v for k, v in self.headers.items()},
            body=self.rfile.read(length) if length else b"",
        )
        status, payload, content_type = self.server.fake.handle(request)  # type: ignore[attr-defined]
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:  # тишина в выводе тестов
        pass


def _json(status: int, payload: object) -> tuple[int, bytes, str]:
    return status, json.dumps(payload).encode("utf-8"), "application/json"


def _wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


# --- клиент ----------------------------------------------------------------


def test_client_hits_stats_and_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KOUKI_API_TOKEN", "secret-token-xyz")
    with FakeApi() as api:
        api.routes[("GET", "/api/v1/telemetry/stats")] = _json(
            200, {"totalSamples": 3, "byLabel": {"TP": 1}}
        )
        client = KoukiApiClient(api.base_url, retries=0)
        assert client.stats() == {"totalSamples": 3, "byLabel": {"TP": 1}}
        assert api.requests[0].headers["authorization"] == "Bearer secret-token-xyz"


def test_client_labeled_passes_category_and_paging() -> None:
    with FakeApi() as api:
        api.routes[("GET", "/api/v1/telemetry/labeled")] = _json(200, [{"sampleId": "a"}])
        client = KoukiApiClient(api.base_url, token="t", retries=0)
        assert client.labeled("TP", limit=2, offset=4) == [{"sampleId": "a"}]
        request = api.requests[0]
        assert request.query == {
            "category": ["TP"],
            "limit": ["2"],
            "offset": ["4"],
        }


def test_client_audio_returns_bytes() -> None:
    payload = _wav(b"\x00\x00" * 100)
    with FakeApi() as api:
        api.routes[("GET", "/api/v1/telemetry/audio/abc")] = (200, payload, "audio/wav")
        client = KoukiApiClient(api.base_url, token="t", retries=0)
        assert client.audio("abc") == payload


def test_client_classify_posts_label() -> None:
    with FakeApi() as api:
        api.routes[("POST", "/api/v1/telemetry/classify/abc")] = _json(
            200, {"status": "ok"}
        )
        client = KoukiApiClient(api.base_url, token="t", retries=0)
        assert client.classify("abc", "TP") == {"status": "ok"}
        assert json.loads(api.requests[0].body) == {"label": "TP"}


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "API требует токен: задайте KOUKI_API_TOKEN"),
        (404, "сэмпл не найден"),
        (500, "API ответил 500"),
    ],
)
def test_client_error_texts(status: int, expected: str) -> None:
    with FakeApi() as api:
        api.routes[("GET", "/api/v1/telemetry/audio/abc")] = (
            status,
            b'{"error":"boom"}',
            "application/json",
        )
        client = KoukiApiClient(api.base_url, token="t", retries=0, sleep=lambda _: None)
        with pytest.raises(ApiError, match=expected):
            client.audio("abc")


def test_client_401_and_404_have_specific_types() -> None:
    with FakeApi() as api:
        api.routes[("GET", "/api/v1/telemetry/audio/a")] = _json(401, {"error": "x"})
        api.routes[("GET", "/api/v1/telemetry/audio/b")] = _json(404, {"error": "x"})
        client = KoukiApiClient(api.base_url, token="t", retries=0, sleep=lambda _: None)
        with pytest.raises(ApiAuthError):
            client.audio("a")
        with pytest.raises(ApiNotFoundError):
            client.audio("b")


def test_client_retries_server_errors_twice() -> None:
    calls = {"n": 0}

    def flaky(_request: FakeRequest) -> tuple[int, bytes, str]:
        calls["n"] += 1
        if calls["n"] < 3:
            return _json(500, {"error": "temporary"})
        return _json(200, {"totalSamples": 1})

    with FakeApi() as api:
        api.routes[("GET", "/api/v1/telemetry/stats")] = flaky
        client = KoukiApiClient(api.base_url, token="t", retries=2, sleep=lambda _: None)
        assert client.stats() == {"totalSamples": 1}
        assert calls["n"] == 3


def test_client_does_not_leak_token_into_errors_or_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    token = "super-secret-token-42"
    monkeypatch.setenv("KOUKI_API_TOKEN", token)
    caplog.set_level(logging.DEBUG)
    with FakeApi() as api:
        api.routes[("GET", "/api/v1/telemetry/stats")] = _json(500, {"error": "boom"})
        client = KoukiApiClient(api.base_url, retries=0, sleep=lambda _: None)
        with pytest.raises(ApiError) as excinfo:
            client.stats()
        assert token not in str(excinfo.value)
        assert token not in caplog.text


def test_client_network_error_has_hint() -> None:
    # порт, на котором никто не слушает
    client = KoukiApiClient("http://127.0.0.1:1", token="t", retries=0, sleep=lambda _: None)
    with pytest.raises(ApiError, match="API недоступен"):
        client.stats()


# --- стадия harvest --------------------------------------------------------


def _config(tmp_path: Path, base_url: str) -> dict:
    return {
        "project_root": str(tmp_path),
        "paths": {
            "baseline": "data/baseline",
            "telemetry_labeled": "data/telemetry_harvested",
            "artifacts": "artifacts",
        },
        "api": {"base_url": base_url, "timeout_s": 5, "retries": 0},
        "promote": {"categories": {"TP": "positive", "NOISE": "negative_noise"}},
    }


def _context(
    tmp_path: Path, base_url: str, *, dry_run: bool = False, params: dict | None = None
) -> StageContext:
    config = _config(tmp_path, base_url)
    return StageContext(
        config=config,
        paths=PathGuard(tmp_path, config),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        logger=logging.getLogger("test"),
        dry_run=dry_run,
        params=dict(params or {}),
    )


def _labeled(sample_id: str, sha256: str, category: str = "TP") -> dict:
    return {
        "sampleId": sample_id,
        "sha256": sha256,
        "manualLabel": category,
        "sampleRate": 16000,
        "pcmByteLength": 32000,
        "_audio": _wav(b"\x01\x00" * 16000),
    }


def _install_harvest_routes(api: FakeApi, samples: list[dict]) -> None:
    by_category: dict[str, list[dict]] = {}
    for sample in samples:
        by_category.setdefault(sample["manualLabel"], []).append(sample)

    def labeled(request: FakeRequest) -> tuple[int, bytes, str]:
        category = request.query["category"][0]
        limit = int(request.query["limit"][0])
        page = [
            {key: value for key, value in sample.items() if key != "_audio"}
            for sample in by_category.get(category, [])
        ]
        return _json(200, page[:limit])

    def audio(request: FakeRequest) -> tuple[int, bytes, str]:
        sample_id = request.path.rsplit("/", 1)[-1]
        for sample in samples:
            if sample["sampleId"] == sample_id:
                return 200, sample["_audio"], "audio/wav"
        return _json(404, {"error": "not found"})

    api.routes[("GET", "/api/v1/telemetry/stats")] = _json(200, {"totalSamples": len(samples)})
    api.routes[("GET", "/api/v1/telemetry/labeled")] = labeled
    for sample in samples:
        api.routes[("GET", f"/api/v1/telemetry/audio/{sample['sampleId']}")] = audio


def test_harvest_downloads_samples_and_writes_manifest(tmp_path: Path) -> None:
    samples = [_labeled("s1", "sha-1"), _labeled("s2", "sha-2"), _labeled("n1", "sha-3", "NOISE")]
    with FakeApi() as api:
        _install_harvest_routes(api, samples)
        result = HarvestStage().run(_context(tmp_path, api.base_url, params={"limit": 4}))

    assert result.status == "ok"
    root = tmp_path / "data" / "telemetry_harvested"
    tp_files = sorted((root / "TP").glob("*.wav"))
    assert [p.name for p in tp_files] == ["s1.wav", "s2.wav"]
    assert (root / "NOISE" / "n1.wav").exists()
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 3
    first = manifest[0]
    assert set(first) == {"id", "label", "sha256", "duration_s", "source", "harvested_at"}
    assert first["id"] == "s1"
    assert first["label"] == "TP"
    assert first["duration_s"] == 1.0
    assert first["source"] == "kouki-api"
    assert result.metrics["harvested"] == 3


def test_harvest_is_idempotent_by_sha256(tmp_path: Path) -> None:
    samples = [_labeled("s1", "sha-1")]
    with FakeApi() as api:
        _install_harvest_routes(api, samples)
        ctx = _context(tmp_path, api.base_url, params={"limit": 5})
        first = HarvestStage().run(ctx)
        audio_calls_before = len([r for r in api.requests if "/audio/" in r.path])
        second = HarvestStage().run(ctx)
        audio_calls_after = len([r for r in api.requests if "/audio/" in r.path])

    assert first.metrics["harvested"] == 1
    assert second.metrics["harvested"] == 0
    assert second.metrics["skipped"] == 1
    assert audio_calls_before == audio_calls_after  # повторно не скачивали
    manifest = (tmp_path / "data" / "telemetry_harvested" / "manifest.jsonl").read_text()
    assert len([line for line in manifest.splitlines() if line.strip()]) == 1


def test_harvest_dry_run_writes_nothing(tmp_path: Path) -> None:
    samples = [_labeled("s1", "sha-1"), _labeled("s2", "sha-2")]
    with FakeApi() as api:
        _install_harvest_routes(api, samples)
        result = HarvestStage().run(
            _context(tmp_path, api.base_url, dry_run=True, params={"limit": 2})
        )

    assert result.status == "dry_run"
    assert result.outputs["planned"] == {"TP": 2, "NOISE": 0}
    root = tmp_path / "data" / "telemetry_harvested"
    assert not root.exists() or not any(root.rglob("*.wav"))


def test_harvest_requires_limit(tmp_path: Path) -> None:
    result = HarvestStage().run(_context(tmp_path, "http://127.0.0.1:1"))
    assert result.status == "failed"
    assert "--limit" in result.outputs["error"]


def test_harvest_api_unavailable_returns_error_not_empty_success(tmp_path: Path) -> None:
    result = HarvestStage().run(
        _context(tmp_path, "http://127.0.0.1:1", params={"limit": 1})
    )
    assert result.status == "failed"
    assert "API недоступен" in result.outputs["error"]
    assert "api.base_url" in result.outputs["error"]
