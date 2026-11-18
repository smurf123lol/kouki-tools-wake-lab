"""HTTP-клиент API Kouki для wake-lab.

Правила (T159 §2.4): ``base_url`` из конфига, токен — только из окружения
(``KOUKI_API_TOKEN``), токен не логируется и не пишется в файлы; ``retries`` с
backoff только для 5xx и сетевых ошибок; 401 и 404 дают понятные тексты.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

_TOKENS_ENV = "KOUKI_API_TOKEN"
_BODY_LIMIT = 500


class ApiError(RuntimeError):
    """Ошибка обращения к API Kouki."""


class ApiAuthError(ApiError):
    """API ответил 401: нужен токен."""


class ApiNotFoundError(ApiError):
    """API ответил 404: объект не найден."""


class KoukiApiClient:
    """Минимальный клиент ``/api/v1/telemetry/*``.

    Токен берётся из аргумента ``token`` или (если не задан) из env
    ``KOUKI_API_TOKEN``. Пустой токен не отправляется: сервер сам ответит 401,
    и клиент превратит это в понятную ошибку.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout_s: float = 10.0,
        retries: int = 2,
        *,
        sleep: Callable[[float], None] = time.sleep,
        backoff_base_s: float = 0.5,
    ) -> None:
        if not base_url:
            raise ValueError("base_url обязателен: укажите api.base_url в конфиге")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.retries = max(0, int(retries))
        self._sleep = sleep
        self._backoff_base_s = float(backoff_base_s)
        resolved = token if token is not None else os.environ.get(_TOKENS_ENV, "")
        self._token = resolved or ""

    # --- публичные методы -------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """``GET /api/v1/telemetry/stats``."""
        return self._json("GET", "/api/v1/telemetry/stats")

    def review_queue(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """``GET /api/v1/telemetry/review-queue``."""
        return self._json(
            "GET",
            "/api/v1/telemetry/review-queue",
            params={"limit": limit, "offset": offset},
        )

    def labeled(
        self, category: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """``GET /api/v1/telemetry/labeled?category=...`` (П4, модуль telemetry)."""
        return self._json(
            "GET",
            "/api/v1/telemetry/labeled",
            params={"category": category, "limit": limit, "offset": offset},
        )

    def audio(self, sample_id: str) -> bytes:
        """``GET /api/v1/telemetry/audio/{id}`` — WAV в байтах (Range не нужен)."""
        return self._request("GET", f"/api/v1/telemetry/audio/{_quote(sample_id)}")

    def classify(self, sample_id: str, label: str) -> dict[str, Any]:
        """``POST /api/v1/telemetry/classify/{id}``."""
        return self._json(
            "POST",
            f"/api/v1/telemetry/classify/{_quote(sample_id)}",
            body={"label": label},
        )

    def undo(self, sample_id: str) -> dict[str, Any]:
        """``POST /api/v1/telemetry/undo/{id}``."""
        return self._json(
            "POST", f"/api/v1/telemetry/undo/{_quote(sample_id)}", body={}
        )

    # --- внутреннее -------------------------------------------------------

    def _json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        raw = self._request(method, path, params=params, body=body)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(f"API вернул не JSON: {exc}") from None

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> bytes:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        attempts = self.retries + 1
        last_error: str = ""
        for attempt in range(attempts):
            request = urllib.request.Request(
                url, data=data, headers=headers, method=method
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                status = exc.code
                error_body = _read_body(exc)
                if status == 401:
                    raise ApiAuthError(
                        "API требует токен: задайте KOUKI_API_TOKEN"
                    ) from None
                if status == 404:
                    raise ApiNotFoundError("сэмпл не найден") from None
                if status >= 500 and attempt < attempts - 1:
                    last_error = f"API ответил {status}: {error_body}"
                    self._sleep(self._backoff_base_s * (2**attempt))
                    continue
                raise ApiError(f"API ответил {status}: {error_body}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                if attempt < attempts - 1:
                    last_error = f"API недоступен: {reason}"
                    self._sleep(self._backoff_base_s * (2**attempt))
                    continue
                raise ApiError(
                    f"API недоступен: {reason} — проверьте api.base_url "
                    "и что сервер Kouki запущен"
                ) from None
        raise ApiError(last_error or "API недоступен")


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _read_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:  # pragma: no cover - защита от нечитаемого тела
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) > _BODY_LIMIT:
        return text[:_BODY_LIMIT] + "…"
    return text
