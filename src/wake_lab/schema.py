from __future__ import annotations

from typing import Any

KNOWN_CATEGORIES = ("TP", "CONFUSER", "SPEECH_NEG", "NOISE")

_ROOT_KEYS = ("project_root", "paths", "api", "baseline", "promote")
_PATHS_KEYS = ("baseline", "telemetry_labeled", "artifacts")
_API_KEYS = ("base_url", "timeout_s", "retries")
_BASELINE_KEYS = ("manifest", "sha256")
_PROMOTE_KEYS = (
    "categories",
    "dedup_iou_threshold",
    "confuser_ratio",
    "test_size",
    "random_state",
)


class ConfigError(ValueError):
    """Ошибка конфигурации.

    Сообщение всегда начинается с полного пути ключа, например
    ``promote.test_size: ожидается число в интервале (0, 1)``.
    """


def _error(path: str, message: str) -> ConfigError:
    return ConfigError(f"{path}: {message}")


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _error(key, "ожидается отображение (dict)")
    return value


def _check_unknown_keys(section: dict[str, Any], allowed: tuple[str, ...], prefix: str) -> None:
    for key in section:
        if key not in allowed:
            full = f"{prefix}.{key}" if prefix else str(key)
            raise _error(full, "неизвестный ключ")


def _require_string(section: dict[str, Any], key: str, prefix: str) -> None:
    full = f"{prefix}.{key}"
    if key not in section:
        raise _error(full, "обязательный ключ отсутствует")
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise _error(full, "ожидается непустая строка")


def _optional_string(section: dict[str, Any], key: str, prefix: str) -> None:
    if key not in section:
        return
    value = section[key]
    if value is None:
        return
    if not isinstance(value, str):
        raise _error(f"{prefix}.{key}", "ожидается строка")


def _number(section: dict[str, Any], key: str, prefix: str) -> float | None:
    if key not in section:
        return None
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _error(f"{prefix}.{key}", "ожидается число")
    return float(value)


def validate_config(config: Any) -> dict[str, Any]:
    """Проверить конфиг wake-lab и вернуть его же.

    Любая проблема поднимает :class:`ConfigError` с полным путём ключа
    (``paths.baseline``, ``promote.categories.FOO`` и т.п.).
    """
    if not isinstance(config, dict):
        raise _error("конфиг", "ожидается отображение (dict)")
    _check_unknown_keys(config, _ROOT_KEYS, "")
    _optional_string(config, "project_root", "")

    paths = _section(config, "paths")
    if not paths:
        raise _error("paths", "обязательная секция отсутствует")
    _check_unknown_keys(paths, _PATHS_KEYS, "paths")
    for key in _PATHS_KEYS:
        _require_string(paths, key, "paths")

    api = _section(config, "api")
    if not api:
        raise _error("api", "обязательная секция отсутствует")
    _check_unknown_keys(api, _API_KEYS, "api")
    _require_string(api, "base_url", "api")
    timeout_s = _number(api, "timeout_s", "api")
    if timeout_s is not None and timeout_s <= 0:
        raise _error("api.timeout_s", "ожидается число больше нуля")
    retries = _number(api, "retries", "api")
    if retries is not None and (retries < 0 or retries != int(retries)):
        raise _error("api.retries", "ожидается целое число >= 0")

    baseline = _section(config, "baseline")
    _check_unknown_keys(baseline, _BASELINE_KEYS, "baseline")
    if "manifest" in baseline:
        _require_string(baseline, "manifest", "baseline")
    _optional_string(baseline, "sha256", "baseline")

    promote = _section(config, "promote")
    _check_unknown_keys(promote, _PROMOTE_KEYS, "promote")
    if "categories" in promote:
        categories = promote["categories"]
        if not isinstance(categories, dict) or not categories:
            raise _error("promote.categories", "ожидается непустое отображение категорий")
        for category, label in categories.items():
            full = f"promote.categories.{category}"
            if category not in KNOWN_CATEGORIES:
                raise _error(
                    full,
                    "неизвестная категория, допустимы: " + ", ".join(KNOWN_CATEGORIES),
                )
            if not isinstance(label, str) or not label.strip():
                raise _error(full, "ожидается непустая строка-метка")

    test_size = _number(promote, "test_size", "promote")
    if test_size is not None and not 0 < test_size < 1:
        raise _error("promote.test_size", "ожидается число в интервале (0, 1)")
    dedup = _number(promote, "dedup_iou_threshold", "promote")
    if dedup is not None and not 0 < dedup < 1:
        raise _error("promote.dedup_iou_threshold", "ожидается число в интервале (0, 1)")
    confuser_ratio = _number(promote, "confuser_ratio", "promote")
    if confuser_ratio is not None and not 0 <= confuser_ratio < 1:
        raise _error("promote.confuser_ratio", "ожидается число в интервале [0, 1)")
    random_state = _number(promote, "random_state", "promote")
    if random_state is not None and random_state != int(random_state):
        raise _error("promote.random_state", "ожидается целое число")

    return config
