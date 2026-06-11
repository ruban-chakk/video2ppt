from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(start: Path | None = None) -> Path | None:
    env_path = find_dotenv(start or Path.cwd())
    if not env_path:
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_env_value(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


def find_dotenv(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for folder in [current, *current.parents]:
        candidate = folder / ".env"
        if candidate.exists():
            return candidate
    return None


def strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
