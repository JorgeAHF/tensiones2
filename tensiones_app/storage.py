"""Utilities for persisting user configuration across sessions."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_STORAGE_ENV_VAR = "TENSIONES_APP_STORAGE"
_DEFAULT_SUBDIR = ".tensiones_app"
_MAPPING_FILENAME = "mapping.json"


def _storage_dir() -> Path:
    """Return the directory where persistent files should be stored."""
    custom_path = os.environ.get(_STORAGE_ENV_VAR)
    if custom_path:
        return Path(custom_path).expanduser().resolve()
    return Path.home() / _DEFAULT_SUBDIR


def _mapping_path() -> Path:
    return _storage_dir() / _MAPPING_FILENAME


def load_last_mapping_text() -> str:
    """Load the last saved mapping, returning an empty JSON object if missing."""
    path = _mapping_path()
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "{}"
    except OSError:
        return "{}"

    try:
        # Ensure the stored mapping is valid JSON before returning it.
        json.loads(content)
    except json.JSONDecodeError:
        return "{}"
    return content


def save_mapping_text(mapping: str | dict[str, Any]) -> None:
    """Persist the provided mapping string or dictionary to disk."""
    if isinstance(mapping, dict):
        serialised = json.dumps(mapping, ensure_ascii=False, indent=2)
    else:
        serialised = mapping
        try:
            json.loads(serialised)
        except json.JSONDecodeError:
            return

    path = _mapping_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialised, encoding="utf-8")
    except OSError:
        # The application should continue working even if persistence fails.
        return
