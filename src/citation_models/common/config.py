from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return config


def deep_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
