from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


Config = dict[str, Any]


def load_config(path: Path | str) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"config root must be an object: {config_path}")

    return data


def merge_cli_overrides(config: Config, overrides: Config) -> Config:
    merged = copy.deepcopy(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged

