import yaml
from pathlib import Path


def load_config(path: str, overrides: dict = None) -> dict:
    """Load YAML config and merge CLI overrides."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        _deep_merge(config, overrides)

    return config


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
