import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Set up Python logging with console and optional file handler."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    return logging.getLogger("smolvla")
