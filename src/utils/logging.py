"""Structured logging utility with console and file handlers."""

import logging
import sys
from pathlib import Path


def get_logger(name: str = "seg", log_file: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger.
        log_file: Optional path to a file where logs will be appended.
        level: Logging level (default INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
