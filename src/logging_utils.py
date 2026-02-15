from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_logger: logging.Logger | None = None


def setup_logging(
    log_file: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    global _logger

    logger = logging.getLogger("paraphrase_finder")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    if _logger is None:
        return setup_logging()
    return _logger


@contextmanager
def timed_section(name: str) -> Iterator[None]:
    logger = get_logger()
    logger.info(f"Starting: {name}")
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"Completed: {name} ({elapsed:.2f}s)")
