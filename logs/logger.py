"""Structured JSON logging utilities."""

import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from pythonjsonlogger import jsonlogger

DEFAULT_LOG_PATH = Path(__file__).parent / "app.log"


def _build_handler(log_path: Path) -> logging.Handler:
    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(pathname)s %(lineno)d"
    )
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str = "transcriber", level: int = logging.INFO) -> Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.addHandler(_build_handler(DEFAULT_LOG_PATH))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(jsonlogger.JsonFormatter())
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
