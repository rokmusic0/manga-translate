import logging
import sys
import warnings
from pathlib import Path

from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level = record.levelname if record.levelname in logger._core.levels else record.levelno
        logger.opt(exception=record.exc_info, depth=6).log(level, record.getMessage())


def configure_logging(verbose: bool = False) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    logging.captureWarnings(True)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    warnings.simplefilter("default")


def add_file_log_handlers(log_dir: Path, verbose: bool = False) -> tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    full_log_path = log_dir / "pipeline.log"
    problem_log_path = log_dir / "problem_images.log"

    logger.add(
        full_log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8",
    )
    logger.add(
        problem_log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=lambda record: bool(record["extra"].get("problem_image")),
        encoding="utf-8",
    )

    return full_log_path, problem_log_path
