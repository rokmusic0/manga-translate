import logging
import sys
import warnings

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
