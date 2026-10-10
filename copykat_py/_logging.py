"""Progress messages go to the ``copykat_py`` logger.

Applications can configure that logger like any other. When nothing is
configured, the public entry points print progress to stdout as they always
have (see ``default_progress_output``).
"""

import functools
import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import ParamSpec, TextIO, TypeVar

LOGGER_NAME = "copykat_py"

_P = ParamSpec("_P")
_R = TypeVar("_R")


@contextmanager
def log_progress_to(*streams: TextIO) -> Iterator[None]:
    """Also write ``copykat_py`` messages to ``streams``, unformatted, while active.

    Unless the application has set the logger's level, it is lowered to INFO
    for the duration so progress messages are emitted.
    """
    logger = logging.getLogger(LOGGER_NAME)
    handlers = [logging.StreamHandler(stream) for stream in streams]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    previous_level = logger.level
    if previous_level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.setLevel(previous_level)
        for handler in handlers:
            logger.removeHandler(handler)


@contextmanager
def default_progress_output() -> Iterator[None]:
    """Print progress to stdout, unless the application has configured logging.

    "Configured" means a handler is reachable from the ``copykat_py`` logger
    (including the root logger's, e.g. after ``logging.basicConfig()``) or
    the logger's level has been set; then its configuration is left alone.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if logger.hasHandlers() or logger.level != logging.NOTSET:
        yield
        return
    with log_progress_to(sys.stdout):
        yield


def with_default_progress_output(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run ``func`` inside ``default_progress_output()``."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with default_progress_output():
            return func(*args, **kwargs)

    return wrapper
