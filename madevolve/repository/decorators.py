"""
Database decorators for resilient operations.
"""

import functools
import logging
import sqlite3
import time
from typing import Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


def db_retry(
    max_attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying database operations on transient failures.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_error = e
                    if "database is locked" in str(e):
                        logger.warning(
                            f"Database locked, attempt {attempt + 1}/{max_attempts}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
                except sqlite3.IntegrityError as e:
                    # Don't retry integrity errors
                    raise

            raise last_error

        return wrapper
    return decorator


def transaction(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to wrap function in a database transaction.

    Expects the first argument (after self) to be a database connection.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        conn = getattr(self, "_conn", None)
        if conn is None:
            return func(self, *args, **kwargs)

        try:
            result = func(self, *args, **kwargs)
            conn.commit()
            return result
        except Exception as e:
            conn.rollback()
            raise

    return wrapper
