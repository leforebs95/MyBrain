from typing import Dict, Optional
import logging
from dataclasses import dataclass, field
from functools import wraps
from time import sleep
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RetryStrategy:
    """
    Defines the retry behavior for failed operations.

    Attributes:
        max_retries (int): Maximum number of retry attempts
        initial_delay (float): Initial delay between retries in seconds
        max_delay (float): Maximum delay between retries in seconds
        backoff_factor (float): Factor to increase delay between retries
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a specific retry attempt."""
        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)


def retry_with_backoff(retry_strategy: RetryStrategy = RetryStrategy()):
    """
    Decorator for retrying operations with exponential backoff.

    Args:
        retries (int): Maximum number of retries
        backoff_in_seconds (int): Initial backoff time in seconds
    """
    retries = retry_strategy.max_retries
    backoff_in_seconds = retry_strategy.backoff_factor

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries:
                        raise e
                    sleep_time = backoff_in_seconds * 2**i
                    logger.warning(
                        f"Attempt {i + 1} failed: {str(e)}. Retrying in {sleep_time} seconds..."
                    )
                    sleep(sleep_time)
            return None

        return wrapper

    return decorator
