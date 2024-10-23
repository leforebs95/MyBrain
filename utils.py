from functools import wraps
from time import sleep
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """
    Decorator for retrying operations with exponential backoff.

    Args:
        retries (int): Maximum number of retries
        backoff_in_seconds (int): Initial backoff time in seconds
    """

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
