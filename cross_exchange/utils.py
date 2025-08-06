# perf_utils.py

import time
import functools
import asyncio
import logging

# log initilizing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("perf")


def timeit(func):
    """decoration for just functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000  # ms
        logger.info(f"[TimeIt] {func.__name__} took {duration:.3f} ms")
        return result
    return wrapper


def async_timeit(func):
    """decoration for asyn"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000  # ms
        logger.info(f"[AsyncTimeIt] {func.__name__} took {duration:.3f} ms")
        return result
    return wrapper


def log_duration(name="block"):
    """context management for timing functions"""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            duration = (time.perf_counter() - self.start) * 1000
            logger.info(f"[Block] {name} took {duration:.3f} ms")

    return Timer()
