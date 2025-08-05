# perf_utils.py

import time
import functools
import asyncio
import logging

# 日志初始化（你可以统一配置也可以在这里初始化）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("perf")


def timeit(func):
    """同步函数计时器装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000  # ms
        logger.info(f"[TimeIt] {func.__name__} took {duration:.3f} ms")
        return result
    return wrapper


def async_timeit(func):
    """异步函数计时器装饰器"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000  # ms
        logger.info(f"[AsyncTimeIt] {func.__name__} took {duration:.3f} ms")
        return result
    return wrapper


def log_duration(name="block"):
    """上下文管理器用来手动记录任意代码块的耗时"""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            duration = (time.perf_counter() - self.start) * 1000
            logger.info(f"[Block] {name} took {duration:.3f} ms")

    return Timer()
