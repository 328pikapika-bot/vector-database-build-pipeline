"""
指数退避 + 抖动重试工具
"""
import asyncio
import random
from typing import Callable, Awaitable, TypeVar
from app.config import settings
from app.observability.logger import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


async def with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
) -> T:
    """
    指数退避重试执行异步函数

    Args:
        func: 异步可调用对象
        max_retries: 最大重试次数（默认取自 settings.MAX_RETRY）
        base_delay: 基础延迟秒数（默认取自 settings.RETRY_BASE_DELAY_SEC）
        max_delay: 最大延迟秒数（默认取自 settings.RETRY_MAX_DELAY_SEC）
        jitter: 是否启用随机抖动
        retryable_exceptions: 可重试的异常类型元组

    Returns:
        函数返回值

    Raises:
        最后一次尝试的异常（达到最大重试次数后）
    """
    max_retries = max_retries or settings.MAX_RETRY
    base_delay = base_delay or settings.RETRY_BASE_DELAY_SEC
    max_delay = max_delay or settings.RETRY_MAX_DELAY_SEC

    last_exception = None

    for attempt in range(max_retries + 1):  # +1 包含首次尝试
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(
                    "retry_exhausted",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(e),
                )
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay *= random.uniform(0.75, 1.25)

            logger.warning(
                "retry_wait",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay_seconds=round(delay, 2),
                error=str(e),
            )
            await asyncio.sleep(delay)

    # 理论上不会到这里
    raise last_exception  # type: ignore


def classify_error(error: Exception) -> str:
    """
    将异常分类为 RETRYABLE 或 NON_RETRYABLE
    """
    import httpx

    # HTTP 错误
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 429:
            return "RETRYABLE"       # 限流
        if status >= 500:
            return "RETRYABLE"       # 服务端错误
        if status == 400 or status == 413:
            return "NON_RETRYABLE"   # 请求参数错误 / payload 过大

    # 网络错误
    if isinstance(error, (ConnectionError, TimeoutError)):
        return "RETRYABLE"

    # httpx 网络错误
    if isinstance(error, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)):
        return "RETRYABLE"

    # 默认不可重试
    return "NON_RETRYABLE"
