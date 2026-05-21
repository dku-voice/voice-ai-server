"""
app/services/threadpool.py - 무거운 작업용 threadpool 제한

Starlette 기본 threadpool은 넉넉해서 모델 추론이 한꺼번에 몰리면
CPU/RAM을 먼저 다 써버릴 수 있다. 실제 모델 실행 수만 작게 제한한다.
"""
from collections.abc import Callable
from typing import Any

import anyio

from app.config import LLM_THREAD_LIMIT, MODEL_THREAD_LIMIT

_model_limiter: anyio.CapacityLimiter | None = None
_llm_limiter: anyio.CapacityLimiter | None = None


def _get_model_limiter() -> anyio.CapacityLimiter:
    global _model_limiter
    if _model_limiter is None:
        _model_limiter = anyio.CapacityLimiter(MODEL_THREAD_LIMIT)
    return _model_limiter


def _get_llm_limiter() -> anyio.CapacityLimiter:
    global _llm_limiter
    if _llm_limiter is None:
        _llm_limiter = anyio.CapacityLimiter(LLM_THREAD_LIMIT)
    return _llm_limiter


async def run_model_task(func: Callable[..., Any], *args: Any) -> Any:
    """STT, VAD, DeepFace 같은 로컬 모델 작업을 제한해서 실행한다."""
    return await anyio.to_thread.run_sync(func, *args, limiter=_get_model_limiter())


async def run_llm_task(func: Callable[..., Any], *args: Any) -> Any:
    """OpenAI 호출처럼 블로킹 I/O가 있는 작업을 제한해서 실행한다."""
    return await anyio.to_thread.run_sync(func, *args, limiter=_get_llm_limiter())
