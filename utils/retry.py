# utils/retry.py
from __future__ import annotations

import time
import random
from functools import wraps
from typing import Callable, Type, Tuple

import requests
import openai


"""
Decorator for simple exponential backoff retries.

- Retries on OpenAI rate limit / transient errors
- Retries on HTTP 429/5xx for requests
"""
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter: float = 0.5,
    retry_on: Tuple[Type[BaseException], ...] = (
        openai.RateLimitError,
        openai.APIError,
        requests.HTTPError,
    ),
) -> Callable:


    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except retry_on as e:
                    # Special-case HTTP errors
                    if isinstance(e, requests.HTTPError):
                        status = e.response.status_code if e.response is not None else None
                        if status not in (429, 500, 502, 503, 504):
                            raise  # non-retriable
                    if attempt == max_retries - 1:
                        # Give up
                        raise
                    sleep_for = delay + random.random() * jitter
                    time.sleep(sleep_for)
                    delay *= 2
        return wrapper

    return decorator
