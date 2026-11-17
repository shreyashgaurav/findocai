import logging
import random
import re
import time

import httpx
from google.genai import errors

logger = logging.getLogger(__name__)

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4  # 1 try + 3 retries
BASE_DELAY = 1.0  # in sec; the backoff ceiling doubles from here
MAX_DELAY = 30.0  # max duration b/w attempts


def is_retryable(exc):
    # Gives true if failure is temporary and same call can succeed later
    if isinstance(exc, errors.APIError):
        return exc.code in RETRYABLE_STATUSES
    return isinstance(exc, httpx.TransportError)


def server_retry_delay(exc):
    body = getattr(exc, "details", None)
    if not isinstance(body, dict):
        return None
    for item in body.get("error", {}).get("details", []):
        if item.get("@type", "").endswith("RetryInfo"):
            match = re.fullmatch(r"([\d.]+)s", item.get("retryDelay", ""))
            if match:
                return float(match.group(1))
    return None


def backoff_delay(attempt):
    # Exponential backoff with full jitter -> a random wait in this tange [0, ceiling].
    ceiling = min(MAX_DELAY, BASE_DELAY * 2**attempt)  # 1, 2, 4, 8, ... capped
    return random.uniform(0, ceiling)


def with_retries(fn, *args, **kwargs):
    for attempt in range(MAX_ATTEMPTS):  # attempt = 0, 1, 2, 3
        try:
            return fn(*args, **kwargs)  # success- return straight away
        except Exception as exc:
            if not is_retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                raise

            delay = server_retry_delay(exc)
            if delay is None:
                delay = backoff_delay(attempt)  # server didn't say: back off
            elif delay > MAX_DELAY:
                raise  # daily quota exhausted

            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs.",
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
