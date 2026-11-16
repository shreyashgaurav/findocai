import pytest
from google.genai import errors

from findocai import retry


@pytest.fixture
def sleeps(monkeypatch):
    """Record each sleep instead of sleeping, so tests run instantly."""
    recorded = []
    monkeypatch.setattr(retry.time, "sleep", recorded.append)
    return recorded


def flaky(*failures):
    """A fake API call: raises each given error once, in order, then returns 'ok'."""
    remaining = list(failures)
    calls = []

    def call():
        calls.append(1)  # count every attempt
        if remaining:
            raise remaining.pop(0)
        return "ok"

    return call, calls


def overloaded():
    """The 503 Gemini returned to you in Step 0."""
    return errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})


def test_retries_transient_errors_then_succeeds(sleeps):
    call, calls = flaky(overloaded(), overloaded())
    assert retry.with_retries(call) == "ok"
    assert len(calls) == 3  # failed, failed, succeeded
    assert len(sleeps) == 2  # one wait before each retry


def test_does_not_retry_bad_requests(sleeps):
    bad = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
    call, calls = flaky(bad)
    with pytest.raises(errors.ClientError):
        retry.with_retries(call)
    assert len(calls) == 1  # no second attempt
    assert sleeps == []


def test_gives_up_after_max_attempts(sleeps):
    call, calls = flaky(*[overloaded() for _ in range(10)])
    with pytest.raises(errors.ServerError):
        retry.with_retries(call)
    assert len(calls) == retry.MAX_ATTEMPTS


def test_waits_as_long_as_the_server_asks(sleeps):
    rate_limited = errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}
                ],
            }
        },
    )
    call, _ = flaky(rate_limited)
    assert retry.with_retries(call) == "ok"
    assert sleeps == [7.0]  # the server's delay, not a random backoff