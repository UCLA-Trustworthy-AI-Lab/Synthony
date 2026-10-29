"""Shared exception types and helpers used across Phase 3 agents/orchestrator."""

from __future__ import annotations


class RunCancelledError(Exception):
    """Raised when a run's cancel_event is observed between stages/turns."""


def is_transient(exc: BaseException) -> bool:
    """Best-effort classification of a retryable LLM/DB error.

    Neither `OpenAIClient` nor `AnthropicClient` (Phase 1) expose a typed
    "this is retryable" marker exception yet, so this does narrow,
    lazily-imported SDK-exception-type matching instead of relying on a
    clean typed hierarchy. A future Phase 1 addendum (a `TransientLLMError`
    wrapper in `synthony.llm`) would let this collapse to one isinstance
    check.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    try:
        import sqlalchemy.exc as sa_exc

        if isinstance(exc, sa_exc.OperationalError):
            return True
    except ImportError:
        pass

    try:
        import openai

        if isinstance(
            exc,
            (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            ),
        ):
            return True
    except ImportError:
        pass

    try:
        import anthropic

        if isinstance(
            exc,
            (
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.InternalServerError,
            ),
        ):
            return True
    except ImportError:
        pass

    return False
