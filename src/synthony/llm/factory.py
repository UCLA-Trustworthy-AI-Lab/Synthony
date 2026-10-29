"""Factory for constructing LLM clients from configuration.

Provider selection is driven by arguments or environment, never by the
calling code — callers depend only on :class:`~synthony.llm.base.LLMClient`.

Environment variables:
    SYNTHONY_LLM_PROVIDER   Provider name (default: "openai")
    SYNTHONY_LLM_MODEL      Model override for the chosen provider
    SYNTHONY_LLM_BASE_URL   Endpoint override (OpenAI-compatible servers)
    OPENAI_API_KEY          API key for the openai provider
    ANTHROPIC_API_KEY       API key for the anthropic provider
    SYNTHONY_LLM_PROVIDER=none  Run without an LLM (see resolve_llm_client)
    VLLM_URL / VLLM_API_KEY / VLLM_MODEL   Existing vLLM configuration, used
                            by the "vllm" provider alias. If VLLM_MODEL is
                            unset, the server is asked what it's serving
                            via GET /models rather than assuming a fixed
                            name (falls back to a hardcoded default only if
                            that call fails).
"""

from __future__ import annotations

import logging
import os

from .base import LLMClient

logger = logging.getLogger(__name__)

# Provider names that mean "run without an LLM" (agentic mode then uses the
# documented rule-based path instead of multi-turn reasoning).
NO_LLM_PROVIDERS = frozenset({"none", "rule_based"})

_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
}

# Last-resort default if VLLM_MODEL isn't set *and* the server can't be
# asked what it's actually serving (unreachable, empty model list, etc.).
# Not meant to be relied on -- a self-hosted vLLM's model changes over
# time (this one used to serve Qwen/Qwen2.5-32B-Instruct; now gpt-oss-120b),
# so `_discover_vllm_model` asking the server directly is the real fix.
_VLLM_MODEL_FALLBACK = "gpt-oss-120b"


def _discover_vllm_model(client) -> str:
    """Ask an already-constructed vLLM-pointed OpenAI client what model
    it's actually serving (its first reported model ID), rather than
    hardcoding a name that goes stale the moment the server's model is
    swapped. Raises if the server call fails or reports no models --
    callers decide what "discovery failed" means for them."""
    models = client.models.list()
    if models.data:
        return models.data[0].id
    raise RuntimeError("vLLM server reported no available models")


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: object,
) -> LLMClient:
    """Create an :class:`LLMClient` for the requested provider.

    ``provider`` falls back to ``SYNTHONY_LLM_PROVIDER`` then "openai".
    "vllm" is an alias for the OpenAI-compatible adapter configured from the
    existing ``VLLM_*`` environment variables. Imports are lazy so optional
    dependencies are only required for the provider actually used.
    """
    provider = (provider or os.getenv("SYNTHONY_LLM_PROVIDER") or "openai").lower()
    model = model or os.getenv("SYNTHONY_LLM_MODEL") or None
    base_url = base_url or os.getenv("SYNTHONY_LLM_BASE_URL") or None

    if provider == "vllm":
        from .openai_client import OpenAIClient

        vllm_api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        vllm_base_url = base_url or os.getenv("VLLM_URL")
        vllm_model = model or os.getenv("VLLM_MODEL")

        if not vllm_model:
            if vllm_base_url:
                try:
                    from openai import OpenAI as _SyncOpenAI

                    probe = _SyncOpenAI(api_key=vllm_api_key, base_url=vllm_base_url)
                    vllm_model = _discover_vllm_model(probe)
                except Exception:
                    vllm_model = _VLLM_MODEL_FALLBACK
            else:
                vllm_model = _VLLM_MODEL_FALLBACK

        return OpenAIClient(
            api_key=vllm_api_key,
            model=vllm_model,
            base_url=vllm_base_url,
            **kwargs,  # type: ignore[arg-type]
        )

    if provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model or _DEFAULT_MODELS["openai"],
            base_url=base_url,
            **kwargs,  # type: ignore[arg-type]
        )

    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            model=model or _DEFAULT_MODELS["anthropic"],
            base_url=base_url,
            **kwargs,  # type: ignore[arg-type]
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}. Supported: openai, anthropic, vllm")


def resolve_llm_client(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: object,
) -> LLMClient | None:
    """Like :func:`create_llm_client`, but returns ``None`` instead of a client
    that can only fail: when the provider is explicitly ``"none"``, or when the
    chosen provider has no credentials configured. Callers treat ``None`` as
    "no LLM" and take the rule-based path. A key that is present but invalid
    still produces a client, so a misconfiguration fails loudly.
    """
    name = (provider or os.getenv("SYNTHONY_LLM_PROVIDER") or "openai").lower()
    if name in NO_LLM_PROVIDERS:
        return None

    has_credentials = {
        "openai": bool(kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")),
        "vllm": bool(kwargs.get("base_url") or os.getenv("VLLM_URL")),
    }.get(name, True)  # unknown names fall through so create_llm_client raises
    if not has_credentials:
        logger.warning(
            "No credentials for LLM provider %r; running without an LLM (rule-based).", name
        )
        return None
    return create_llm_client(provider=name, model=model, **kwargs)
