"""Best-effort per-token pricing for cost accounting.

Prices are USD per 1M tokens as (input, output). The table is advisory:
unknown models (self-hosted vLLM/Ollama endpoints, new releases) cost 0.0 and
log a one-time warning, so cost fields are a lower bound, never an error.
Override or extend via ``register_pricing()`` without touching this table.
"""

from __future__ import annotations

import logging

from .types import LLMUsage

logger = logging.getLogger("synthony.llm")

# USD per 1M tokens: model prefix -> (input, output). Longest prefix wins,
# so point releases inherit their family pricing unless overridden.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o3": (2.00, 8.00),
    "claude-opus-4": (5.00, 25.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (1.00, 5.00),
    "claude-3-5-haiku": (0.80, 4.00),
}

_warned_models: set[str] = set()


def register_pricing(model_prefix: str, input_per_m: float, output_per_m: float) -> None:
    """Register or override pricing for a model prefix."""
    _PRICING[model_prefix] = (input_per_m, output_per_m)


def get_pricing(model: str) -> tuple[float, float] | None:
    """Return (input, output) USD per 1M tokens, or None if unknown."""
    best: str | None = None
    for prefix in _PRICING:
        if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return _PRICING[best] if best else None


def calculate_cost(model: str, usage: LLMUsage) -> float:
    """Compute cost in USD for a usage record; 0.0 for unknown models."""
    pricing = get_pricing(model)
    if pricing is None:
        if model not in _warned_models:
            _warned_models.add(model)
            logger.warning("No pricing known for model %r; reporting cost as 0.0", model)
        return 0.0
    input_rate, output_rate = pricing
    return (usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1_000_000
