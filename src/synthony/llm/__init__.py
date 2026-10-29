"""Provider-agnostic LLM client layer.

Public surface:
    - Neutral types: LLMMessage, LLMResponse, LLMUsage, ToolSpec, ToolCall, ToolResult
    - LLMClient ABC and create_llm_client() factory
    - Pricing helpers: calculate_cost, register_pricing

Provider adapters (OpenAIClient, AnthropicClient) are importable from their
modules but callers should normally go through create_llm_client() so that
provider choice stays configuration, not code.
"""

from .base import LLMClient
from .factory import NO_LLM_PROVIDERS, create_llm_client, resolve_llm_client
from .pricing import calculate_cost, get_pricing, register_pricing
from .types import (
    LLMMessage,
    LLMResponse,
    LLMUsage,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMUsage",
    "StopReason",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "calculate_cost",
    "create_llm_client",
    "resolve_llm_client",
    "NO_LLM_PROVIDERS",
    "get_pricing",
    "register_pricing",
]
