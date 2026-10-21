"""Abstract base class for provider-agnostic LLM clients."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from .types import LLMMessage, LLMResponse, ToolSpec


class LLMClient(ABC):
    """Neutral interface every provider adapter implements.

    Agents and the recommendation engine depend only on this interface;
    all provider-specific request/response mapping lives in the adapters.
    """

    provider: str = "unknown"
    model: str = ""

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
        json_mode: bool = False,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Run one completion turn.

        Args:
            messages: Conversation history in neutral form.
            system: System prompt, passed separately from the history.
            tools: Tools the model may call.
            tool_choice: None (auto), "any" (must call some tool), or a tool
                name (must call that tool).
            json_mode: Ask the provider for a JSON object response where
                supported; adapters without native support fall back to
                prompt-level instruction.
            max_tokens: Response token cap.
            temperature: Sampling temperature; adapters may ignore it when the
                configured model does not accept sampling parameters.
        """

    def complete_sync(
        self,
        messages: list[LLMMessage],
        **kwargs: object,
    ) -> LLMResponse:
        """Synchronous convenience wrapper for non-async callers (CLI, engine).

        Must not be called from a running event loop; async callers use
        :meth:`complete` directly.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.complete(messages, **kwargs))  # type: ignore[arg-type]
        raise RuntimeError(
            "complete_sync() called from a running event loop; use 'await complete()' instead"
        )
