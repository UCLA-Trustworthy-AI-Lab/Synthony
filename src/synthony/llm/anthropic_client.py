"""LLMClient adapter for the Anthropic Messages API."""

from __future__ import annotations

from typing import Any

from .base import LLMClient
from .pricing import calculate_cost
from .types import LLMMessage, LLMResponse, LLMUsage, StopReason, ToolCall, ToolSpec

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


class AnthropicClient(LLMClient):
    """Adapter for the Anthropic Python SDK.

    Provider quirks are handled here via capability flags rather than
    model-name special cases, so agents stay provider-blind:

    - ``supports_sampling_params=False`` drops ``temperature`` for models
      that reject sampling parameters.
    - Assistant content blocks are carried in ``LLMMessage.raw`` and replayed
      verbatim on multi-turn so thinking blocks survive round-trips.
    - ``json_mode`` has no native equivalent here; a JSON instruction is
      appended to the system prompt instead.
    """

    provider = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        base_url: str | None = None,
        timeout: float | None = 120.0,
        max_retries: int = 2,
        supports_sampling_params: bool = True,
    ):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicClient requires the anthropic package. "
                "Install with: pip install 'synthony[llm]'"
            ) from e

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**client_kwargs)
        self.model = model
        self.supports_sampling_params = supports_sampling_params

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
        if json_mode:
            instruction = "Respond with a single valid JSON object and nothing else."
            system = f"{system}\n\n{instruction}" if system else instruction

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire_messages(messages),
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None and self.supports_sampling_params:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
            if tool_choice == "any":
                kwargs["tool_choice"] = {"type": "any"}
            elif tool_choice:
                kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        response = await self._client.messages.create(**kwargs)
        return self._from_wire_response(response)

    @staticmethod
    def _to_wire_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "assistant":
                if msg.raw is not None:
                    # Replay provider-native blocks verbatim (preserves
                    # thinking blocks across turns).
                    wire.append({"role": "assistant", "content": msg.raw})
                    continue
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
                wire.append({"role": "assistant", "content": blocks})
            elif msg.role == "tool":
                wire.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.tool_call_id,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in msg.tool_results
                        ],
                    }
                )
            else:
                wire.append({"role": "user", "content": msg.content or ""})
        return wire

    def _from_wire_response(self, response: Any) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_content: list[Any] = []
        for block in response.content:
            raw_content.append(block.model_dump() if hasattr(block, "model_dump") else block)
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )

        usage = LLMUsage()
        if getattr(response, "usage", None):
            usage = LLMUsage(
                input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            )
            usage.cost_usd = calculate_cost(self.model, usage)

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=_STOP_REASONS.get(response.stop_reason or "", "other"),
            usage=usage,
            model=getattr(response, "model", self.model) or self.model,
            raw_content=raw_content,
        )
