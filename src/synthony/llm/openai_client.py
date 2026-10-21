"""LLMClient adapter for OpenAI-compatible chat-completions endpoints.

Covers OpenAI itself plus any server speaking the same protocol (vLLM,
Ollama, LM Studio, ...) via ``base_url``, preserving Synthony's existing
vLLM fallback behavior.
"""

from __future__ import annotations

import json
from typing import Any

from .base import LLMClient
from .pricing import calculate_cost
from .types import LLMMessage, LLMResponse, LLMUsage, StopReason, ToolCall, ToolSpec

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAIClient(LLMClient):
    """Adapter for the OpenAI Python SDK (chat completions)."""

    provider = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        timeout: float | None = 120.0,
        max_retries: int = 2,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIClient requires the openai package. "
                "Install with: pip install 'synthony[llm]'"
            ) from e

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self.base_url = base_url

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
        payload_messages = self._to_wire_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
            if tool_choice == "any":
                kwargs["tool_choice"] = "required"
            elif tool_choice:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        response = await self._client.chat.completions.create(**kwargs)
        return self._from_wire_response(response)

    @staticmethod
    def _to_wire_messages(messages: list[LLMMessage], system: str | None) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                wire.append(entry)
            elif msg.role == "tool":
                for result in msg.tool_results:
                    wire.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.tool_call_id,
                            "content": result.content,
                        }
                    )
            else:
                wire.append({"role": "user", "content": msg.content or ""})
        return wire

    def _from_wire_response(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                arguments = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        usage = LLMUsage()
        if response.usage:
            cached = 0
            details = getattr(response.usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                cache_read_tokens=cached,
            )
            usage.cost_usd = calculate_cost(self.model, usage)

        return LLMResponse(
            text=message.content,
            tool_calls=tool_calls,
            stop_reason=_STOP_REASONS.get(choice.finish_reason or "", "other"),
            usage=usage,
            model=getattr(response, "model", self.model) or self.model,
        )
