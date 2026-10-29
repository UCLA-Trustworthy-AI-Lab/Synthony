"""Provider-agnostic message, tool, and usage types for LLM clients.

These types are the neutral contract between Synthony agents and any LLM
provider adapter. Nothing in this module depends on a vendor SDK.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


class ToolSpec(BaseModel):
    """A tool the LLM may call, described in provider-neutral terms."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The result of executing a tool call, sent back to the LLM."""

    tool_call_id: str
    content: str
    is_error: bool = False


class LLMMessage(BaseModel):
    """A single conversation message.

    ``raw`` optionally carries provider-native content (e.g. assistant content
    blocks) so adapters can replay history losslessly on multi-turn exchanges.
    Adapters that set it must also be the ones consuming it; other providers
    fall back to the neutral fields.
    """

    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    raw: Any = None


class LLMUsage(BaseModel):
    """Token and cost accounting, accumulable across calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """A completed LLM turn in provider-neutral form."""

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: StopReason = "other"
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model: str = ""
    raw_content: Any = None

    def to_assistant_message(self) -> LLMMessage:
        """Convert this response into the assistant message for history replay."""
        return LLMMessage(
            role="assistant",
            content=self.text,
            tool_calls=self.tool_calls,
            raw=self.raw_content,
        )
