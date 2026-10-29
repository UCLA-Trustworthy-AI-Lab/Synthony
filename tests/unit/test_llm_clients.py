"""Unit tests for the provider-agnostic LLM client layer (synthony.llm)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synthony.llm import (
    LLMMessage,
    LLMResponse,
    LLMUsage,
    ToolCall,
    ToolResult,
    ToolSpec,
    calculate_cost,
    create_llm_client,
    get_pricing,
    register_pricing,
    resolve_llm_client,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def test_usage_accumulation():
    a = LLMUsage(input_tokens=100, output_tokens=20, cache_read_tokens=5, cost_usd=0.01)
    b = LLMUsage(input_tokens=50, output_tokens=10, cost_usd=0.005)
    total = a + b
    assert total.input_tokens == 150
    assert total.output_tokens == 30
    assert total.cache_read_tokens == 5
    assert total.cost_usd == pytest.approx(0.015)
    assert total.total_tokens == 180


def test_response_to_assistant_message_preserves_raw():
    raw = [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "hi"}]
    response = LLMResponse(
        text="hi",
        tool_calls=[ToolCall(id="t1", name="foo", arguments={"x": 1})],
        raw_content=raw,
    )
    msg = response.to_assistant_message()
    assert msg.role == "assistant"
    assert msg.content == "hi"
    assert msg.tool_calls[0].name == "foo"
    assert msg.raw == raw


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def test_pricing_longest_prefix_wins():
    assert get_pricing("gpt-4o-mini-2024-07-18") == (0.15, 0.60)
    assert get_pricing("gpt-4o-2024-08-06") == (2.50, 10.00)


def test_unknown_model_costs_zero():
    usage = LLMUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert calculate_cost("some-vllm-model", usage) == 0.0


def test_cost_math_and_registration():
    register_pricing("test-model", 1.0, 2.0)
    usage = LLMUsage(input_tokens=500_000, output_tokens=250_000)
    assert calculate_cost("test-model-v2", usage) == pytest.approx(0.5 + 0.5)


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


def _openai_response(content=None, tool_calls=None, finish_reason="stop", prompt=10, completion=5):
    return SimpleNamespace(
        model="gpt-4o-test",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
        ),
    )


@pytest.fixture
def openai_client():
    pytest.importorskip("openai")
    from synthony.llm.openai_client import OpenAIClient

    client = OpenAIClient(api_key="test-key", model="gpt-4o")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
    )
    return client


@pytest.mark.asyncio
async def test_openai_text_roundtrip(openai_client):
    openai_client._client.chat.completions.create.return_value = _openai_response(content="hello")
    result = await openai_client.complete(
        [LLMMessage(role="user", content="hi")], system="be brief", temperature=0.3
    )
    assert result.text == "hello"
    assert result.stop_reason == "end_turn"
    assert result.usage.input_tokens == 10
    assert result.usage.cache_read_tokens == 3

    kwargs = openai_client._client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "be brief"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hi"}
    assert kwargs["temperature"] == 0.3


@pytest.mark.asyncio
async def test_openai_tool_call_parsing(openai_client):
    wire_tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_stats", arguments='{"columns": ["a"]}'),
    )
    openai_client._client.chat.completions.create.return_value = _openai_response(
        tool_calls=[wire_tc], finish_reason="tool_calls"
    )
    tools = [ToolSpec(name="get_stats", description="d", input_schema={"type": "object"})]
    result = await openai_client.complete(
        [LLMMessage(role="user", content="go")], tools=tools, tool_choice="any"
    )
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].arguments == {"columns": ["a"]}

    kwargs = openai_client._client.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"] == "required"
    assert kwargs["tools"][0]["function"]["name"] == "get_stats"


@pytest.mark.asyncio
async def test_openai_tool_result_and_assistant_history(openai_client):
    openai_client._client.chat.completions.create.return_value = _openai_response(content="done")
    history = [
        LLMMessage(role="user", content="go"),
        LLMMessage(
            role="assistant",
            tool_calls=[ToolCall(id="call_1", name="get_stats", arguments={"x": 1})],
        ),
        LLMMessage(
            role="tool",
            tool_results=[ToolResult(tool_call_id="call_1", content='{"ok": true}')],
        ),
    ]
    await openai_client.complete(history)
    wire = openai_client._client.chat.completions.create.call_args.kwargs["messages"]
    assert wire[1]["tool_calls"][0]["function"]["arguments"] == json.dumps({"x": 1})
    assert wire[2] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}


@pytest.mark.asyncio
async def test_openai_json_mode_and_forced_tool(openai_client):
    openai_client._client.chat.completions.create.return_value = _openai_response(content="{}")
    tools = [ToolSpec(name="submit", description="d")]
    await openai_client.complete(
        [LLMMessage(role="user", content="x")],
        tools=tools,
        tool_choice="submit",
        json_mode=True,
    )
    kwargs = openai_client._client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit"}}


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


def _anthropic_block(**kwargs):
    block = SimpleNamespace(**kwargs)
    block.model_dump = lambda: kwargs
    return block


def _anthropic_response(blocks, stop_reason="end_turn", input_tokens=10, output_tokens=5):
    return SimpleNamespace(
        model="claude-test",
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=2,
        ),
    )


@pytest.fixture
def anthropic_client():
    pytest.importorskip("anthropic")
    from synthony.llm.anthropic_client import AnthropicClient

    client = AnthropicClient(api_key="test-key", model="claude-test")
    client._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))
    return client


@pytest.mark.asyncio
async def test_anthropic_text_roundtrip(anthropic_client):
    anthropic_client._client.messages.create.return_value = _anthropic_response(
        [_anthropic_block(type="text", text="hello")]
    )
    result = await anthropic_client.complete(
        [LLMMessage(role="user", content="hi")], system="be brief"
    )
    assert result.text == "hello"
    assert result.stop_reason == "end_turn"
    assert result.usage.cache_read_tokens == 2
    assert result.raw_content == [{"type": "text", "text": "hello"}]

    kwargs = anthropic_client._client.messages.create.call_args.kwargs
    assert kwargs["system"] == "be brief"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_anthropic_sampling_params_flag(anthropic_client):
    anthropic_client._client.messages.create.return_value = _anthropic_response(
        [_anthropic_block(type="text", text="x")]
    )
    anthropic_client.supports_sampling_params = False
    await anthropic_client.complete([LLMMessage(role="user", content="hi")], temperature=0.3)
    assert "temperature" not in anthropic_client._client.messages.create.call_args.kwargs

    anthropic_client.supports_sampling_params = True
    await anthropic_client.complete([LLMMessage(role="user", content="hi")], temperature=0.3)
    assert anthropic_client._client.messages.create.call_args.kwargs["temperature"] == 0.3


@pytest.mark.asyncio
async def test_anthropic_tool_use_and_raw_replay(anthropic_client):
    raw_blocks = [
        {"type": "thinking", "thinking": "hmm"},
        {"type": "tool_use", "id": "t1", "name": "get_stats", "input": {"a": 1}},
    ]
    anthropic_client._client.messages.create.return_value = _anthropic_response(
        [
            _anthropic_block(type="thinking", thinking="hmm"),
            _anthropic_block(type="tool_use", id="t1", name="get_stats", input={"a": 1}),
        ],
        stop_reason="tool_use",
    )
    first = await anthropic_client.complete(
        [LLMMessage(role="user", content="go")],
        tools=[ToolSpec(name="get_stats", description="d")],
    )
    assert first.stop_reason == "tool_use"
    assert first.tool_calls[0].arguments == {"a": 1}
    assert first.raw_content == raw_blocks

    # Replay: assistant raw blocks must round-trip verbatim
    anthropic_client._client.messages.create.return_value = _anthropic_response(
        [_anthropic_block(type="text", text="done")]
    )
    history = [
        LLMMessage(role="user", content="go"),
        first.to_assistant_message(),
        LLMMessage(
            role="tool",
            tool_results=[ToolResult(tool_call_id="t1", content="42", is_error=False)],
        ),
    ]
    await anthropic_client.complete(history)
    wire = anthropic_client._client.messages.create.call_args.kwargs["messages"]
    assert wire[1] == {"role": "assistant", "content": raw_blocks}
    assert wire[2]["content"][0]["type"] == "tool_result"
    assert wire[2]["content"][0]["tool_use_id"] == "t1"


@pytest.mark.asyncio
async def test_anthropic_json_mode_appends_instruction(anthropic_client):
    anthropic_client._client.messages.create.return_value = _anthropic_response(
        [_anthropic_block(type="text", text="{}")]
    )
    await anthropic_client.complete(
        [LLMMessage(role="user", content="x")], system="sys", json_mode=True
    )
    system = anthropic_client._client.messages.create.call_args.kwargs["system"]
    assert system.startswith("sys")
    assert "JSON" in system


@pytest.mark.asyncio
async def test_anthropic_forced_tool_choice(anthropic_client):
    anthropic_client._client.messages.create.return_value = _anthropic_response(
        [_anthropic_block(type="tool_use", id="t1", name="submit", input={})],
        stop_reason="tool_use",
    )
    await anthropic_client.complete(
        [LLMMessage(role="user", content="x")],
        tools=[ToolSpec(name="submit", description="d")],
        tool_choice="submit",
    )
    kwargs = anthropic_client._client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit"}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_client(provider="nope")


def test_factory_openai_default(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("SYNTHONY_LLM_PROVIDER", raising=False)
    client = create_llm_client(api_key="k")
    assert client.provider == "openai"
    assert client.model == "gpt-4o"


def test_factory_env_provider_and_model(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("SYNTHONY_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("SYNTHONY_LLM_MODEL", "claude-custom")
    client = create_llm_client(api_key="k")
    assert client.provider == "anthropic"
    assert client.model == "claude-custom"


def test_factory_vllm_alias(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("VLLM_URL", "http://localhost:8001/v1")
    monkeypatch.delenv("SYNTHONY_LLM_MODEL", raising=False)
    client = create_llm_client(provider="vllm")
    assert client.provider == "openai"
    assert client.base_url == "http://localhost:8001/v1"


# ---------------------------------------------------------------------------
# complete_sync + engine integration
# ---------------------------------------------------------------------------

# FakeLLMClient now lives in tests/conftest.py (generalized to a scripted
# response sequence for Phase 3's multi-turn agent/orchestrator tests).
from tests.conftest import FakeLLMClient  # noqa: E402


def test_complete_sync_outside_loop():
    fake = FakeLLMClient(LLMResponse(text="ok"))
    result = fake.complete_sync([LLMMessage(role="user", content="hi")])
    assert result.text == "ok"


@pytest.mark.asyncio
async def test_complete_sync_rejects_running_loop():
    fake = FakeLLMClient(LLMResponse(text="ok"))
    with pytest.raises(RuntimeError, match="running event loop"):
        fake.complete_sync([LLMMessage(role="user", content="hi")])


def test_engine_accepts_injected_client():
    from synthony.recommender.engine import ModelRecommendationEngine

    fake = FakeLLMClient(LLMResponse(text=json.dumps({"recommended_model": "CTGAN"})))
    engine = ModelRecommendationEngine(llm_client=fake)
    assert engine.llm_available is True
    assert engine.openai_client is None
    assert engine.openai_model == "fake-model"

    parsed = engine._call_llm_json("sys", "prompt")
    assert parsed == {"recommended_model": "CTGAN"}
    messages, kwargs = fake.calls[0]
    assert kwargs["system"] == "sys"
    assert kwargs["json_mode"] is True


def test_engine_without_client_unchanged():
    from synthony.recommender.engine import ModelRecommendationEngine

    engine = ModelRecommendationEngine()
    assert engine.llm_client is None
    # Legacy attributes still present for backward compatibility
    assert hasattr(engine, "openai_client")
    assert hasattr(engine, "llm_available")


# ---------------------------------------------------------------------------
# resolve_llm_client: "no LLM" is an option, and missing credentials mean no client
# ---------------------------------------------------------------------------

_CRED_VARS = ("SYNTHONY_LLM_PROVIDER", "SYNTHONY_LLM_BASE_URL", "OPENAI_API_KEY",
              "ANTHROPIC_API_KEY", "VLLM_URL", "VLLM_API_KEY", "VLLM_MODEL")


@pytest.fixture
def no_llm_env(monkeypatch):
    for var in _CRED_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize("provider", ["none", "NONE", "rule_based"])
def test_resolve_explicit_no_llm_returns_none(no_llm_env, provider):
    no_llm_env.setenv("OPENAI_API_KEY", "sk-would-otherwise-be-used")
    assert resolve_llm_client(provider=provider) is None


def test_resolve_env_provider_none_returns_none(no_llm_env):
    no_llm_env.setenv("SYNTHONY_LLM_PROVIDER", "none")
    no_llm_env.setenv("OPENAI_API_KEY", "sk-would-otherwise-be-used")
    assert resolve_llm_client() is None


@pytest.mark.parametrize("provider", [None, "openai", "anthropic", "vllm"])
def test_resolve_without_credentials_returns_none(no_llm_env, provider):
    assert resolve_llm_client(provider=provider) is None


def test_resolve_with_credentials_builds_client(no_llm_env):
    pytest.importorskip("openai")
    no_llm_env.setenv("OPENAI_API_KEY", "sk-test")
    client = resolve_llm_client()
    assert client is not None and client.provider == "openai"
