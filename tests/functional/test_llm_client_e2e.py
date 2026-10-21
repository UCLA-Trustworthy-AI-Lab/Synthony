"""End-to-end tests for the provider-agnostic LLM client layer (synthony.llm).

Unlike tests/unit/test_llm_clients.py (mocked provider SDKs), these tests hit
the real OpenAI/Anthropic APIs over the network through the actual public
surface an application uses: create_llm_client() -> LLMClient.complete(), and
ModelRecommendationEngine.recommend(method="llm") with an injected client.

Each provider skips cleanly (pytest.mark.skip, not a failure) when its API
key isn't configured, matching the existing convention in
tests/functional/test_openai_connection.py. Calls are kept small
(max_tokens, a tiny sample dataset) to minimize cost.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest
from dotenv import load_dotenv

from synthony.core.analyzer import StochasticDataAnalyzer
from synthony.llm import LLMMessage, create_llm_client
from synthony.recommender.engine import ModelRecommendationEngine

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SAMPLE_CSV = DATA_DIR / "titanic_sample.csv"

_HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
_HAS_ANTHROPIC_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))


def _tcp_reachable(url: str, timeout: float = 2.0) -> bool:
    """Best-effort check that a host:port from a base URL accepts connections.

    VLLM_URL being configured doesn't mean the server is actually up (it's a
    self-hosted endpoint, unlike the OpenAI/Anthropic SaaS APIs), so this
    guards against a hanging request against a dead LAN endpoint.
    """
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


_VLLM_URL = os.getenv("VLLM_URL")
_VLLM_REACHABLE = bool(_VLLM_URL) and _tcp_reachable(_VLLM_URL)

pytestmark = pytest.mark.requires_llm


def _profile():
    return StochasticDataAnalyzer().analyze(str(SAMPLE_CSV))


@pytest.mark.skipif(
    not _VLLM_REACHABLE, reason=f"vLLM endpoint not reachable at {_VLLM_URL!r}"
)
class TestVLLMClientE2E:
    """Real network calls against a self-hosted vLLM (OpenAI-compatible) endpoint.

    Uses the "vllm" provider alias in create_llm_client(), which reads
    VLLM_URL / VLLM_API_KEY / VLLM_MODEL rather than OPENAI_API_KEY, but
    otherwise goes through the same OpenAIClient adapter (vLLM speaks the
    OpenAI chat-completions protocol).
    """

    def test_complete_roundtrip(self):
        client = create_llm_client(provider="vllm")
        # max_tokens=100, not a tighter budget: the currently-deployed model
        # (gpt-oss-120b) is a reasoning model that spends tokens on a hidden
        # `message.reasoning` field before any visible `content` -- a
        # smaller budget (10 was used previously) gets consumed entirely by
        # reasoning, leaving text=None with stop_reason="max_tokens" even
        # though the call itself succeeded. Verified: 100 reliably leaves
        # room for both.
        response = client.complete_sync(
            [LLMMessage(role="user", content="Say the single word: pong")],
            max_tokens=100,
        )
        assert response.text

    def test_engine_recommend_llm_method(self):
        engine = ModelRecommendationEngine(llm_client=create_llm_client(provider="vllm"))
        result = engine.recommend(
            _profile(), constraints={"cpu_only": True}, method="llm", top_n=2
        )
        assert result.recommended_model.model_name
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        assert result.recommended_model.reasoning


@pytest.mark.skipif(not _HAS_OPENAI_KEY, reason="OPENAI_API_KEY not configured")
class TestOpenAIClientE2E:
    """Real network calls against the OpenAI API via synthony.llm."""

    def test_complete_roundtrip(self):
        client = create_llm_client(provider="openai", model="gpt-4o-mini")
        response = client.complete_sync(
            [LLMMessage(role="user", content="Say the single word: pong")],
            max_tokens=10,
        )
        assert response.text
        assert "pong" in response.text.lower()
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.cost_usd > 0

    def test_engine_recommend_llm_method(self):
        engine = ModelRecommendationEngine(
            llm_client=create_llm_client(provider="openai", model="gpt-4o-mini")
        )
        result = engine.recommend(
            _profile(), constraints={"cpu_only": True}, method="llm", top_n=2
        )
        assert result.recommended_model.model_name
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        assert result.recommended_model.reasoning


@pytest.mark.skipif(not _HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not configured")
class TestAnthropicClientE2E:
    """Real network calls against the Anthropic API via synthony.llm."""

    def test_complete_roundtrip(self):
        client = create_llm_client(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            supports_sampling_params=False,
        )
        response = client.complete_sync(
            [LLMMessage(role="user", content="Say the single word: pong")],
            max_tokens=10,
        )
        assert response.text
        assert "pong" in response.text.lower()
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_engine_recommend_llm_method(self):
        engine = ModelRecommendationEngine(
            llm_client=create_llm_client(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                supports_sampling_params=False,
            )
        )
        result = engine.recommend(
            _profile(), constraints={"cpu_only": True}, method="llm", top_n=2
        )
        assert result.recommended_model.model_name
        assert 0.0 <= result.recommended_model.confidence_score <= 1.0
        assert result.recommended_model.reasoning


@pytest.mark.skipif(
    not (_HAS_OPENAI_KEY and _HAS_ANTHROPIC_KEY),
    reason="Both OPENAI_API_KEY and ANTHROPIC_API_KEY required for cross-provider comparison",
)
def test_both_providers_agree_on_method_field():
    """Both providers should route through the same engine method path."""
    profile = _profile()
    openai_engine = ModelRecommendationEngine(
        llm_client=create_llm_client(provider="openai", model="gpt-4o-mini")
    )
    anthropic_engine = ModelRecommendationEngine(
        llm_client=create_llm_client(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            supports_sampling_params=False,
        )
    )
    openai_result = openai_engine.recommend(
        profile, constraints={"cpu_only": True}, method="llm", top_n=2
    )
    anthropic_result = anthropic_engine.recommend(
        profile, constraints={"cpu_only": True}, method="llm", top_n=2
    )
    assert openai_result.method == anthropic_result.method == "llm"
