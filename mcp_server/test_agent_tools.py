#!/usr/bin/env python3
"""
Test AgentTools (Agentic Mode Phase 5: MCP surface for runs)

Unlike test_server.py/test_protocol.py's loose import/print-only smoke
checks, these are real assertion-driven tests -- self-contained (no import
from tests/conftest.py, which isn't reliably importable from this
directory's separate `cd mcp_server && pytest` CI step) with local
FakeLLMClient/FakeRunner doubles and DB isolation, mirroring the patterns
already used for Phase 3/4's RunManager/Orchestrator tests under tests/.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthony.agents.manager import RunManager  # noqa: E402
from synthony.llm import LLMClient, LLMResponse, ToolCall  # noqa: E402
from synthony.recommender.engine import ModelRecommendationEngine  # noqa: E402
from synthony.runners.base import SynthesisJobResult, SynthesizerRunner  # noqa: E402

from mcp_server.tools.agent_tools import AgentTools  # noqa: E402


class FakeLLMClient(LLMClient):
    """Scripted LLMClient: returns responses in order, repeating the last
    once exhausted."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _submit_recommendation_call(model_name: str = "CTGAN") -> ToolCall:
    return ToolCall(
        id="submit",
        name="submit_recommendation",
        arguments={
            "ranked_candidates": [
                {
                    "model_name": model_name,
                    "confidence": 0.8,
                    "hyperparameter_hints": {},
                    "rationale": "ok",
                }
            ],
            "reasoning": "test",
        },
    )


def _happy_path_responses():
    return [
        LLMResponse(
            tool_calls=[ToolCall(id="c1", name="get_profile_overview", arguments={})],
            stop_reason="tool_use",
        ),
        LLMResponse(
            tool_calls=[ToolCall(id="c2", name="get_rule_based_scores", arguments={})],
            stop_reason="tool_use",
        ),
        LLMResponse(tool_calls=[_submit_recommendation_call()], stop_reason="tool_use"),
    ]


class FakeRunner(SynthesizerRunner):
    """No real subprocess/synthcity -- writes the training CSV back out as
    the "synthetic" parquet so a real DataQualityBenchmark scores it high."""

    def __init__(self):
        self.calls = []

    def supported_models(self):
        return frozenset({"CTGAN", "TVAE", "ARF"})

    async def train_and_generate(self, job, cancel_event=None):
        self.calls.append(job)
        pd.read_csv(job.train_csv_path).to_parquet(job.output_path)
        return SynthesisJobResult(
            ok=True, error=None, train_seconds=0.01, generate_seconds=0.01, output_path=job.output_path
        )


@pytest.fixture
def isolated_db():
    """Point database.py's global connection at a fresh temp SQLite file,
    restoring prior state afterward. Local copy of tests/conftest.py's
    fixture of the same name -- see module docstring for why."""
    from synthony.api import database as db_module
    from synthony.api.database import init_database

    prior_engine, prior_session_local = db_module._engine, db_module._SessionLocal
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        init_database(f"sqlite:///{db_path}")
        yield
    db_module._engine, db_module._SessionLocal = prior_engine, prior_session_local


@pytest.fixture
def dataset_dir(tmp_path):
    """A tmp SYNTHONY_DATA_DIR with one small real CSV in it."""
    df = pd.DataFrame({"num_0": range(50), "cat_0": ["a", "b"] * 25})
    df.to_csv(tmp_path / "toy.csv", index=False)
    return tmp_path


def _build_agent_tools(llm_client, dataset_dir, runner=None):
    engine = ModelRecommendationEngine()
    manager = RunManager(
        engine=engine,
        llm_client_factory=lambda cfg: llm_client,
        runner_factory=(lambda cfg: runner) if runner is not None else None,
    )
    return AgentTools(manager, data_dir=dataset_dir)


def test_tool_definitions():
    """Tool listing includes exactly the 4 new tools, well-formed."""
    engine = ModelRecommendationEngine()
    tools = AgentTools(RunManager(engine=engine))

    names = tools.get_tool_names()
    assert names == ["run_agentic_synthesis", "get_run_status", "list_runs", "cancel_run"]

    defs = tools.get_tool_definitions()
    assert len(defs) == len(names)
    assert {d.name for d in defs} == set(names)
    for d in defs:
        assert d.inputSchema["type"] == "object"


@pytest.mark.asyncio
async def test_run_agentic_synthesis_then_get_run_status_reaches_completed(
    isolated_db, dataset_dir
):
    fake_llm = FakeLLMClient(_happy_path_responses())
    tools = _build_agent_tools(fake_llm, dataset_dir)

    start_result = await tools.execute_tool(
        "run_agentic_synthesis", {"dataset_name": "toy", "mode": "advise"}
    )
    assert start_result["status"] == "pending"
    run_id = start_result["run_id"]

    # advise mode with a 3-response script resolves synchronously fast --
    # poll briefly rather than assuming it's already done the instant
    # start_run's create_task() call returns.
    for _ in range(50):
        status = await tools.execute_tool("get_run_status", {"run_id": run_id})
        if status["status"] in ("completed", "failed", "cancelled"):
            break
        await asyncio.sleep(0.02)

    assert status["status"] == "completed"
    assert status["result"]["ranked_candidates"][0]["model_name"] == "CTGAN"
    assert [s["step_name"] for s in status["steps"]] == [
        "analyze",
        "reason_turn_1",
        "reason_turn_2",
        "reason_turn_3",
    ]


@pytest.mark.asyncio
async def test_run_agentic_synthesis_full_mode_with_fake_runner(isolated_db, dataset_dir):
    fake_llm = FakeLLMClient(_happy_path_responses())
    fake_runner = FakeRunner()
    tools = _build_agent_tools(fake_llm, dataset_dir, runner=fake_runner)

    start_result = await tools.execute_tool(
        "run_agentic_synthesis", {"dataset_name": "toy", "mode": "full"}
    )
    run_id = start_result["run_id"]

    status = None
    for _ in range(50):
        status = await tools.execute_tool("get_run_status", {"run_id": run_id})
        if status["status"] in ("completed", "failed", "cancelled"):
            break
        await asyncio.sleep(0.02)

    assert status["status"] == "completed"
    assert status["result"]["met_threshold"] is True
    assert len(fake_runner.calls) == 1


@pytest.mark.asyncio
async def test_list_runs_includes_started_run(isolated_db, dataset_dir):
    fake_llm = FakeLLMClient(_happy_path_responses())
    tools = _build_agent_tools(fake_llm, dataset_dir)

    start_result = await tools.execute_tool("run_agentic_synthesis", {"dataset_name": "toy"})
    run_id = start_result["run_id"]

    listing = await tools.execute_tool("list_runs", {})
    assert any(r["run_id"] == run_id for r in listing["runs"])
    assert listing["total"] >= 1


@pytest.mark.asyncio
async def test_get_run_status_unknown_run_raises(isolated_db, dataset_dir):
    tools = _build_agent_tools(FakeLLMClient(_happy_path_responses()), dataset_dir)
    with pytest.raises(ValueError, match="not found"):
        await tools.execute_tool("get_run_status", {"run_id": "does-not-exist"})


@pytest.mark.asyncio
async def test_run_agentic_synthesis_unknown_dataset_raises(isolated_db, dataset_dir):
    tools = _build_agent_tools(FakeLLMClient(_happy_path_responses()), dataset_dir)
    with pytest.raises(FileNotFoundError):
        await tools.execute_tool("run_agentic_synthesis", {"dataset_name": "does-not-exist"})


@pytest.mark.asyncio
async def test_cancel_run_refuses_a_foreign_run(isolated_db, dataset_dir):
    """A run created directly (owner=None, e.g. via the REST API or a
    different MCP session) must not be cancellable through this tool."""
    from synthony.api.database import create_run

    tools = _build_agent_tools(FakeLLMClient(_happy_path_responses()), dataset_dir)
    foreign_run = create_run(mode="advise", owner=None)

    with pytest.raises(ValueError, match="not owned by this MCP session"):
        await tools.execute_tool("cancel_run", {"run_id": foreign_run.run_id})


@pytest.mark.asyncio
async def test_cancel_run_unknown_run_raises(isolated_db, dataset_dir):
    tools = _build_agent_tools(FakeLLMClient(_happy_path_responses()), dataset_dir)
    with pytest.raises(ValueError, match="not found"):
        await tools.execute_tool("cancel_run", {"run_id": "does-not-exist"})


@pytest.mark.asyncio
async def test_cancel_run_own_terminal_run_is_a_noop(isolated_db, dataset_dir):
    fake_llm = FakeLLMClient(_happy_path_responses())
    tools = _build_agent_tools(fake_llm, dataset_dir)

    start_result = await tools.execute_tool("run_agentic_synthesis", {"dataset_name": "toy"})
    run_id = start_result["run_id"]
    for _ in range(50):
        status = await tools.execute_tool("get_run_status", {"run_id": run_id})
        if status["status"] == "completed":
            break
        await asyncio.sleep(0.02)
    assert status["status"] == "completed"

    result = await tools.execute_tool("cancel_run", {"run_id": run_id})
    assert result["status"] == "completed"
    assert "no action taken" in result["message"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
