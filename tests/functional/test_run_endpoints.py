"""Functional tests for /runs/* endpoints: read-only history (Agentic Mode
Phase 2) plus the write side -- POST /runs, POST /runs/{run_id}/cancel
(Agentic Mode Phase 3) -- driven through the real app with a FakeLLMClient
injected via the RunManager the `client` fixture's startup event wires up.

Uses the shared `client` fixture from conftest.py, which runs against the
real app DB (same convention as test_session_endpoints.py) -- so these
assert on specific run_ids being present/absent rather than assuming an
empty table, since other tests in the same session may have written rows.
"""

import time

import pytest

from synthony.api.database import add_run_step, create_run, finish_run, update_run_status
from tests.conftest import FakeLLMClient, FakeRunner, happy_path_llm_responses

# `client` and `sample_csv_file` fixtures are shared via conftest.py


@pytest.fixture
def sample_run(client):
    """A run with one completed step, created directly via database.py
    (no orchestrator exists yet -- that's Phase 3). Depends on `client` so
    the app's startup (init_database()) has already run against the same
    DB this fixture writes to, rather than relying on fixture-order luck.
    """
    run = create_run(mode="advise", llm_provider="openai", llm_model="gpt-4o-mini")
    update_run_status(run.run_id, "reasoning")
    step = add_run_step(run.run_id, agent="profiler", step_name="analyze", input_data={"rows": 100})
    finish_run(run.run_id, "completed", result={"recommended_model": "ARF"})
    return run.run_id, step.step_id


class TestListRuns:
    def test_list_runs_includes_created_run(self, client, sample_run):
        run_id, _ = sample_run
        response = client.get("/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(r["run_id"] == run_id for r in data["runs"])

    def test_list_runs_filters_by_status(self, client, sample_run):
        run_id, _ = sample_run
        response = client.get("/runs", params={"status": "completed"})
        assert response.status_code == 200
        data = response.json()
        assert all(r["status"] == "completed" for r in data["runs"])
        assert any(r["run_id"] == run_id for r in data["runs"])

    def test_list_runs_pagination_params_echoed(self, client):
        response = client.get("/runs", params={"limit": 5, "offset": 0})
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 0


class TestGetRunDetail:
    def test_get_run_detail_includes_steps_and_result(self, client, sample_run):
        run_id, step_id = sample_run
        response = client.get(f"/runs/{run_id}")
        assert response.status_code == 200
        detail = response.json()
        assert detail["run_id"] == run_id
        assert detail["status"] == "completed"
        assert detail["result"] == {"recommended_model": "ARF"}
        assert detail["llm_provider"] == "openai"
        assert len(detail["steps"]) == 1
        assert detail["steps"][0]["step_id"] == step_id
        assert detail["steps"][0]["input"] == {"rows": 100}

    def test_get_run_detail_not_found(self, client):
        response = client.get("/runs/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestListRunSteps:
    def test_list_run_steps(self, client, sample_run):
        run_id, step_id = sample_run
        response = client.get(f"/runs/{run_id}/steps")
        assert response.status_code == 200
        steps = response.json()
        assert len(steps) == 1
        assert steps[0]["step_id"] == step_id
        assert steps[0]["agent"] == "profiler"
        assert steps[0]["step_name"] == "analyze"

    def test_list_run_steps_not_found(self, client):
        response = client.get("/runs/00000000-0000-0000-0000-000000000000/steps")
        assert response.status_code == 404


def _poll_until_terminal(client, run_id, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    detail = None
    while time.monotonic() < deadline:
        detail = client.get(f"/runs/{run_id}").json()
        if detail["status"] in ("completed", "failed", "cancelled"):
            return detail
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach terminal status in {timeout_s}s: {detail}")


@pytest.fixture
def analyzed_dataset_id(client, sample_csv_file):
    response = client.post("/analyze", files={"file": sample_csv_file})
    assert response.status_code == 200, response.text
    return response.json()["dataset_id"]


class TestStartRun:
    """POST /runs -> real Orchestrator/RecommenderAgent through the actual
    app, with an injected FakeLLMClient (via the real RunManager's
    llm_client_factory seam) so the reasoning loop is deterministic."""

    def test_post_runs_completes_advise_mode(self, client, analyzed_dataset_id, monkeypatch):
        from synthony.api import run_endpoints

        fake = FakeLLMClient(happy_path_llm_responses())
        assert run_endpoints._run_manager is not None
        monkeypatch.setattr(run_endpoints._run_manager, "_llm_client_factory", lambda cfg: fake)

        response = client.post("/runs", json={"dataset_id": analyzed_dataset_id, "mode": "advise"})
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]

        detail = _poll_until_terminal(client, run_id)
        assert detail["status"] == "completed"

        steps_response = client.get(f"/runs/{run_id}/steps")
        step_names = [s["step_name"] for s in steps_response.json()]
        assert step_names == ["analyze", "reason_turn_1", "reason_turn_2", "reason_turn_3"]

    def test_post_runs_completes_full_mode(self, client, analyzed_dataset_id, monkeypatch):
        """Injects a FakeRunner via the RunManager's runner_factory seam
        (mirrors the llm_client_factory seam above) so this stays a fast,
        deterministic functional test -- no real synthcity subprocess."""
        from synthony.api import run_endpoints

        fake_llm = FakeLLMClient(happy_path_llm_responses())
        fake_runner = FakeRunner("success")
        assert run_endpoints._run_manager is not None
        monkeypatch.setattr(run_endpoints._run_manager, "_llm_client_factory", lambda cfg: fake_llm)
        monkeypatch.setattr(run_endpoints._run_manager, "_runner_factory", lambda cfg: fake_runner)

        response = client.post("/runs", json={"dataset_id": analyzed_dataset_id, "mode": "full"})
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]

        detail = _poll_until_terminal(client, run_id)
        assert detail["status"] == "completed"
        assert detail["result"]["met_threshold"] is True

    def test_post_runs_dataset_not_found(self, client):
        response = client.post("/runs", json={"dataset_id": "does-not-exist", "mode": "advise"})
        assert response.status_code == 404


class TestCancelRun:
    def test_cancel_already_terminal_run_is_a_noop(self, client, sample_run):
        run_id, _ = sample_run  # sample_run fixture creates a "completed" run
        response = client.post(f"/runs/{run_id}/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    def test_cancel_unknown_run_returns_404(self, client):
        response = client.post("/runs/00000000-0000-0000-0000-000000000000/cancel")
        assert response.status_code == 404
