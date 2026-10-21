"""Unit tests for the run/run_step persistence functions in
synthony.api.database (Agentic Mode Phase 2), independent of the HTTP layer.

Uses the shared `isolated_db` fixture (tests/conftest.py): each test points
database.py's module-level connection at a fresh temp SQLite file so tests
are order-independent.
"""

import json

import pytest

from synthony.api.database import (
    accumulate_run_usage,
    add_run_step,
    count_runs,
    create_run,
    delete_run,
    finish_run,
    finish_run_step,
    get_run,
    get_run_with_steps,
    list_run_steps,
    list_runs,
    update_run_status,
)


class TestCreateRun:
    def test_create_run_defaults_to_pending(self, isolated_db):
        run = create_run(mode="advise")
        assert run.status == "pending"
        assert run.mode == "advise"
        assert run.started_at is None
        assert run.finished_at is None
        assert run.total_input_tokens == 0
        assert run.total_cost_usd == 0.0

    def test_create_run_stores_config_and_llm_fields(self, isolated_db):
        run = create_run(
            mode="full",
            owner="pid:1234",
            config={"quality_threshold": 0.75},
            llm_provider="anthropic",
            llm_model="claude-haiku-4-5-20251001",
        )
        assert run.owner == "pid:1234"
        assert run.llm_provider == "anthropic"
        detail = get_run_with_steps(run.run_id)
        assert detail["config"] == {"quality_threshold": 0.75}


class TestUpdateAndFinishRun:
    def test_update_run_status_stamps_started_at_once(self, isolated_db):
        run = create_run(mode="advise")
        assert run.started_at is None

        updated = update_run_status(run.run_id, "profiling")
        assert updated.status == "profiling"
        assert updated.started_at is not None
        first_started_at = updated.started_at

        updated_again = update_run_status(run.run_id, "reasoning")
        assert updated_again.started_at == first_started_at

    def test_update_run_status_missing_run_returns_none(self, isolated_db):
        assert update_run_status("nonexistent", "reasoning") is None

    def test_finish_run_stamps_finished_at_and_result(self, isolated_db):
        run = create_run(mode="advise")
        finished = finish_run(run.run_id, "completed", result={"recommended_model": "ARF"})
        assert finished.status == "completed"
        assert finished.finished_at is not None
        detail = get_run_with_steps(run.run_id)
        assert detail["result"] == {"recommended_model": "ARF"}

    def test_finish_run_with_error(self, isolated_db):
        run = create_run(mode="advise")
        finished = finish_run(run.run_id, "failed", error_message="LLM budget exhausted")
        assert finished.status == "failed"
        assert finished.error_message == "LLM budget exhausted"


class TestRunSteps:
    def test_add_and_finish_run_step(self, isolated_db):
        run = create_run(mode="advise")
        step = add_run_step(
            run.run_id, agent="profiler", step_name="analyze", input_data={"path": "x.csv"}
        )
        assert step.status == "running"
        assert step.finished_at is None

        finished = finish_run_step(
            step.step_id,
            status="completed",
            output_data={"row_count": 100},
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        )
        assert finished.status == "completed"
        assert finished.finished_at is not None
        assert finished.duration_ms is not None and finished.duration_ms >= 0
        assert finished.input_tokens == 10
        assert finished.cost_usd == 0.001

    def test_finish_run_step_missing_returns_none(self, isolated_db):
        assert finish_run_step(999999, status="completed") is None

    def test_list_run_steps_execution_order(self, isolated_db):
        run = create_run(mode="advise")
        s1 = add_run_step(run.run_id, agent="profiler", step_name="analyze")
        s2 = add_run_step(run.run_id, agent="recommender", step_name="reason")
        steps = list_run_steps(run.run_id)
        assert [s.step_id for s in steps] == [s1.step_id, s2.step_id]

    def test_get_run_with_steps_includes_step_dicts(self, isolated_db):
        run = create_run(mode="advise")
        add_run_step(run.run_id, agent="profiler", step_name="analyze")
        detail = get_run_with_steps(run.run_id)
        assert detail["run_id"] == run.run_id
        assert len(detail["steps"]) == 1
        assert detail["steps"][0]["agent"] == "profiler"

    def test_step_io_truncated_beyond_16kb_degrades_gracefully(self, isolated_db):
        run = create_run(mode="advise")
        huge = {"blob": "x" * 20_000}
        add_run_step(run.run_id, agent="profiler", step_name="analyze", input_data=huge)
        # Truncation lands mid-JSON, so the stored value is no longer valid
        # JSON -- get_run_with_steps must not raise, and should surface the
        # raw truncated string (marked) rather than losing the field.
        detail = get_run_with_steps(run.run_id)
        stored_input = detail["steps"][0]["input"]
        assert isinstance(stored_input, str)
        assert stored_input.endswith("...<truncated>")
        assert len(stored_input.encode("utf-8")) < len(json.dumps(huge).encode("utf-8"))


class TestUsageAccumulation:
    def test_accumulate_run_usage_rolls_up_totals(self, isolated_db):
        run = create_run(mode="advise")
        accumulate_run_usage(run.run_id, input_tokens=100, output_tokens=20, cost_usd=0.01)
        accumulate_run_usage(run.run_id, input_tokens=50, output_tokens=10, cost_usd=0.005)
        updated = get_run(run.run_id)
        assert updated.total_input_tokens == 150
        assert updated.total_output_tokens == 30
        assert updated.total_cost_usd == pytest.approx(0.015)

    def test_accumulate_run_usage_missing_run_returns_none(self, isolated_db):
        assert accumulate_run_usage("nonexistent", input_tokens=1) is None


class TestListAndCountRuns:
    def test_list_runs_empty(self, isolated_db):
        assert list_runs() == []
        assert count_runs() == 0

    def test_list_runs_most_recent_first(self, isolated_db):
        r1 = create_run(mode="advise")
        r2 = create_run(mode="full")
        runs = list_runs()
        assert [r.run_id for r in runs] == [r2.run_id, r1.run_id]
        assert count_runs() == 2

    def test_list_runs_filters_by_status(self, isolated_db):
        r1 = create_run(mode="advise")
        create_run(mode="advise")
        update_run_status(r1.run_id, "completed")
        assert count_runs(status="completed") == 1
        assert [r.run_id for r in list_runs(status="completed")] == [r1.run_id]


class TestDeleteRunCascade:
    def test_delete_run_cascades_to_steps(self, isolated_db):
        run = create_run(mode="advise")
        add_run_step(run.run_id, agent="profiler", step_name="analyze")
        add_run_step(run.run_id, agent="recommender", step_name="reason")
        assert len(list_run_steps(run.run_id)) == 2

        assert delete_run(run.run_id) is True

        assert get_run(run.run_id) is None
        assert list_run_steps(run.run_id) == []

    def test_delete_run_missing_returns_false(self, isolated_db):
        assert delete_run("nonexistent") is False


class TestSchemaMetaInit:
    def test_init_database_survives_concurrent_worker_seeding_schema_version(
        self, isolated_db, tmp_path
    ):
        """uvicorn --workers N runs init_database() in every worker at once.
        Reproduce the losing interleaving deterministically: another worker
        commits the schema_version row right after this worker's existence
        check came back empty, but before this worker's own insert."""
        from sqlalchemy import create_engine, event, select
        from sqlalchemy.orm import Session

        from synthony.api import database as db_module

        url = f"sqlite:///{tmp_path / 'race.db'}"
        other_worker = create_engine(url)
        fired = []

        def other_worker_wins_race(orm_execute_state):
            if fired or "schema_meta" not in str(orm_execute_state.statement):
                return None
            result = orm_execute_state.invoke_statement()  # this worker's check: no row yet
            with other_worker.begin() as conn:
                conn.execute(
                    db_module.SchemaMeta.__table__.insert().values(
                        key="schema_version", value=db_module._CURRENT_SCHEMA_VERSION
                    )
                )
            fired.append(True)
            return result

        event.listen(Session, "do_orm_execute", other_worker_wins_race)
        try:
            db_module.init_database(url)
        finally:
            event.remove(Session, "do_orm_execute", other_worker_wins_race)

        assert fired, "race hook never ran; test is not exercising the interleaving"
        with other_worker.connect() as conn:
            rows = conn.execute(select(db_module.SchemaMeta.__table__)).all()
        assert [(r.key, r.value) for r in rows] == [
            ("schema_version", db_module._CURRENT_SCHEMA_VERSION)
        ]
        other_worker.dispose()

    def test_init_database_enables_wal_while_another_worker_holds_write_lock(
        self, isolated_db, tmp_path
    ):
        """While another worker is mid-write (holding SQLite's RESERVED lock), SQLite
        fails PRAGMA journal_mode=WAL immediately with 'database is locked' --
        busy_timeout is not consulted for this lock. Hold that lock briefly from a
        second connection, as a concurrently starting uvicorn worker would."""
        import sqlite3
        import threading

        from synthony.api import database as db_module

        path = tmp_path / "locked.db"
        other_worker = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        other_worker.execute("BEGIN IMMEDIATE")
        release = threading.Timer(0.5, lambda: other_worker.execute("ROLLBACK"))
        release.start()
        try:
            db_module.init_database(f"sqlite:///{path}")
        finally:
            release.join()
            other_worker.close()

        with sqlite3.connect(path) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_init_database_creates_every_table_when_another_worker_creates_one_first(
        self, isolated_db, tmp_path
    ):
        """create_all checks which tables are missing, then creates them. If another
        worker creates one in between, CREATE TABLE fails with 'already exists'; the
        remaining tables must still be created (a worker crashed on 'no such table:
        runs' when the guard swallowed that error and stopped)."""
        from sqlalchemy import create_engine, event, inspect

        from synthony.api import database as db_module

        url = f"sqlite:///{tmp_path / 'tables.db'}"
        other_worker = create_engine(url)
        won = []

        def other_worker_creates_first_table(target, connection, tables=(), **kw):
            if won or not tables:
                return
            won.append(tables[0].name)
            tables[0].create(other_worker)

        event.listen(db_module.Base.metadata, "before_create", other_worker_creates_first_table)
        try:
            db_module.init_database(url)
        finally:
            event.remove(db_module.Base.metadata, "before_create", other_worker_creates_first_table)

        assert won, "race hook never ran; test is not exercising the interleaving"
        missing = set(db_module.Base.metadata.tables) - set(inspect(other_worker).get_table_names())
        other_worker.dispose()
        assert not missing, f"tables never created: {sorted(missing)}"
