"""
Unit tests for the session/dataset/analysis query functions in
synthony.api.database, independent of the HTTP layer.

database.py uses module-level globals (_engine, _SessionLocal) for its
connection pool, so each test points them at an isolated temp SQLite file
via init_database() and restores the prior globals afterward -- this keeps
the test order-independent regardless of what else runs in the same pytest
session (e.g. functional tests that initialize the real app DB).
"""

import tempfile
from pathlib import Path

import pytest

from synthony.api import database as db_module
from synthony.api.database import (
    count_datasets,
    count_sessions,
    create_analysis,
    create_dataset,
    create_session,
    delete_session_by_id,
    get_analysis_with_relations,
    get_session_with_details,
    init_database,
    list_sessions,
)


@pytest.fixture
def isolated_db():
    """Point database.py's global connection at a fresh temp SQLite file,
    restoring the prior state afterward."""
    prior_engine, prior_session_local = db_module._engine, db_module._SessionLocal
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        init_database(f"sqlite:///{db_path}")
        yield
    db_module._engine, db_module._SessionLocal = prior_engine, prior_session_local


def _make_session_with_dataset(retention_days: int = 30):
    session = create_session("127.0.0.1", "pytest", retention_days=retention_days)
    dataset = create_dataset(
        session_id=session.session_id,
        filename="test.csv",
        file_path="/tmp/test.csv",
        file_size=42,
        format="csv",
    )
    return session, dataset


class TestListAndCountSessions:
    def test_list_sessions_empty(self, isolated_db):
        assert list_sessions() == []
        assert count_sessions() == 0

    def test_list_sessions_returns_created(self, isolated_db):
        session, _ = _make_session_with_dataset()
        sessions = list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == session.session_id
        assert count_sessions() == 1

    def test_list_sessions_excludes_expired_by_default(self, isolated_db):
        _make_session_with_dataset(retention_days=-1)  # already expired
        assert list_sessions() == []
        assert count_sessions() == 0
        assert len(list_sessions(include_expired=True)) == 1
        assert count_sessions(include_expired=True) == 1

    def test_list_sessions_pagination(self, isolated_db):
        for _ in range(5):
            _make_session_with_dataset()
        assert len(list_sessions(limit=2)) == 2
        assert len(list_sessions(limit=2, offset=4)) == 1
        assert count_sessions() == 5


class TestCountDatasets:
    def test_count_datasets(self, isolated_db):
        assert count_datasets() == 0
        _make_session_with_dataset()
        _make_session_with_dataset()
        assert count_datasets() == 2


class TestGetSessionWithDetails:
    def test_zero_dataset_session(self, isolated_db):
        session = create_session("127.0.0.1", "pytest")
        detail = get_session_with_details(session.session_id)
        assert detail["session_id"] == session.session_id
        assert detail["datasets"] == []

    def test_nested_datasets_and_analyses(self, isolated_db):
        session, dataset = _make_session_with_dataset()
        create_analysis(dataset.dataset_id, '{"row_count": 5}', "{}")

        detail = get_session_with_details(session.session_id)
        assert len(detail["datasets"]) == 1
        assert detail["datasets"][0]["dataset_id"] == dataset.dataset_id
        assert len(detail["datasets"][0]["analyses"]) == 1

    def test_not_found(self, isolated_db):
        assert get_session_with_details("00000000-0000-0000-0000-000000000000") is None


class TestGetAnalysisWithRelations:
    def test_safe_to_dict_after_close(self, isolated_db):
        _, dataset = _make_session_with_dataset()
        analysis = create_analysis(dataset.dataset_id, '{"row_count": 5}', "{}")

        fetched = get_analysis_with_relations(analysis.analysis_id)
        # Would raise DetachedInstanceError on a plain get_analysis() result
        # if it touched system_prompt without eager loading.
        as_dict = fetched.to_dict()
        assert as_dict["analysis_id"] == analysis.analysis_id
        assert as_dict["prompt_version"] is None

    def test_not_found(self, isolated_db):
        assert get_analysis_with_relations("00000000-0000-0000-0000-000000000000") is None


class TestDeleteSessionById:
    def test_deletes_session_and_cascades(self, isolated_db):
        session, dataset = _make_session_with_dataset()
        create_analysis(dataset.dataset_id, '{"row_count": 5}', "{}")

        result = delete_session_by_id(session.session_id)
        assert result == {"datasets_deleted": 1, "analyses_deleted": 1}
        assert get_session_with_details(session.session_id) is None
        assert count_sessions() == 0
        assert count_datasets() == 0

    def test_not_found_returns_none(self, isolated_db):
        assert delete_session_by_id("00000000-0000-0000-0000-000000000000") is None
