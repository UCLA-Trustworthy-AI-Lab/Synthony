"""
Functional tests for /sessions/* and /storage/stats endpoints.

Tests session listing (gated), session detail, file download, analysis
detail, session deletion (DB + filesystem cascade), and storage stats.
"""

import pytest

# `client` and `sample_csv_file` fixtures are shared via conftest.py


@pytest.fixture
def analyzed_session(client, sample_csv_file):
    """Upload + analyze a dataset, returning (session_id, dataset_id, analysis_id)."""
    response = client.post("/analyze", files={"file": sample_csv_file})
    assert response.status_code == 200, response.text
    data = response.json()
    return data["session_id"], data["dataset_id"], data["analysis_id"]


class TestListSessions:
    """GET /sessions is gated behind ENABLE_SESSION_LISTING."""

    def test_disabled_by_default(self, client, monkeypatch):
        monkeypatch.delenv("ENABLE_SESSION_LISTING", raising=False)
        response = client.get("/sessions")
        assert response.status_code == 404

    def test_enabled_via_env_var(self, client, monkeypatch, analyzed_session):
        session_id, _, _ = analyzed_session
        monkeypatch.setenv("ENABLE_SESSION_LISTING", "true")
        response = client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(s["session_id"] == session_id for s in data["sessions"])
        # Bulk IP exposure is intentionally omitted from the list view.
        assert "ip_address" not in data["sessions"][0]


class TestSessionDetail:
    def test_get_session_detail_after_analyze(self, client, analyzed_session):
        session_id, dataset_id, analysis_id = analyzed_session
        response = client.get(f"/sessions/{session_id}")
        assert response.status_code == 200
        detail = response.json()
        assert detail["session_id"] == session_id
        assert len(detail["datasets"]) == 1
        assert detail["datasets"][0]["dataset_id"] == dataset_id
        assert len(detail["datasets"][0]["analyses"]) == 1
        assert detail["datasets"][0]["analyses"][0]["analysis_id"] == analysis_id

    def test_get_session_detail_not_found(self, client):
        response = client.get("/sessions/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestDownloadDatasetFile:
    def test_download_dataset_file(self, client, analyzed_session):
        session_id, dataset_id, _ = analyzed_session
        response = client.get(f"/sessions/{session_id}/data/{dataset_id}")
        assert response.status_code == 200
        assert b"id,value,category" in response.content

    def test_download_wrong_session_404(self, client, analyzed_session):
        _, dataset_id, _ = analyzed_session
        response = client.get(
            f"/sessions/00000000-0000-0000-0000-000000000000/data/{dataset_id}"
        )
        assert response.status_code == 404

    def test_download_session_not_found(self, client):
        response = client.get(
            "/sessions/00000000-0000-0000-0000-000000000000/data/"
            "11111111-1111-1111-1111-111111111111"
        )
        assert response.status_code == 404


class TestAnalysisDetail:
    def test_get_analysis_detail(self, client, analyzed_session):
        session_id, _, analysis_id = analyzed_session
        response = client.get(f"/sessions/{session_id}/analyses/{analysis_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_profile"] is not None
        assert data["dataset_profile"]["row_count"] == 1000
        assert data["column_analysis"] is not None

    def test_get_analysis_detail_wrong_session_404(self, client, analyzed_session):
        _, _, analysis_id = analyzed_session
        response = client.get(
            f"/sessions/00000000-0000-0000-0000-000000000000/analyses/{analysis_id}"
        )
        assert response.status_code == 404

    def test_get_analysis_detail_not_found(self, client, analyzed_session):
        session_id, _, _ = analyzed_session
        response = client.get(
            f"/sessions/{session_id}/analyses/00000000-0000-0000-0000-000000000000"
        )
        assert response.status_code == 404


class TestDeleteSession:
    def test_delete_session(self, client, analyzed_session):
        session_id, dataset_id, _ = analyzed_session

        response = client.delete(f"/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["datasets_deleted"] == 1
        assert data["analyses_deleted"] == 1
        assert data["files_deleted"] >= 1

        # DB cascade confirmed
        assert client.get(f"/sessions/{session_id}").status_code == 404
        # Filesystem cleanup confirmed
        assert client.get(f"/sessions/{session_id}/data/{dataset_id}").status_code == 404

    def test_delete_session_not_found(self, client):
        response = client.delete("/sessions/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestStorageStats:
    def test_storage_stats(self, client, analyzed_session):
        response = client.get("/storage/stats")
        assert response.status_code == 200
        data = response.json()
        for key in (
            "total_size_gb", "storage_limit_gb", "usage_percent",
            "active_sessions", "total_datasets",
            "db_active_sessions", "db_total_datasets",
        ):
            assert key in data
        assert data["db_active_sessions"] >= 1
        assert data["db_total_datasets"] >= 1
