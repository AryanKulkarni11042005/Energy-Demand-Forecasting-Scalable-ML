"""Offline unit tests for monitoring endpoints in FastAPI serving layer (Phase 12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from src.serving.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestServingMonitoringEndpoints:
    def test_monitoring_report_404_when_missing(self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        non_existent = tmp_path / "non_existent.html"
        monkeypatch.setattr("src.serving.app.MONITORING_HTML", non_existent)

        response = client.get("/monitoring/report")
        assert response.status_code == 404
        assert "Evidently Monitoring Report Not Generated Yet" in response.text

    def test_monitoring_report_200_when_present(self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mock_html = tmp_path / "drift_report.html"
        mock_html.write_text("<html><body><h1>Evidently Test Report</h1></body></html>", encoding="utf-8")
        monkeypatch.setattr("src.serving.app.MONITORING_HTML", mock_html)

        response = client.get("/monitoring/report")
        assert response.status_code == 200
        assert "Evidently Test Report" in response.text

    def test_monitoring_status_not_available_when_missing(self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        non_existent = tmp_path / "missing.json"
        monkeypatch.setattr("src.serving.app.MONITORING_JSON", non_existent)

        response = client.get("/api/monitoring/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_available"

    def test_monitoring_status_available_when_present(self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mock_json = tmp_path / "drift_metrics.json"
        mock_json.write_text(
            json.dumps({
                "dataset_drift_detected": False,
                "total_features_evaluated": 10,
                "drifted_features_count": 0,
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr("src.serving.app.MONITORING_JSON", mock_json)

        response = client.get("/api/monitoring/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "available"
        assert data["dataset_drift_detected"] is False
        assert data["total_features_evaluated"] == 10
