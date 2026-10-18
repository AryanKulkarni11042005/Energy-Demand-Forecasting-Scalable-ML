"""Offline unit tests for Phase 12 Evidently AI monitoring.

Runs 100% offline using synthetic in-memory DataFrames without network or EIA API access.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import DriftSummary, ModelMonitor


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    records = []
    for d in dates:
        for resp in ["PJM", "ERCO"]:
            records.append({
                "target_date": d,
                "forecast_date": d - pd.Timedelta(days=1),
                "respondent": resp,
                "target": 1000.0 + float(np.random.normal(0, 50)),
                "prediction": 990.0 + float(np.random.normal(0, 50)),
                "demand_lag_1": 1000.0 + float(np.random.normal(0, 30)),
                "demand_lag_7": 1000.0 + float(np.random.normal(0, 30)),
                "demand_rolling_mean_7": 1000.0 + float(np.random.normal(0, 20)),
                "demand_rolling_std_7": 20.0,
                "trend_slope_7d": 0.5,
                "is_holiday": 0,
            })
    return pd.DataFrame(records)


class TestEvidentlyMonitoring:
    def test_prepare_datasets_temporal_separation(self, synthetic_features_df: pd.DataFrame):
        monitor = ModelMonitor()
        ref, curr = monitor.prepare_datasets(
            synthetic_features_df,
            reference_days=60,
            current_days=30,
        )
        assert len(ref) > 0
        assert len(curr) > 0
        assert ref["target_date"].max() < curr["target_date"].min()

    def test_prepare_datasets_validates_input(self):
        monitor = ModelMonitor()
        with pytest.raises(ValueError, match="DataFrame must contain non-empty"):
            monitor.prepare_datasets(pd.DataFrame())

    def test_build_report_structure(self, synthetic_features_df: pd.DataFrame):
        monitor = ModelMonitor()
        ref, curr = monitor.prepare_datasets(synthetic_features_df, 60, 30)
        report, summary = monitor.build_report(ref, curr)

        assert isinstance(summary, DriftSummary)
        assert summary.total_features_evaluated > 0
        assert isinstance(summary.dataset_drift_detected, bool)
        assert summary.reference_start_date != "unknown"

    def test_drift_detected_on_shifted_distribution(self, synthetic_features_df: pd.DataFrame):
        monitor = ModelMonitor(drift_share_threshold=0.20)
        ref, curr = monitor.prepare_datasets(synthetic_features_df, 60, 30)

        # Deliberately shock current dataset features
        curr_shocked = curr.copy()
        curr_shocked["demand_lag_1"] = curr_shocked["demand_lag_1"] * 50.0
        curr_shocked["demand_rolling_mean_7"] = curr_shocked["demand_rolling_mean_7"] * 50.0

        _, summary = monitor.build_report(ref, curr_shocked)
        assert summary.drifted_features_count >= 1

    def test_save_reports_outputs_html_and_json(self, tmp_path: Path, synthetic_features_df: pd.DataFrame):
        monitor = ModelMonitor()
        ref, curr = monitor.prepare_datasets(synthetic_features_df, 60, 30)
        report, summary = monitor.build_report(ref, curr)

        html_out = tmp_path / "drift_report.html"
        json_out = tmp_path / "drift_metrics.json"

        monitor.save_reports(report, summary, html_out, json_out)

        assert html_out.exists()
        assert html_out.stat().st_size > 500
        assert "<html" in html_out.read_text(encoding="utf-8").lower()

        assert json_out.exists()
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert "dataset_drift_detected" in data
        assert "drifted_features_count" in data
        assert data["total_features_evaluated"] > 0
