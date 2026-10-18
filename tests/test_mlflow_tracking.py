"""Unit tests for Phase 6 MLflow Experiment Tracking.

Runs offline without network or external database dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.mlflow_utils import (
    log_full_evaluation,
    log_tier_metrics,
    setup_mlflow,
)


@pytest.fixture
def isolated_mlflow(tmp_path: Path):
    """Fixture providing an isolated SQLite MLflow tracking environment."""
    db_path = tmp_path / "test_tracking.db"
    uri = f"sqlite:///{db_path}"
    exp_name = "test-energy-demand"
    exp_id = setup_mlflow(tracking_uri=uri, experiment_name=exp_name)
    return uri, exp_name, exp_id


class TestMLflowTracking:
    def test_setup_mlflow_creates_experiment(self, isolated_mlflow):
        uri, exp_name, exp_id = isolated_mlflow
        client = MlflowClient(tracking_uri=uri)
        exp = client.get_experiment_by_name(exp_name)
        assert exp is not None
        assert exp.experiment_id == exp_id

    def test_log_tier_metrics(self, isolated_mlflow):
        uri, exp_name, _ = isolated_mlflow
        client = MlflowClient(tracking_uri=uri)

        with mlflow.start_run(run_name="test_tier_run") as run:
            sample_metrics = {
                "mae": 1234.5,
                "smape": 4.25,
                "wmape": 3.80,
                "r2": 0.985,
                "n_samples": 500,
            }
            log_tier_metrics("Large", sample_metrics, prefix="model_")

        run_data = client.get_run(run.info.run_id).data
        assert run_data.metrics["model_large_mae"] == 1234.5
        assert run_data.metrics["model_large_smape"] == 4.25
        assert run_data.metrics["model_large_wmape"] == 3.80
        assert pytest.approx(run_data.metrics["model_large_r2"]) == 0.985
        assert run_data.metrics["model_large_n_samples"] == 500.0

    def test_log_full_evaluation(self, isolated_mlflow):
        uri, exp_name, _ = isolated_mlflow
        client = MlflowClient(tracking_uri=uri)

        eval_model = {
            "Large": {"mae": 1000.0, "smape": 3.0, "wmape": 3.2, "r2": 0.99, "n_samples": 100},
            "Medium": {"mae": 500.0, "smape": 4.0, "wmape": 4.1, "r2": 0.98, "n_samples": 200},
            "Small": {"mae": 50.0, "smape": 5.0, "wmape": 4.5, "r2": 0.97, "n_samples": 300},
            "Overall": {"mae": 450.0, "smape": 4.2, "wmape": 3.9, "r2": 0.99, "n_samples": 600},
        }
        eval_eia = {
            "Large": {"mae": 1200.0, "smape": 3.5, "wmape": 3.6, "r2": 0.98, "n_samples": 100},
            "Medium": {"mae": 600.0, "smape": 5.0, "wmape": 5.2, "r2": 0.96, "n_samples": 200},
            "Small": {"mae": 60.0, "smape": 6.0, "wmape": 5.0, "r2": 0.95, "n_samples": 300},
            "Overall": {"mae": 550.0, "smape": 5.1, "wmape": 4.8, "r2": 0.97, "n_samples": 600},
        }
        eval_naive = {
            "Large": {"mae": 1500.0, "smape": 4.5, "wmape": 4.6, "r2": 0.97, "n_samples": 100},
            "Medium": {"mae": 700.0, "smape": 5.5, "wmape": 5.8, "r2": 0.95, "n_samples": 200},
            "Small": {"mae": 70.0, "smape": 6.5, "wmape": 5.5, "r2": 0.94, "n_samples": 300},
            "Overall": {"mae": 650.0, "smape": 5.8, "wmape": 5.2, "r2": 0.96, "n_samples": 600},
        }

        with mlflow.start_run(run_name="full_eval_run") as run:
            log_full_evaluation(eval_model, eval_eia, eval_naive)

        metrics = client.get_run(run.info.run_id).data.metrics
        assert "model_large_mae" in metrics
        assert "eia_medium_smape" in metrics
        assert "naive_small_wmape" in metrics
        assert "model_overall_r2" in metrics
