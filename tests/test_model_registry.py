"""Unit tests for Phase 7 MLflow Model Registry and Versioning.

Runs offline without network or external database dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import (
    REGISTERED_MODEL_NAMES,
    load_tier_model_from_registry,
    promote_registered_tier_models,
    register_all_tier_models,
    register_tier_model,
)
from src.models.tier_model import TierModel


@pytest.fixture
def isolated_registry(tmp_path: Path):
    """Fixture providing an isolated SQLite MLflow tracking and artifact store."""
    db_path = tmp_path / "test_registry.db"
    uri = f"sqlite:///{db_path}"
    return uri


class MockLGBM:
    """Top-level mock regressor for pickling in tests."""
    def predict(self, X):
        return np.ones(len(X)) * 50000.0


@pytest.fixture
def sample_tier_models(tmp_path: Path) -> Path:
    """Create and persist dummy TierModel artifacts for testing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for tier in ["Large", "Medium", "Small"]:
        tm = TierModel(tier=tier, feature_cols=["demand_t"])
        tm.model = MockLGBM()
        tm.save(models_dir / f"model_tier_{tier.lower()}.joblib")

    return models_dir


class TestModelRegistry:
    def test_register_tier_model_and_alias(self, isolated_registry: str, sample_tier_models: Path):
        large_path = sample_tier_models / "model_tier_large.joblib"
        version = register_tier_model(
            tier="Large",
            model_path=large_path,
            eval_metrics={"wmape": 3.48, "mae": 45000.0},
            alias="candidate",
            tracking_uri=isolated_registry,
        )

        assert version == "1"

        client = MlflowClient(tracking_uri=isolated_registry)
        reg_name = REGISTERED_MODEL_NAMES["Large"]
        mv = client.get_model_version_by_alias(name=reg_name, alias="candidate")
        assert mv is not None
        assert str(mv.version) == "1"

    def test_promote_registered_models_to_champion(self, isolated_registry: str, sample_tier_models: Path):
        # Register model as candidate
        large_path = sample_tier_models / "model_tier_large.joblib"
        register_tier_model(
            tier="Large",
            model_path=large_path,
            alias="candidate",
            tracking_uri=isolated_registry,
        )

        # Promote to champion
        promoted = promote_registered_tier_models(
            from_alias="candidate",
            to_alias="champion",
            tracking_uri=isolated_registry,
        )

        assert "Large" in promoted
        assert promoted["Large"] == "1"

        client = MlflowClient(tracking_uri=isolated_registry)
        reg_name = REGISTERED_MODEL_NAMES["Large"]
        champion_mv = client.get_model_version_by_alias(name=reg_name, alias="champion")
        assert str(champion_mv.version) == "1"

    def test_load_tier_model_from_registry(self, isolated_registry: str, sample_tier_models: Path):
        # Register and promote
        large_path = sample_tier_models / "model_tier_large.joblib"
        register_tier_model(
            tier="Large",
            model_path=large_path,
            alias="champion",
            tracking_uri=isolated_registry,
        )

        loaded_model = load_tier_model_from_registry(
            tier="Large",
            alias="champion",
            tracking_uri=isolated_registry,
        )

        assert loaded_model is not None
        assert isinstance(loaded_model, TierModel)
        assert loaded_model.tier == "Large"
        assert loaded_model.feature_cols == ["demand_t"]

    def test_register_all_tier_models(self, isolated_registry: str, sample_tier_models: Path):
        summary = {
            "lightgbm_models": {
                "Large": {"wmape": 3.48},
                "Medium": {"wmape": 4.59},
                "Small": {"wmape": 4.12},
            }
        }
        registered = register_all_tier_models(
            models_dir=sample_tier_models,
            eval_summary=summary,
            alias="candidate",
            tracking_uri=isolated_registry,
        )

        assert set(registered.keys()) == {"Large", "Medium", "Small"}
        assert registered["Large"] == "1"
        assert registered["Medium"] == "1"
        assert registered["Small"] == "1"
