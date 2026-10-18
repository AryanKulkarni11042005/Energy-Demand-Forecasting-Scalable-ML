"""Unit tests for Phase 5 Walk-Forward Cross-Validation and Optuna HPO.

Runs offline without network or disk dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_features import build_features
from src.models.cv import generate_walk_forward_folds
from src.models.hpo import (
    load_tuned_params,
    run_tier_hpo,
    save_tuned_params,
    tier_hpo_objective,
)


@pytest.fixture
def synthetic_panel_df() -> pd.DataFrame:
    """Generate synthetic panel data for 30 daily periods across 3 tiers."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    for d in dates:
        # Large: PJM
        rows.append({
            "period": d,
            "respondent": "PJM",
            "demand": 600000.0 + 1000.0 * np.sin(d.dayofyear),
            "demand_forecast": 595000.0,
            "net_generation": 590000.0,
            "interchange": 10000.0,
        })
        # Medium: NYIS
        rows.append({
            "period": d,
            "respondent": "NYIS",
            "demand": 150000.0 + 500.0 * np.sin(d.dayofyear),
            "demand_forecast": 149000.0,
            "net_generation": 145000.0,
            "interchange": 5000.0,
        })
        # Small: SCL
        rows.append({
            "period": d,
            "respondent": "SCL",
            "demand": 25000.0 + 100.0 * np.sin(d.dayofyear),
            "demand_forecast": 24800.0,
            "net_generation": 24000.0,
            "interchange": 1000.0,
        })
    df = pd.DataFrame(rows)
    return build_features(df, drop_na_target=True)


class TestTimeSeriesCV:
    def test_walk_forward_folds_temporal_order(self, synthetic_panel_df: pd.DataFrame):
        large_df = synthetic_panel_df[synthetic_panel_df["tier"] == "Large"].copy()
        folds = generate_walk_forward_folds(large_df, n_splits=3, min_train_frac=0.50)

        assert len(folds) == 3

        prev_train_len = 0
        for i, fold in enumerate(folds):
            assert fold.fold_idx == i
            # Strict non-overlap: max train date < min val date
            assert fold.train_df["target_date"].max() < fold.val_df["target_date"].min()
            # Expanding training window
            train_len = len(fold.train_df)
            assert train_len > prev_train_len
            prev_train_len = train_len
            # Val is not empty
            assert len(fold.val_df) > 0

    def test_walk_forward_folds_errors_on_empty(self):
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Cannot generate folds on an empty DataFrame"):
            generate_walk_forward_folds(empty_df)

    def test_walk_forward_folds_errors_on_few_dates(self):
        short_df = pd.DataFrame({
            "target_date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "target": [1, 2, 3],
        })
        with pytest.raises(ValueError, match="Insufficient unique dates"):
            generate_walk_forward_folds(short_df, n_splits=3)


class TestOptunaHPO:
    def test_run_tier_hpo_offline(self, synthetic_panel_df: pd.DataFrame):
        # Run 2 quick trials on Small tier using in-memory Optuna
        best_params, best_wmape = run_tier_hpo(
            tier="Small",
            train_val_df=synthetic_panel_df,
            n_trials=2,
            n_splits=2,
            storage_url=None,  # In-memory
        )

        assert isinstance(best_params, dict)
        assert "learning_rate" in best_params
        assert "num_leaves" in best_params
        assert "max_depth" in best_params
        assert "min_child_samples" in best_params
        assert best_wmape >= 0.0

    def test_save_and_load_tuned_params(self, tmp_path: Path):
        test_file = tmp_path / "tuned_params.json"
        sample_params = {
            "Large": {"learning_rate": 0.04, "num_leaves": 40},
            "Small": {"learning_rate": 0.02, "num_leaves": 20},
        }
        save_tuned_params(sample_params, test_file)
        assert test_file.exists()

        loaded = load_tuned_params(test_file)
        assert loaded == sample_params

        # Loading non-existent file returns empty dict without throwing error
        assert load_tuned_params(tmp_path / "missing.json") == {}
