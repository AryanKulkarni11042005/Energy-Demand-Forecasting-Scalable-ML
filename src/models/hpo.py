"""Optuna hyperparameter tuning engine per tier (Phase 5).

Optimizes LightGBM hyperparameters for each tier (Large, Medium, Small)
using Bayesian optimization (TPE) over walk-forward chronological folds,
minimizing validation WMAPE (Weighted MAPE).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.evaluation.metrics import compute_wmape
from src.features.build_features import FEATURE_COLUMNS
from src.models.cv import generate_walk_forward_folds
from src.models.tier_model import TierModel

logger = logging.getLogger("eia.hpo")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tier_hpo_objective(
    trial: optuna.Trial,
    tier: str,
    train_val_df: pd.DataFrame,
    n_splits: int = 3,
    feature_cols: list[str] = FEATURE_COLUMNS,
    target_col: str = "target",
) -> float:
    """Evaluate candidate hyperparameters across chronological walk-forward folds."""
    params: dict[str, Any] = {
        "objective": "regression_l1",
        "metric": "l1",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    tier_df = train_val_df[train_val_df["tier"] == tier].dropna(subset=[target_col]).copy()
    if tier_df.empty:
        raise ValueError(f"No rows available for tier {tier} in HPO dataset")

    folds = generate_walk_forward_folds(tier_df, n_splits=n_splits)
    fold_wmapes: list[float] = []

    for fold in folds:
        tm = TierModel(tier=tier, feature_cols=feature_cols)
        tm.fit(fold.train_df, fold.val_df, target_col=target_col, params=params)

        preds = tm.predict(fold.val_df, enable_fallback=True)
        y_true = fold.val_df[target_col].to_numpy()

        valid = (~np.isnan(y_true)) & (~np.isnan(preds))
        if not valid.any():
            continue

        wmape = compute_wmape(y_true[valid], preds[valid])
        fold_wmapes.append(wmape)

    if not fold_wmapes:
        return float("inf")

    return float(np.mean(fold_wmapes))


def run_tier_hpo(
    tier: str,
    train_val_df: pd.DataFrame,
    n_trials: int = 30,
    n_splits: int = 3,
    storage_url: str | None = None,
    study_name: str | None = None,
) -> tuple[dict[str, Any], float]:
    """Execute an Optuna study for one tier and return (best_params, best_wmape).

    Parameters
    ----------
    tier:
        'Large', 'Medium', or 'Small'.
    train_val_df:
        Combined chronological train + val DataFrame (test set held out!).
    n_trials:
        Number of Optuna trials to run.
    n_splits:
        Number of walk-forward folds.
    storage_url:
        Optional SQLite/RDBMS connection string (e.g. 'sqlite:///optuna.db').
    study_name:
        Optional custom study identifier.
    """
    if study_name is None:
        study_name = f"optuna_tier_{tier.lower()}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    logger.info(
        "Starting Optuna HPO for tier %s (%d trials, %d walk-forward splits)...",
        tier, n_trials, n_splits,
    )

    study.optimize(
        lambda trial: tier_hpo_objective(trial, tier, train_val_df, n_splits=n_splits),
        n_trials=n_trials,
    )

    best_params = study.best_params
    best_value = study.best_value

    logger.info(
        "Completed HPO for %s: Best Mean Val WMAPE=%.2f%% with params: %s",
        tier, best_value, best_params,
    )
    return best_params, best_value


def save_tuned_params(params: dict[str, dict[str, Any]], filepath: Path) -> None:
    """Save tuned hyperparameters by tier to JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(params, indent=2))
    logger.info("Saved tuned parameters to %s", filepath)


def load_tuned_params(filepath: Path) -> dict[str, dict[str, Any]]:
    """Load tuned hyperparameters from JSON file."""
    if not filepath.exists():
        return {}
    try:
        return json.loads(filepath.read_text())
    except Exception as e:
        logger.warning("Could not read tuned parameters from %s: %s", filepath, e)
        return {}
