"""MLflow Tracking Utilities (Phase 6).

Provides centralized configuration, run management, and metric/artifact logging
for the 3-tier energy demand forecasting pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger("eia.mlflow")

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
DEFAULT_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "energy-demand-forecasting")


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> str:
    """Initialize MLflow tracking URI and active experiment.

    Returns the experiment ID.
    """
    uri = tracking_uri or DEFAULT_TRACKING_URI
    exp_name = experiment_name or DEFAULT_EXPERIMENT_NAME

    mlflow.set_tracking_uri(uri)
    logger.info("Configured MLflow tracking URI: %s", uri)

    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
        logger.info("Created new MLflow experiment '%s' (ID: %s)", exp_name, exp_id)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(exp_name)
    return str(exp_id)


def log_tier_metrics(
    tier: str,
    metrics: dict[str, Any],
    prefix: str = "",
    step: int | None = None,
) -> None:
    """Log standard headline metrics (MAE, SMAPE, WMAPE, R2) for a given tier.

    Parameters
    ----------
    tier:
        'Large', 'Medium', 'Small', or 'Overall'.
    metrics:
        Dict containing metric values (e.g. {'mae': 46276.7, 'smape': 3.39, ...}).
    prefix:
        Optional additional prefix (e.g. 'eia_' or 'naive_').
    step:
        Optional step number for time-series logging.
    """
    p = f"{prefix}{tier.lower()}_"
    flat_metrics: dict[str, float] = {}

    for k in ["mae", "smape", "wmape", "r2"]:
        if k in metrics and metrics[k] is not None:
            try:
                flat_metrics[f"{p}{k}"] = float(metrics[k])
            except (ValueError, TypeError):
                continue

    if "n_samples" in metrics and metrics["n_samples"] is not None:
        flat_metrics[f"{p}n_samples"] = float(metrics["n_samples"])

    if flat_metrics:
        mlflow.log_metrics(flat_metrics, step=step)


def log_full_evaluation(
    eval_model: dict[str, dict[str, Any]],
    eval_eia: dict[str, dict[str, Any]],
    eval_naive: dict[str, dict[str, Any]],
    step: int | None = None,
) -> None:
    """Log full multi-tier evaluation comparing Model vs. EIA vs. Naive Baseline."""
    for tier in ["Large", "Medium", "Small", "Overall"]:
        if tier in eval_model:
            log_tier_metrics(tier, eval_model[tier], prefix="model_", step=step)
        if tier in eval_eia:
            log_tier_metrics(tier, eval_eia[tier], prefix="eia_", step=step)
        if tier in eval_naive:
            log_tier_metrics(tier, eval_naive[tier], prefix="naive_", step=step)
