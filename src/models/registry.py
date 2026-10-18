"""MLflow Model Registry interface and version lifecycle management (Phase 7).

Manages model versions and aliases across the 3 distinct tier models:
  - energy-demand-forecast-large
  - energy-demand-forecast-medium
  - energy-demand-forecast-small

Standard aliases:
  - @candidate: freshly trained model pending validation / promotion
  - @champion: currently active production model
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.models.tier_model import TierModel

logger = logging.getLogger("eia.registry")

REGISTERED_MODEL_NAMES: dict[str, str] = {
    "Large": "energy-demand-forecast-large",
    "Medium": "energy-demand-forecast-medium",
    "Small": "energy-demand-forecast-small",
}


def get_client(tracking_uri: str | None = None) -> MlflowClient:
    """Get an initialized MlflowClient instance."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def ensure_registered_model_exists(client: MlflowClient, model_name: str, tier: str) -> None:
    """Create registered model in MLflow catalog if it does not yet exist."""
    try:
        client.create_registered_model(
            name=model_name,
            description=f"Next-day energy demand forecasting model for {tier} tier balancing authorities.",
        )
        logger.info("Created registered model '%s'", model_name)
    except Exception:
        # Already exists
        pass


def register_tier_model(
    tier: str,
    model_path: Path,
    eval_metrics: dict[str, Any] | None = None,
    alias: str = "candidate",
    tracking_uri: str | None = None,
) -> str:
    """Log and register a TierModel artifact in the MLflow Model Registry.

    Parameters
    ----------
    tier:
        'Large', 'Medium', or 'Small'.
    model_path:
        Path to local .joblib artifact.
    eval_metrics:
        Optional metrics dictionary (e.g. WMAPE, MAE) to record with the run.
    alias:
        Initial alias to attach (default 'candidate').
    tracking_uri:
        Optional MLflow tracking URI override.

    Returns
    -------
    str: Registered model version number.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = get_client(tracking_uri)
    reg_name = REGISTERED_MODEL_NAMES[tier]
    ensure_registered_model_exists(client, reg_name, tier)

    # Ensure an active experiment exists in this tracking store
    exp_name = "energy-demand-models"
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        try:
            exp_id = client.create_experiment(exp_name)
        except Exception:
            exp_id = "0"
    else:
        exp_id = exp.experiment_id

    with mlflow.start_run(experiment_id=exp_id, run_name=f"register_{tier.lower()}_model") as run:
        if eval_metrics:
            flat = {f"{tier.lower()}_{k}": float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(flat)

        mlflow.log_artifact(str(model_path), artifact_path="model")
        artifact_uri = f"{run.info.artifact_uri}/model/{model_path.name}"

        mv = client.create_model_version(
            name=reg_name,
            source=artifact_uri,
            run_id=run.info.run_id,
            description=f"Artifact: {model_path.name}",
        )

        client.set_registered_model_alias(name=reg_name, alias=alias, version=mv.version)
        logger.info("Registered %s model version %s with alias @%s", reg_name, mv.version, alias)
        return str(mv.version)


def register_all_tier_models(
    models_dir: Path,
    eval_summary: dict[str, Any] | None = None,
    alias: str = "candidate",
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Register all 3 tier models from a directory into the registry."""
    registered: dict[str, str] = {}
    for tier in ["Large", "Medium", "Small"]:
        model_path = models_dir / f"model_tier_{tier.lower()}.joblib"
        if not model_path.exists():
            logger.warning("Artifact %s not found for tier %s; skipping registration", model_path, tier)
            continue

        metrics = None
        if eval_summary and "lightgbm_models" in eval_summary:
            metrics = eval_summary["lightgbm_models"].get(tier)

        version = register_tier_model(
            tier=tier,
            model_path=model_path,
            eval_metrics=metrics,
            alias=alias,
            tracking_uri=tracking_uri,
        )
        registered[tier] = version

    return registered


def promote_registered_tier_models(
    from_alias: str = "candidate",
    to_alias: str = "champion",
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Promote candidate models to champion alias in the MLflow Model Registry."""
    client = get_client(tracking_uri)
    promoted: dict[str, str] = {}

    for tier, reg_name in REGISTERED_MODEL_NAMES.items():
        try:
            mv = client.get_model_version_by_alias(name=reg_name, alias=from_alias)
            client.set_registered_model_alias(name=reg_name, alias=to_alias, version=mv.version)
            try:
                client.delete_registered_model_alias(name=reg_name, alias=from_alias)
            except Exception:
                pass
            promoted[tier] = str(mv.version)
            logger.info("Promoted %s v%s to @%s", reg_name, mv.version, to_alias)
        except Exception as e:
            logger.warning("Could not promote %s from @%s: %s", reg_name, from_alias, e)

    return promoted


def load_tier_model_from_registry(
    tier: str,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> TierModel | None:
    """Load a TierModel instance directly from MLflow Model Registry by alias.

    Falls back to returning None if not found or on error.
    """
    client = get_client(tracking_uri)
    reg_name = REGISTERED_MODEL_NAMES.get(tier)
    if not reg_name:
        return None

    try:
        mv = client.get_model_version_by_alias(name=reg_name, alias=alias)
        download_path = client.download_artifacts(
            run_id=mv.run_id,
            path=f"model/model_tier_{tier.lower()}.joblib",
        )
        return TierModel.load(Path(download_path))
    except Exception as e:
        logger.warning("Could not load %s model from registry alias @%s: %s", tier, alias, e)
        return None
